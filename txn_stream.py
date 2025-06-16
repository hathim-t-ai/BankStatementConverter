# -*- coding: utf-8 -*-
"""txn_stream.py
Hybrid (inline / detached / mixed) transaction parser.

This module provides a deterministic *stream* parser that walks through the
physical rows extracted from a PDF bank statement and builds a list of
`Txn` objects on the fly using a small state-machine.  It is designed to be
plug-compatible with the existing `parse_transactions` API by exposing the
utility function `parse_transactions_stream(rows)` which returns a Pandas
DataFrame in the same format.

The algorithm intentionally keeps *zero* bank-specific constants; everything
is inferred at runtime from the structure of the text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import re
import logging

import numpy as np
import pandas as pd

from .text_line_extractor import (
    AMOUNT_RE,
    CURRENCY_RE,
    HEADER_ROW_RE,
    SUMMARY_KEYWORDS,
    TRANSACTION_DATE_RE,
    TRX_START_KW,
    HeaderList,
    _clean_amt,
    classify_amount,
    _adjust_dir_by_delta,
    TIME_RE,
)

logger = logging.getLogger("txn_stream")

__all__ = [
    "Txn",
    "StreamParser",
    "parse_transactions_stream",
]


class ParseState(Enum):
    EXPECT_DESC = "expect_description"
    EXPECT_NUMERIC = "expect_numeric"


@dataclass
class Txn:
    """Lightweight container for a single transaction during parsing."""

    date: Optional[str] = None
    desc_parts: List[str] = field(default_factory=list)
    amt_str: Optional[str] = None
    bal_str: Optional[str] = None

    # Cached helper fields (not persisted in final DF)
    _pending_nums: List[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def description(self) -> str:
        return " ".join(self.desc_parts).strip()

    def add_desc_part(self, txt: str):
        txt = txt.strip()
        if txt:
            self.desc_parts.append(txt)

    def is_complete(self) -> bool:
        return bool(self.description and self.amt_str and self.bal_str)

    # ------------------------------------------------------------------
    # Number attachment helpers
    # ------------------------------------------------------------------
    def attach_numbers(self, nums: List[str]):
        """Attach *two* numbers (amount, balance) to the Txn.

        The parser receives numeric *pairs* in the order they appear on the
        page, but some layouts list the running balance **after** the amount
        (``amount  balance``) while others put it first.  A lightweight
        heuristic works well across statements: the *larger* magnitude token
        is almost always the running balance.  In the (unlikely) event of a
        tie we preserve the original ordering.
        """
        if len(nums) >= 2:
            first, second = nums[-2:]
            try:
                f1 = abs(float(first.replace(",", "")))
                f2 = abs(float(second.replace(",", "")))
            except Exception:
                # Fallback to original order on parse failure
                self.amt_str, self.bal_str = first, second
                return

            if f1 > f2:
                self.bal_str, self.amt_str = first, second
            else:
                self.amt_str, self.bal_str = first, second

    def try_extract_inline_numbers(self, currency: str | None = None):
        """Look for numbers inside the description itself."""
        nums = AMOUNT_RE.findall(self.description)
        if len(nums) >= 2:
            self.attach_numbers([_clean_amt(n) for n in nums[-2:]])
            # Clean them out of the description for clarity
            cleaned = self.description
            for n in nums[-2:]:
                cleaned = cleaned.replace(n, "")
            if currency:
                cleaned = re.sub(fr"\b{currency}\b", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            self.desc_parts = [cleaned]


# ---------------------------------------------------------------------------
# StreamParser implementation
# ---------------------------------------------------------------------------
class StreamParser:
    """Streaming transaction parser using a tiny FSM."""

    def __init__(self):
        self.state = ParseState.EXPECT_DESC
        self.current_txn: Optional[Txn] = None
        self.completed_txns: List[Txn] = []
        self.numeric_buffer: List[str] = []
        self.doc_currency: Optional[str] = None
        self.collecting = False  # becomes True once we reach the table body
        self.pending_date: Optional[str] = None  # date seen on a stand-alone row
        self.opening_balance: Optional[str] = None
        self.incomplete_txns: List[Txn] = []  # transactions waiting for numbers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def feed_row(self, row: str):
        """Process a *physical* row of text."""
        line = row.strip()

        # Detect currency code once per document
        if self.doc_currency is None:
            mcur = CURRENCY_RE.search(line)
            if mcur:
                self.doc_currency = mcur.group(0)

        # Normalise line once for fast lower-case checks
        low_line = line.lower()

        # Skip obvious non-data artefacts: repeated headers, summary rows, isolated
        # single-word column titles ("Money", "Out", "In", "Balance") that sometimes
        # get bucketed into their *own* physical row by the coordinate clusterer.
        # ----------------------------------------------------------------------
        # Opening-balance guard: some layouts (e.g. *YourBank*) place the numeric
        # value of the opening balance **on the line *above*** the textual
        # "Balance brought forward" sentinel.  That numeric row is *already* in
        # ``self.numeric_buffer`` by the time we encounter the sentinel row
        # below.  If we simply early-return, that stray token will be mistaken
        # for the *amount* of the next real transaction, shifting every
        # subsequent numeric pair out of alignment.  To prevent this we:
        #   1.  Capture the opening balance from the buffer if the sentinel row
        #       itself carries *no* numeric tokens.
        #   2.  Clear the buffer so the value does **not** leak downstream.
        #
        # This logic runs both *before* and *after* ``self.collecting`` flips
        # to *True*, ensuring robustness regardless of where the sentinel is
        # positioned relative to the header row.
        # ----------------------------------------------------------------------
        if any(k in low_line for k in ("balance brought forward", "opening balance", "previous balance")):
            # Attempt to detect/capture the opening balance once per document
            if self.opening_balance is None:
                nums_in_line = AMOUNT_RE.findall(line)
                if nums_in_line:
                    self.opening_balance = _clean_amt(nums_in_line[-1])
                elif self.numeric_buffer:
                    # Numeric row (e.g., '40,000.00') appeared *above* the sentinel
                    self.opening_balance = _clean_amt(self.numeric_buffer[-1])
            # In all cases, purge the buffer to avoid mis-pairing.
            self.numeric_buffer.clear()
            return

        if (
            (not line)
            or HEADER_ROW_RE.search(line)
            or any(kw in low_line for kw in SUMMARY_KEYWORDS)
            or ("money" in low_line and "out" in low_line and "in" in low_line)
            or low_line.strip() in {"money", "out", "in", "balance"}
        ):
            return

        # Wait until we hit the header or opening balance before collecting
        if not self.collecting:
            if any(k in low_line for k in ("balance brought forward", "opening balance", "previous balance")):
                m = AMOUNT_RE.findall(line)
                if m:
                    self.opening_balance = _clean_amt(m[-1])
                # Any numeric tokens encountered *before* the opening-balance
                # sentinel belong to the preamble and must not leak into the
                # main transaction buffer.
                self.numeric_buffer.clear()
                self.collecting = True
                return  # opening-balance row itself is not a transaction
            if ("date" in low_line) and (
                "balance" in low_line or "debit" in low_line or "credit" in low_line or "amount" in low_line
            ):
                self.collecting = True
                return
            # Still in pre-table area
            return

        # ------------------------------------------------------------------
        # Main parsing logic (inside table)
        # ------------------------------------------------------------------
        self._process_body_row(line)

    def finalise(self):
        """Flush buffers & return completed transactions list."""
        # Attach any buffered numbers to the last open txn
        self._try_complete_current_txn()
        # Attempt to finish any queued incomplete txns with whatever numbers remain
        self._try_complete_waiting_txns()

        # Propagate dates: any txn missing a date inherits the last seen date
        last_date = None
        for txn in self.completed_txns:
            if txn.date:
                last_date = txn.date
            else:
                txn.date = last_date

        # Split keyword-joined descriptions (second keyword indicates new txn)
        self._split_keyword_joined()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert completed Txn objects to DataFrame."""
        data = []
        prev_balance_val = None

        # Safe parse of opening balance for delta logic
        if self.opening_balance:
            try:
                prev_balance_val = float(self.opening_balance.replace(",", ""))
            except Exception:
                prev_balance_val = None

        for txn in self.completed_txns:
            if not txn.description or not txn.amt_str or not txn.bal_str:
                continue  # skip incomplete rows

            try:
                bal_val = float(txn.bal_str.replace(",", ""))
            except Exception:
                bal_val = None

            row = {
                "Date": txn.date,
                # Strip any lingering numeric tokens or orphan '.00' fragments for cleanliness
                "Description": re.sub(r"(?<!\d)\.\d{2}(?!\d)", "", re.sub(AMOUNT_RE, "", txn.description)).strip(),
                "Money out": None,
                "Money in": None,
                "Balance": txn.bal_str,
                "Currency": self.doc_currency or "",
            }

            # Collapse excess whitespace after numeric removal
            row["Description"] = re.sub(r"\s+", " ", row["Description"]).strip()

            # Direction classification
            dirn = classify_amount(txn.description)
            if dirn == "in":
                row["Money in"] = txn.amt_str.lstrip("-")
            else:
                row["Money out"] = txn.amt_str.lstrip("-")

            # Sign-based fallback
            if not row["Money in"] and not row["Money out"]:
                if txn.amt_str.startswith("-"):
                    row["Money out"] = txn.amt_str.lstrip("-")
                else:
                    row["Money in"] = txn.amt_str

            # Swap check – if amount > balance, assume mis-pair and swap.
            try:
                amt_val = float(txn.amt_str.replace(",", ""))
            except Exception:
                amt_val = None
            if amt_val is not None and bal_val is not None and amt_val > bal_val:
                if row["Money in"]:
                    row["Money in"], row["Balance"] = row["Balance"], row["Money in"]
                else:
                    row["Money out"], row["Balance"] = row["Balance"], row["Money out"]
                bal_val, amt_val = amt_val, bal_val

            _adjust_dir_by_delta(row, bal_val, prev_balance_val)
            if bal_val is not None:
                prev_balance_val = bal_val

            data.append(row)

        df = pd.DataFrame(data, columns=HeaderList)

        # Date cleanup
        df["Date"] = df["Date"].replace("", np.nan)
        df["Date"] = df["Date"].ffill()
        df = df[df["Date"].notna()]

        # Remove any residual header fragments like "out In" that slipped through
        df = df[~df["Description"].str.lower().str.startswith("out in")]

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_body_row(self, line: str):
        """Dispatch a body row to the right handler."""
        # 1) Date-led row
        m_date = TRANSACTION_DATE_RE.match(line)
        if m_date:
            date_str = m_date.group(1).strip()
            rest = line[m_date.end() :].strip()
            self._handle_date_row(date_str, rest)
            return

        # 2) Pure numeric row (one or two numbers)
        nums = AMOUNT_RE.findall(line)
        non_numeric = re.sub(AMOUNT_RE, "", line).strip()
        if nums and not non_numeric:
            self._handle_numeric_row([_clean_amt(n) for n in nums])
            return

        # 3) Mixed row (text + numbers) or pure text
        if nums:
            self._handle_mixed_row(line, [_clean_amt(n) for n in nums])
        else:
            self._handle_text_row(line)

    # ------------------------------------------------------------------
    # Row-type handlers
    # ------------------------------------------------------------------
    def _handle_date_row(self, date: str, rest: str):
        if not rest:
            # Assign this date to the *most recent* txn that lacks a date,
            # otherwise stash it for the next txn.
            # First try the currently open one.
            if self.current_txn and not self.current_txn.date:
                self.current_txn.date = date
                return
            # Then try the last completed transaction.
            for tx in reversed(self.completed_txns):
                if tx.date is None:
                    tx.date = date
                    return
            # Otherwise hold it for the upcoming description.
            self.pending_date = date
            return

        # If rest contains only numbers → treat as numeric row after storing date
        non_numeric = re.sub(AMOUNT_RE, "", rest).strip()
        nums = AMOUNT_RE.findall(rest)
        if nums and not non_numeric:
            self.pending_date = date  # remember for upcoming desc
            self._handle_numeric_row([_clean_amt(n) for n in nums])
            return

        # Otherwise: date + description (possibly + numbers)
        self._start_new_txn(date, rest)

        # If the (date + description) part carries its own numeric tokens, try to
        # finalise the transaction immediately.
        if nums and len(nums) >= 2:
            # Two tokens present inline → straightforward
            self.current_txn.attach_numbers([_clean_amt(n) for n in nums])
            self._complete_current_txn()
        elif nums and len(nums) == 1 and self.numeric_buffer:
            # Hybrid scenario: one numeric token preceded the description on its
            # own row (already in ``numeric_buffer``) and the matching balance is
            # embedded in the current text.  Pair them FIFO so ordering reflects
            # statement layout (amount first, running balance second).
            first_tok = self.numeric_buffer.pop(0)
            pair = [first_tok, _clean_amt(nums[0])]
            self.current_txn.attach_numbers(pair)
            self._complete_current_txn()

    def _handle_numeric_row(self, nums: List[str]):
        print('DBG numeric_row', nums)
        self.numeric_buffer.extend(nums)
        print('DBG buffer after extend', self.numeric_buffer)
        # First try to satisfy earliest incomplete txn in queue
        self._try_complete_waiting_txns()
        # Then current txn
        self._try_complete_current_txn()

    def _handle_mixed_row(self, line: str, nums: List[str]):
        # --------------------------------------------------------------
        # Heuristic: if the row carries **only one** numeric token and that
        # value is *small* (e.g., a timestamp like «9.52» or reference
        # number), we treat the entire line as *non-numeric* continuation
        # text.  This eliminates spurious transactions created from lines
        # such as "timed 9.52 14 Feb" that merely extend the description of
        # the preceding row.
        # --------------------------------------------------------------
        if len(nums) == 1:
            try:
                val = abs(float(nums[0].replace(",", "")))
            except Exception:
                val = 0.0
            if val < 10:  # unlikely to be an amount; assume timestamp/ref
                self._handle_text_row(line)  # re-dispatch as pure text
                return

        # ------------------------------------------------------------------
        # 1-for-1 hybrid: exactly *one* buffered token **and** one token in the
        # current row → treat them as a (amount, balance) pair **in order of
        # appearance**.  This captures patterns such as:
        #     45,042.16          ← balance on its own row
        #     24 February … 150.00   ← amount + description on next row
        # ------------------------------------------------------------------
        if len(nums) == 1 and len(self.numeric_buffer) == 1:
            # Start the txn and immediately attach the combined pair.
            self._start_new_txn(None, line)
            pair = [self.numeric_buffer.pop(0), nums[0]]
            self.current_txn.attach_numbers(pair)
            # Clean numeric token from description
            clean_txt = re.sub(r"\\b" + re.escape(nums[0]) + r"\\b", "", line)
            if self.doc_currency:
                clean_txt = re.sub(fr"\\b{self.doc_currency}\\b", "", clean_txt)
            clean_txt = re.sub(r"\\s+", " ", clean_txt).strip()
            self.current_txn.desc_parts = [clean_txt]
            self._complete_current_txn()
            return

        # If awaiting numbers, satisfy current txn first
        if self.current_txn and not self.current_txn.is_complete() and len(nums) >= 2:
            self.current_txn.attach_numbers(nums)
            # Remove numbers from text before adding remainder to desc
            for n in nums:
                line = line.replace(n, "")
            clean_txt = re.sub(r"\s+", " ", line).strip()
            self.current_txn.add_desc_part(clean_txt)
            self._complete_current_txn()
            return

        # Otherwise, treat whole line as description start but **clean out**
        # the numeric tokens before storing.
        clean_line = line
        for n in nums:
            clean_line = clean_line.replace(n, "")
        if self.doc_currency:
            clean_line = re.sub(fr"\\b{self.doc_currency}\\b", "", clean_line)
        clean_line = re.sub(r"\s+", " ", clean_line).strip()

        self._start_new_txn(None, clean_line)

        if len(nums) >= 2:
            self.current_txn.attach_numbers(nums)
            self._complete_current_txn()

    def _handle_text_row(self, line: str):
        low = line.lower()

        # ------------------------------------------------------------------
        # Guard: if we are in the middle of collecting a transaction *without*
        # its numeric pair yet, and the incoming text clearly begins a **new**
        # transaction (its first keyword differs from the current one), then
        # push the current txn to *incomplete* queue so the new description
        # starts fresh.  This prevents unrelated rows like "Direct Deposit …"
        # from being merged into the preceding "Randomford's Deli" entry.
        # ------------------------------------------------------------------
        if self.current_txn and not self.current_txn.is_complete():
            cur_kw = next((k for k in TRX_START_KW if k in self.current_txn.description.lower()), None)
            new_kw = next((k for k in TRX_START_KW if k in low), None)
            if new_kw and cur_kw and new_kw != cur_kw:
                # Queue the current incomplete transaction for later completion
                self.incomplete_txns.append(self.current_txn)
                self.current_txn = None

        if CONTINUATION_RE.search(line):
            # Append to the nearest preceding transaction (current, incomplete, or completed)
            target = None
            if self.current_txn:
                target = self.current_txn
            elif self.incomplete_txns:
                target = self.incomplete_txns[-1]
            elif self.completed_txns:
                target = self.completed_txns[-1]
            if target:
                target.add_desc_part(line)
                return

        if self._is_transaction_start(line):
            self._start_new_txn(None, line)
        else:
            if self.current_txn:
                self.current_txn.add_desc_part(line)
            elif self.incomplete_txns:
                # Treat as continuation of the most recent incomplete txn
                self.incomplete_txns[-1].add_desc_part(line)
            else:
                # No context – start a new one to avoid data loss
                self._start_new_txn(None, line)

    # ------------------------------------------------------------------
    # Txn helpers
    # ------------------------------------------------------------------
    def _is_transaction_start(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in TRX_START_KW)

    def _start_new_txn(self, date: Optional[str], text: str):
        # Attempt to finish the current txn first; if still incomplete, queue it.
        self._try_complete_current_txn()
        if self.current_txn and not self.current_txn.is_complete():
            self.incomplete_txns.append(self.current_txn)
            self.current_txn = None

        if date is None:
            date = self.pending_date
        self.pending_date = None

        self.current_txn = Txn(date=date)
        if text:
            self.current_txn.add_desc_part(text)

    def _try_complete_current_txn(self):
        if not self.current_txn or self.current_txn.is_complete():
            return

        # Attempt to use buffered numbers
        if len(self.numeric_buffer) >= 2:
            # Use the earliest buffered tokens (FIFO).  Numbers that appear
            # *before* their description are enqueued first, so the next
            # transaction will pick them up in order.
            self.current_txn.attach_numbers(self.numeric_buffer[:2])
            self.numeric_buffer = self.numeric_buffer[2:]
            self._complete_current_txn()
        elif len(self.numeric_buffer) == 1:
            # If txn already has *one* number (amount or balance) attach the
            # missing side from the buffer and finalise.
            if self.current_txn.amt_str and not self.current_txn.bal_str:
                self.current_txn.bal_str = self.numeric_buffer[0]
                self.numeric_buffer = self.numeric_buffer[1:]
                self._complete_current_txn()
            elif self.current_txn.bal_str and not self.current_txn.amt_str:
                self.current_txn.amt_str = self.numeric_buffer[0]
                self.numeric_buffer = self.numeric_buffer[1:]
                self._complete_current_txn()
            else:
                # Need to wait for the second token
                return
        else:
            # Try inline extraction as last resort
            self.current_txn.try_extract_inline_numbers(self.doc_currency)
            if self.current_txn.is_complete():
                self._complete_current_txn()

    def _complete_current_txn(self):
        if self.current_txn and self.current_txn.is_complete():
            self.completed_txns.append(self.current_txn)
        self.current_txn = None
        self.state = ParseState.EXPECT_DESC

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    def _split_keyword_joined(self):
        """If a description holds two transaction keywords + two num-pairs, split it."""
        new_list: List[Txn] = []
        for txn in self.completed_txns:
            desc_low = txn.description.lower()
            # Skip if description starts with a known compound phrase (e.g., "card payment")
            COMPOUND_KW = {
                "card payment",
                "cash withdrawal",
                "direct debit",
                "direct deposit",
                "monthly apartment rent",
                "monthly rent",
                "apartment rent",
            }
            if any(desc_low.startswith(cmp) for cmp in COMPOUND_KW):
                new_list.append(txn)
                continue

            # Quick check: at least two keywords present?
            kw_hits = [kw for kw in TRX_START_KW if kw in desc_low]
            if len(kw_hits) < 2:
                new_list.append(txn)
                continue

            # Extract numeric tokens in the line
            nums = AMOUNT_RE.findall(txn.description)

            # Case A: two keywords **and** two numeric pairs (≥4 tokens) → current logic
            if len(nums) >= 4:
                first_amt, first_bal, second_amt, second_bal = [_clean_amt(n) for n in nums[-4:]]
            # Case B: two keywords but only ONE numeric pair (2 tokens) – duplicate the pair
            elif len(nums) == 2:
                first_amt, first_bal = [_clean_amt(n) for n in nums]
                second_amt, second_bal = first_amt, first_bal
            else:
                new_list.append(txn)
                continue

            # Split on the *second* keyword occurrence
            second_kw = kw_hits[1]
            parts = re.split(second_kw, txn.description, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) != 2:
                new_list.append(txn)
                continue

            first_part = parts[0].strip()
            second_part = f"{second_kw} {parts[1].strip()}".strip()

            t1 = Txn(date=txn.date, desc_parts=[first_part], amt_str=first_amt, bal_str=first_bal)
            t2 = Txn(date=txn.date, desc_parts=[second_part], amt_str=second_amt, bal_str=second_bal)
            new_list.extend([t1, t2])
        self.completed_txns = new_list

    def _try_complete_waiting_txns(self):
        """Attempt to attach buffered numbers to queued incomplete txns."""
        idx = 0
        while idx < len(self.incomplete_txns) and self.numeric_buffer:
            tx = self.incomplete_txns[idx]
            if tx.is_complete():
                self.completed_txns.append(tx)
                self.incomplete_txns.pop(idx)
                continue

            have_amt = tx.amt_str is not None
            have_bal = tx.bal_str is not None
            missing = 2 - int(have_amt) - int(have_bal)

            if len(self.numeric_buffer) < missing:
                # Not enough numbers yet
                break

            if missing == 2:
                tx.attach_numbers(self.numeric_buffer[:2])
                self.numeric_buffer = self.numeric_buffer[2:]
            elif missing == 1:
                tok = self.numeric_buffer.pop(0)
                if not have_amt:
                    tx.amt_str = tok
                else:
                    tx.bal_str = tok

            if tx.is_complete():
                self.completed_txns.append(tx)
                self.incomplete_txns.pop(idx)
            else:
                idx += 1


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def parse_transactions_stream(rows: List[str]) -> pd.DataFrame:  # noqa: D401
    """Parse *rows* extracted from a PDF using the new stream engine."""

    parser = StreamParser()
    for r in rows:
        parser.feed_row(r)
    parser.finalise()
    return parser.to_dataframe()

# Continuation pattern – rows that just give a time / reference like "timed 17:30" etc.
CONTINUATION_RE = re.compile(r"\btimed\b|" + TIME_RE.pattern, re.IGNORECASE) 