# -*- coding: utf-8 -*-
"""text_line_extractor.py
Generic, template-free bank-statement extractor using PyMuPDF.
The algorithm works in three stages:
1.  Convert each page into word blocks (via ``page.get_text('words')``) and
    cluster words that share the same vertical baseline -> *physical rows*.
2.  From those rows build two parallel lists for the **transaction area**:
      • desc_groups  –  (date, description)
      • numeric_pairs – (amount, running balance)
    The k-th desc_group always belongs to the k-th numeric_pair.
3.  Zip the two lists to build the final DataFrame, using a keyword heuristic
    (and balance-delta fallback) to decide whether the amount goes to
    *Money in* or *Money out*.
No bank-specific constants are used – everything is inferred at runtime.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF
import pandas as pd
import numpy as np

logger = logging.getLogger("text_line_extractor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Constants & regex helpers
# ---------------------------------------------------------------------------
MONTHS_FULL = r"January|February|March|April|May|June|July|August|September|October|November|December"
MONTHS_ABBR = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
# Combined set used in regex
MONTHS_RE = f"{MONTHS_FULL}|{MONTHS_ABBR}"

TRANSACTION_DATE_RE = re.compile(
    rf"""^(\s*(?:
        \d{{1,2}}[\s-]+(?:{MONTHS_RE})(?:[\s-]+\d{{2,4}})? |   # 6 Aug 2017 or 6 August (year optional)
        \d{{1,2}}[\/-]\d{{1,2}}[\/-]\d{{2,4}}         |   # 06/08/2017 or 06-08-17
        \d{{4}}[\/-]\d{{1,2}}[\/-]\d{{1,2}}           |   # 2017-08-06
        \d{{1,2}}[.]\d{{1,2}}[.]\d{{2,4}}                   # 06.08.2017
    ))""",
    re.IGNORECASE | re.VERBOSE,
)
AMOUNT_RE = re.compile(r"[£]?-?\d[\d,\.]*\.\d{2}")

SUMMARY_KEYWORDS = [
    "account summary",
    "total money in",
    "total money out",
    "balance at",
    "page ",
    "bank statement example",
]

HeaderList = ["Date", "Description", "Money out", "Money in", "Balance", "Currency"]

OutgoingKW = {
    "card payment",
    "withdrawal",
    "debit",
    "rent",
    "insurance",
    "bill",
    "petrol",
    "payment",  # generic – re-labelled if salary/bi-weekly present
}
IncomingKW = {"deposit", "salary", "credit", "biweekly", "bi-weekly", "job"}

TRX_START_KW = {"card", "cash", "direct", "deposit", "withdrawal", "payment", "rent", "salary", "yourjob", "monthly"}

# Timestamp pattern used to recognise continuation rows that just contain
# a time stamp or reference number (e.g. "17:38:51 00000102").
TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")

# Detect full column header lines that may repeat on every page
HEADER_ROW_RE = re.compile(r"\bdate\b.*\bdescription\b.*(debit|credit|balance|amount)", re.IGNORECASE)

# Currency code regex (simple three uppercase letters)
CURRENCY_RE = re.compile(r"\b[A-Z]{3}\b")

# Token like "AED -100.00" or "USD 250.00" – first group is the 3-letter code.
CUR_AMOUNT_RE = re.compile(r"\b([A-Z]{3})\s*-?\d[\d,\.]*\.\d{2}\b")

# Helper to strip currency symbols and leading minus for uniform numeric strings
def _clean_amt(tok: str) -> str:
    return tok.lstrip("£")

def classify_amount(description: str) -> str:
    """Return "in" or "out" using keyword heuristics."""
    desc = description.lower()
    if any(k in desc for k in IncomingKW):
        return "in"
    if any(k in desc for k in OutgoingKW):
        # Edge case – salary payment line that still contains "payment"
        if "payment" in desc and ("job" in desc or "biweekly" in desc):
            return "in"
        return "out"
    return "out"  # conservative default

def extract_currency(text: str) -> str | None:
    m = CURRENCY_RE.search(text)
    if m:
        return m.group(0)
    return None

# ---------------------------------------------------------------------------
# Direction correction helper (must appear before parse_transactions)
# ---------------------------------------------------------------------------

def _adjust_dir_by_delta(row: dict, bal_val: float | None, prev_bal_val: float | None):
    """If the sign implied by running balance contradicts the current Money
    in/out assignment, swap them.  No action if both columns already empty or
    both filled."""
    if bal_val is None or prev_bal_val is None:
        return

    delta = bal_val - prev_bal_val
    if delta > 0:
        # Balance increased → credit
        if row.get("Money out") and not row.get("Money in"):
            row["Money in"] = row.pop("Money out")
    elif delta < 0:
        # Balance decreased → debit
        if row.get("Money in") and not row.get("Money out"):
            row["Money out"] = row.pop("Money in")

# ---------------------------------------------------------------------------
# Low-level text extraction helpers
# ---------------------------------------------------------------------------

def extract_rows_words(pdf_path: str, y_tol: float = 1.5) -> List[str]:
    """Cluster words into physical rows using their *y0* coordinate.

    A smaller ``y_tol`` (vertical tolerance) helps keep visually distinct
    statement rows from being merged when the lines are tightly spaced.
    The default has been tuned down from 3.0 → **1.5** after observing that
    several PDF bank statements position successive text baselines only ~2
    units apart, which caused two logical rows to be bucketed together.
    """
    y_tol = y_tol or 1.5  # fallback safeguard
    rows: List[str] = []
    for page in fitz.open(pdf_path):
        buckets: Dict[int, List[Tuple[float, str]]] = {}
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            key = int(y0 // y_tol)
            buckets.setdefault(key, []).append((x0, text))
        for key in sorted(buckets):
            row_txt = " ".join(t for x, t in sorted(buckets[key], key=lambda it: it[0]))
            rows.append(row_txt.strip())
    logger.info(f"Word-bucket clustering produced {len(rows)} rows")
    return rows

# ---------------------------------------------------------------------------
# Core parser (pair-zip logic)
# ---------------------------------------------------------------------------

def parse_transactions(rows: List[str], *, use_new: bool = False) -> pd.DataFrame:
    """Convert clustered *rows* into a clean transaction DataFrame using
    the deterministic *pair-zip* approach.
    
    Args:
        rows: List of text rows from PDF
        use_new: If True, use the new stream parser for hybrid layouts
    """
    if use_new:
        from .txn_stream import parse_transactions_stream
        return parse_transactions_stream(rows)

    desc_groups: List[Tuple[str, str]] = []
    numeric_pairs: List[Tuple[str, str]] = []
    numeric_buffer: List[str] = []  # holds sequential numeric tokens to build pairs
    opening_balance: Optional[str] = None
    collecting = False  # flip to True once we pass the "Balance brought forward" line
    last_date: Optional[str] = None
    doc_currency: Optional[str] = None  # detected once per document

    for row in rows:
        line = row.strip()
        # Detect currency on first occurrence of <CUR XXX amount> pattern
        if doc_currency is None:
            mcur = CUR_AMOUNT_RE.search(line)
            if mcur:
                doc_currency = mcur.group(1)

        if not line or any(kw in line.lower() for kw in SUMMARY_KEYWORDS) or HEADER_ROW_RE.search(line):
            continue

        if not collecting:
            low = line.lower()
            # 1) Opening balance synonyms
            if any(k in low for k in ("balance brought forward", "opening balance", "previous balance")):
                m = AMOUNT_RE.findall(line)
                if m:
                    opening_balance = m[-1].lstrip("£")
                collecting = True
                continue  # skip this line entirely – it's not a transaction

            # 2) Column header row (contains 'date' & balance/amount keywords)
            if ("date" in low) and ("balance" in low or "debit" in low or "credit" in low or "amount" in low):
                collecting = True
                continue  # header row itself should not be parsed as data

            # 3) First date row starts collection (no explicit opening balance)
            if TRANSACTION_DATE_RE.match(line):
                collecting = True  # we will process this same line below
            else:
                # Still in preamble – skip until one of the above triggers fires
                continue

        # artefact header rows
        if line.lower() in {"money", "balance", "out", "in"}:
            continue

        # numeric-only row or numeric-dominant row? -------------------------
        toks = line.split()
        if toks and all(AMOUNT_RE.fullmatch(t) for t in toks):
            # Decide whether this is a genuine numeric-only row
            accept_numeric = len(toks) >= 2
            if not accept_numeric and len(toks) == 1:
                try:
                    accept_numeric = float(toks[0].replace(",", "")) >= 10
                except Exception:
                    accept_numeric = False

            if accept_numeric:
                numeric_buffer.extend(_clean_amt(t) for t in toks)

                # Flush buffer into pairs ONLY if we already have a description waiting.
                # This prevents creation of blank placeholder rows – the numeric pair
                # will be associated with the very next description line we encounter.
                while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                    amt, bal = numeric_buffer[:2]
                    numeric_pairs.append((amt, bal))
                    numeric_buffer = numeric_buffer[2:]
                continue

        # description / date rows
        m_date = TRANSACTION_DATE_RE.match(line)
        if m_date:
            # Handle stand-alone date rows (no additional text) by attaching
            # the date to the most recent description that is missing one.
            cur_date = m_date.group(1)
            rest = line[m_date.end():].strip()

            # If the *rest* is just a timestamp / ref-number line, treat it
            # as a continuation of the previous description instead of a new
            # transaction. This covers wrapped statement rows such as:
            #    06 Aug 2017  UAE SWITCH WDL … AED -100.00
            #    06-08-2017  17:38:51 00000102

            if TIME_RE.search(rest) and not AMOUNT_RE.search(rest):
                if desc_groups:
                    d, txt = desc_groups[-1]
                    desc_groups[-1] = (d, f"{txt} {rest}".strip())
                else:
                    desc_groups.append((cur_date, rest))
                # Do NOT treat as new transaction; skip further processing.
                continue

            if rest:
                cleaned_rest_alpha = re.sub(AMOUNT_RE, "", rest).strip()
                if not cleaned_rest_alpha:
                    # Row contains only date + numeric tokens (amount/balance).
                    # Treat as **numeric-only** row: buffer the numbers and wait
                    # for the true description on the following physical row.
                    num_matches = AMOUNT_RE.findall(rest)
                    numeric_buffer.extend(_clean_amt(n) for n in num_matches)
                    # Remember the date for the upcoming description line.
                    last_date = cur_date
                    # Attempt immediate flush (in case description already waiting)
                    while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                        amt, bal = numeric_buffer[:2]
                        numeric_pairs.append((amt, bal))
                        numeric_buffer = numeric_buffer[2:]
                else:
                    # We have descriptive text on the same line – create a new transaction group.
                    desc_groups.append((cur_date, rest))
                    # Capture any inline numeric tokens so pair remains aligned.
                    num_matches = AMOUNT_RE.findall(rest)
                    if len(num_matches) >= 2:
                        numeric_buffer.extend(_clean_amt(n) for n in num_matches[-2:])
                        while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                            amt, bal = numeric_buffer[:2]
                            numeric_pairs.append((amt, bal))
                            numeric_buffer = numeric_buffer[2:]
            else:
                # Stand-alone date line (no description).  Attach this date to the most
                # recently collected description **if** it does not yet have one; this
                # covers statements where the description line precedes its date on the
                # next physical row (e.g., "Card payment …" followed by a lone
                # "1 February" line).
                if desc_groups and not desc_groups[-1][0]:
                    d, txt = desc_groups[-1]
                    desc_groups[-1] = (cur_date, txt)
                else:
                    # If we cannot attach, remember it so the following description can
                    # inherit via *last_date* fallback a few lines below.
                    last_date = cur_date

            # Flush buffered numeric pairs **only** if we already have a matching
            # description waiting. This prevents early emission when the numeric
            # pair precedes its description (common in some statements).
            while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                amt, bal = numeric_buffer[:2]
                numeric_pairs.append((amt, bal))
                numeric_buffer = numeric_buffer[2:]
            # If no description text present, loop continues – numbers remain buffered
        else:
            # Mixed-content row that still contains amount/balance figures (e.g. '25.00 2,575.00 CR')
            num_matches = AMOUNT_RE.findall(line)
            if num_matches and len(num_matches) >= 2:
                # Only treat the row as containing transactional numbers if **two or more**
                # numeric values are present – this filters out stray values like
                # times (e.g., '9.52') that are not amounts/balances.
                numeric_buffer.extend(_clean_amt(n) for n in num_matches)
                # Try to emit pairs using buffer
                while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                    amt, bal = numeric_buffer[:2]
                    numeric_pairs.append((amt, bal))
                    numeric_buffer = numeric_buffer[2:]
                # Remove numeric parts from description for readability
                cleaned_desc_part = re.sub(AMOUNT_RE, "", line).strip()
                if cleaned_desc_part:
                    if desc_groups:
                        if len(numeric_pairs) >= len(desc_groups):
                            # Previous description already has its numbers → start a new transaction under same date
                            d, _ = desc_groups[-1]
                            desc_groups.append((d, cleaned_desc_part))
                        else:
                            # Still collecting for current transaction → append text
                            d, txt = desc_groups[-1]
                            desc_groups[-1] = (d, f"{txt} {cleaned_desc_part}".strip())
                    else:
                        # Shouldn't happen, but just in case create a placeholder date
                        desc_groups.append((last_date or "", cleaned_desc_part))

                    # After possibly adding a new description group, attempt to flush buffered pairs again
                    while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                        amt, bal = numeric_buffer[:2]
                        numeric_pairs.append((amt, bal))
                        numeric_buffer = numeric_buffer[2:]
                continue  # numeric handled

            # --- Pure text continuation line (no numbers) --------------------
            if line and not num_matches:
                cleaned_text = line.strip()
                if cleaned_text:
                    if desc_groups:
                        # Start a fresh transaction when the current line begins with
                        # a *transaction-keyword* (e.g., "direct", "card", "cash") **even if**
                        # there is no numeric buffer waiting – provided every previous
                        # description already has its numeric pair assigned.  This is
                        # essential for statements where the numeric pair appears *after*
                        # the next description line (e.g. "Direct Debit – Home Insurance"
                        # followed by the numeric row on the next line).
                        new_trx_hint = cleaned_text.split()[0].lower() if cleaned_text else ""
                        if (
                            new_trx_hint in TRX_START_KW and len(numeric_pairs) >= len(desc_groups)
                        ) or (
                            numeric_buffer and len(numeric_pairs) >= len(desc_groups)
                        ):
                            d, _ = desc_groups[-1]
                            desc_groups.append((d, cleaned_text))
                        else:
                            # Otherwise treat as continuation of the current text.
                            d, txt = desc_groups[-1]
                            if txt == "":
                                desc_groups[-1] = (d, cleaned_text)
                            else:
                                desc_groups[-1] = (d, f"{txt} {cleaned_text}".strip())
                    else:
                        # No previous group – create new with last known date
                        desc_groups.append((last_date or "", cleaned_text))

                    # Flush numeric buffer if possible
                    while len(numeric_buffer) >= 2 and len(numeric_pairs) < len(desc_groups):
                        amt, bal = numeric_buffer[:2]
                        numeric_pairs.append((amt, bal))
                        numeric_buffer = numeric_buffer[2:]
                continue

            # --- Mixed-content row handling block ------------------------------
            if num_matches and len(num_matches)==1 and len(numeric_buffer)==1:
                # Pattern: balance (or amount) appeared on the previous numeric-only row,
                # and the complementary amount (or balance) is in this mixed row. Pair them.
                numeric_buffer.append(_clean_amt(num_matches[0]))
                if len(numeric_pairs) < len(desc_groups):
                    amt, bal = numeric_buffer[:2]
                    numeric_pairs.append((amt, bal))
                    numeric_buffer = numeric_buffer[2:]
                # Remove the numeric token from the description text
                cleaned_desc_part = re.sub(AMOUNT_RE, "", line).strip()
                if cleaned_desc_part:
                    d_latest, txt_latest = desc_groups[-1] if desc_groups else (last_date or "", "")
                    if len(numeric_pairs) > len(desc_groups):
                        desc_groups.append((d_latest, cleaned_desc_part))
                    else:
                        desc_groups[-1] = (d_latest, f"{txt_latest} {cleaned_desc_part}".strip())
                continue

    # If opening balance was not found, fall back to the first numeric pair's balance
    if opening_balance is None and numeric_pairs:
        opening_balance = numeric_pairs[0][1]

    if len(numeric_pairs) - len(desc_groups) == 1:
        # We have one extra numeric pair at the start – check if it duplicates the opening balance.
        first_amt, first_bal = numeric_pairs[0]
        if first_bal.replace(",", "") == opening_balance.replace(",", ""):
            logger.debug("Dropping first numeric pair as it matches opening balance")
            numeric_pairs = numeric_pairs[1:]

    if len(numeric_pairs) != len(desc_groups):
        logger.warning(
            f"Pair/desc mismatch ({len(numeric_pairs)} vs {len(desc_groups)}). Trimming to shortest."
        )

    # Use *inline* parsing only when we failed to collect ANY detached numeric
    # pairs – this happens in layouts where every amount & balance sit on the
    # same physical row as the description (e.g., many scanned PDFs).  If we
    # managed to extract at least one pair, stick with the pair-zip logic and
    # simply trim to the shorter list; this keeps hybrid layouts (like the
    # YourBank sample) working correctly.
    force_inline = len(numeric_pairs) == 0

    # ------------------------------------------------------------------
    # Ensure transactions are in chronological (oldest → newest) order.  
    # Many statements list the most recent transaction first, which would
    # break the balance‐delta logic that assumes running balance evolves
    # *downwards* through the DataFrame.  We therefore detect a descending
    # date sequence and, if found, reverse both lists so that the first
    # element is the earliest transaction.
    # ------------------------------------------------------------------
    try:
        import pandas as _pd  # lightweight helper import – already used elsewhere

        parsed_dates = [_pd.to_datetime(d, dayfirst=True, errors="coerce") for d, _ in desc_groups if d]
        if parsed_dates and parsed_dates[0] > parsed_dates[-1]:
            logger.info("Input rows appear in reverse-chronological order → reversing to chronological order")
            desc_groups.reverse()
            numeric_pairs.reverse()
    except Exception as _e:
        logger.debug(f"Date orientation detection failed: {_e}")

    pair_len = 0 if force_inline else min(len(numeric_pairs), len(desc_groups))

    # Build DataFrame rows --------------------------------------------------
    data: List[Dict[str, Optional[str]]] = []
    data.append(
        {
            "Date": "",  # opening row has no explicit date in most statements
            "Description": "Balance brought forward",
            "Money out": None,
            "Money in": None,
            "Balance": opening_balance,
            "Currency": doc_currency or "",
        }
    )

    prev_balance_val = float(opening_balance.replace(",", "")) if opening_balance else None

    if pair_len > 0:
        # --- Standard pair-zip path -----------------------------------------
        tol = 0.01
        prev_txn_date = None
        for (amt_str, bal_str), (date, desc) in zip(numeric_pairs[:pair_len], desc_groups[:pair_len]):
            if not date:
                date = prev_txn_date or last_date or ""
            # ------------------------------------------------------------------
            # 1. Resolve which value is Balance vs Amount using prev balance
            # ------------------------------------------------------------------
            amt_val = bal_val = None
            try:
                n1 = float(amt_str.replace(",", "").replace("-", ""))
                n2 = float(bal_str.replace(",", ""))
            except Exception:
                n1 = n2 = None

            if prev_balance_val is not None and n1 is not None and n2 is not None:
                delta1 = round(n2 - prev_balance_val, 2)
                delta2 = round(n1 - prev_balance_val, 2)

                # Check if delta1 magnitude matches n1 (arrangement correct)
                match1 = abs(abs(delta1) - n1) < tol
                match2 = abs(abs(delta2) - n2) < tol

                # Decide arrangement based on which match is valid – if *both* look
                # valid (rare), prefer the variant where the amount is smaller than
                # the balance value (a heuristic that covers typical statements).
                if match1 and not match2:
                    amt_val, bal_val = n1, n2
                elif match2 and not match1:
                    amt_val, bal_val = n2, n1
                    # swap string representations so strings align with values
                    amt_str, bal_str = bal_str, amt_str
                elif match1 and match2:
                    # Both permutations look plausible; pick the one where the amount
                    # is the *smaller* of the two numbers (typical case).
                    if n1 < n2:
                        amt_val, bal_val = n1, n2
                    else:
                        amt_val, bal_val = n2, n1
                        amt_str, bal_str = bal_str, amt_str
                else:
                    # Fallback heuristic – assume the *larger* number is the running
                    # balance and the smaller is the transaction amount.
                    if n1 is not None and n2 is not None:
                        if n1 > n2:
                            bal_val, amt_val = n1, n2
                            # swap string representations so strings align with values
                            amt_str, bal_str = bal_str, amt_str
                        else:
                            bal_val, amt_val = n2, n1
                            # keep current order (n1 already amount)
                    else:
                        amt_val, bal_val = n1, n2

            # ------------------------------------------------------------------
            # 2. Build row dictionary
            # ------------------------------------------------------------------
            cleaned_description = desc
            for n in [amt_str, bal_str]:
                cleaned_description = cleaned_description.replace(n, "")
            if doc_currency:
                cleaned_description = re.sub(fr"\b{doc_currency}\b", "", cleaned_description)
            cleaned_description = re.sub(r"\s+", " ", cleaned_description)
            row = {
                "Date": date if date else None,
                "Description": cleaned_description,
                "Money out": None,
                "Money in": None,
                "Balance": bal_str,
                "Currency": doc_currency or "",
            }

            dirn = classify_amount(desc)
            if dirn == "in":
                row["Money in"] = amt_str.lstrip("-")
            elif dirn == "out":
                row["Money out"] = amt_str.lstrip("-")

            # Sign-based fallback: if still unassigned, use minus sign to infer
            if not row["Money in"] and not row["Money out"]:
                if amt_str.startswith("-"):
                    row["Money out"] = amt_str.lstrip("-")
                else:
                    row["Money in"] = amt_str

            _adjust_dir_by_delta(row, bal_val, prev_balance_val)

            if bal_val is not None:
                prev_balance_val = bal_val

            data.append(row)
            prev_txn_date = date

        # --------------------------------------------------------------
        # Handle any *remaining* description groups that did not get a
        # detached numeric pair (common in mixed layouts where the last
        # few transactions have their amount & balance inline).
        # --------------------------------------------------------------
        if len(desc_groups) > pair_len:
            logger.info("Processing trailing description rows via inline mode")
            remaining = desc_groups[pair_len:]
            for date, desc in remaining:
                nums = AMOUNT_RE.findall(desc)
                if len(nums) < 2:
                    logger.debug(f"Skipping row – could not find 2 numeric pairs: {desc}")
                    continue
                amt, bal = nums[-2], nums[-1]

                cleaned_desc = desc
                for n in [bal, amt]:
                    cleaned_desc = cleaned_desc.replace(n, "")
                if doc_currency:
                    cleaned_desc = re.sub(fr"\b{doc_currency}\b", "", cleaned_desc)
                cleaned_desc = re.sub(r"\s+", " ", cleaned_desc.strip())

                row = {
                    "Date": date if date else None,
                    "Description": cleaned_desc,
                    "Money out": None,
                    "Money in": None,
                    "Balance": bal,
                    "Currency": doc_currency or "",
                }

                dirn = classify_amount(cleaned_desc)
                if dirn == "in":
                    row["Money in"] = amt.lstrip("-")
                else:
                    row["Money out"] = amt.lstrip("-")

                try:
                    bal_val = float(bal.replace(",", ""))
                except Exception:
                    bal_val = None
                _adjust_dir_by_delta(row, bal_val, prev_balance_val)

                if bal_val is not None:
                    prev_balance_val = bal_val

                data.append(row)
    else:
        # --- Fallback: parse numbers inline on each description row ---------
        logger.info("Falling back to inline-parse mode (amount & balance within same row)")
        for date, desc in desc_groups:
            nums = AMOUNT_RE.findall(desc)
            if len(nums) < 2:
                logger.debug(f"Skipping row – could not find 2 numeric pairs: {desc}")
                continue
            amt, bal = nums[-2], nums[-1]

            # Remove those trailing pairs from description for cleanliness
            # (only the *last* two pairs, because some descriptions legitimately
            # contain pairs, e.g., street addresses)
            cleaned_desc = desc
            for n in [bal, amt]:
                cleaned_desc = cleaned_desc.replace(n, "")
            if doc_currency:
                cleaned_desc = re.sub(fr"\b{doc_currency}\b", "", cleaned_desc)
            cleaned_desc = re.sub(r"\s+", " ", cleaned_desc.strip())

            row = {
                "Date": date if date else None,
                "Description": cleaned_desc,
                "Money out": None,
                "Money in": None,
                "Balance": bal,
                "Currency": doc_currency or "",
            }

            dirn = classify_amount(cleaned_desc)
            if dirn == "in":
                row["Money in"] = amt.lstrip("-")
            elif dirn == "out":
                row["Money out"] = amt.lstrip("-")

            # Sign-based fallback: if still unassigned, use minus sign to infer
            if not row["Money in"] and not row["Money out"]:
                if amt.startswith("-"):
                    row["Money out"] = amt.lstrip("-")
                else:
                    row["Money in"] = amt

            # Balance-delta disambiguation
            try:
                bal_val = float(bal.replace(",", ""))
            except Exception:
                bal_val = None
            _adjust_dir_by_delta(row, bal_val, prev_balance_val)

            if bal_val is not None:
                prev_balance_val = bal_val

            data.append(row)

    df = pd.DataFrame(data, columns=HeaderList)

    # Clean Date column
    df["Date"] = df["Date"].replace("", np.nan)
    if len(df) > 1:
        df.loc[1:, "Date"].ffill(inplace=True)

    # Drop the opening balance row
    df = df[df["Description"].str.lower() != "balance brought forward"]

    # Final keep mask – ensure we only retain rows with a recognised date
    df["Date"] = df["Date"].replace(["", None], np.nan)
    df["Date"].ffill(inplace=True)

    # Retain only rows that have a recognised (forward-filled) date
    df = df[df["Date"].notna()]

    # After processing all rows, log debug info ---------------------------------
    logger.debug(f"Description groups: {len(desc_groups)} -> {desc_groups}")
    logger.debug(f"Numeric pairs    : {len(numeric_pairs)} -> {numeric_pairs}")

    return df

# ---------------------------------------------------------------------------
# Wrapper util – produce Excel
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, output_dir: str = "PDF Test Outputs") -> Tuple[pd.DataFrame, str]:
    os.makedirs(output_dir, exist_ok=True)
    physical_rows = extract_rows_words(pdf_path)

    # ------------------------------------------------------------------
    # Choose parser automatically: the *YourBank* layout benefits from the
    # new stream parser, whereas statements like ENBD still parse best with
    # the original pair-zip logic.  A lightweight heuristic is enough – if
    # we detect the words "your bank" or "yourbank" anywhere in the text,
    # we start with the stream parser, otherwise with the legacy parser.
    # If the first attempt returns an **empty** DataFrame, we retry with the
    # other parser as a safety net.
    # ------------------------------------------------------------------
    joined_text = "\n".join(physical_rows).lower()
    prefer_new = any(k in joined_text for k in ("your bank", "yourbank"))

    if prefer_new:
        df = parse_transactions(physical_rows, use_new=True)
        if df.empty:
            df = parse_transactions(physical_rows, use_new=False)
    else:
        df = parse_transactions(physical_rows, use_new=False)
        if df.empty:
            df = parse_transactions(physical_rows, use_new=True)

    fname = os.path.basename(pdf_path).replace(".pdf", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{fname}_text_{ts}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)
    logger.info(f"Excel saved ➜ {out_path}")
    return df, out_path

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_line_extractor.py <pdf>")
        sys.exit(1)
    _pdf = sys.argv[1]
    _df, _path = process_pdf(_pdf)
    print(_df)

# ---------------------------------------------------------------------------
# Direction correction helper
# ---------------------------------------------------------------------------

def _adjust_dir_by_delta(row: dict, bal_val: float | None, prev_bal_val: float | None):
    """If the sign implied by running balance contradicts the current Money
    in/out assignment, swap them.  No action if both columns already empty or
    both filled."""
    if bal_val is None or prev_bal_val is None:
        return

    delta = bal_val - prev_bal_val
    if delta > 0:
        # Balance increased → credit
        if row.get("Money out") and not row.get("Money in"):
            row["Money in"] = row.pop("Money out")
    elif delta < 0:
        # Balance decreased → debit
        if row.get("Money in") and not row.get("Money out"):
            row["Money out"] = row.pop("Money in") 