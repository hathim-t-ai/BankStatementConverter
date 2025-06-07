"""Advanced bank statement converter with multiple extraction strategies."""

import re
import pandas as pd
import pdfplumber
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging

# Optional imports with fallbacks
try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logging.warning("Tabula not available. PDF table extraction will be limited.")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available. Lattice table extraction will be limited.")

try:
    from .ocr_converter import OCRBankStatementConverter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR not available. Image-based PDF extraction will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBankStatementConverter:
  def __init__(self, headers: List[str]):
    self.headers = [h.strip() for h in headers]
    self.normalized_headers = [h.strip().lower() for h in self.headers]
    
  def convert_multiple(self, pdf_paths: List[str], progress_callback: Optional[Callable] = None) -> pd.DataFrame:
    """Process multiple PDF files and combine results."""
    all_results = []
    total_files = len(pdf_paths)
    
    for i, path in enumerate(pdf_paths):
      if progress_callback:
        progress = 20 + (i * 60 // total_files)
        progress_callback(progress, f'Processing {path.split("/")[-1]}...')
      
      try:
        results = self.convert_single(path)
        if not results.empty:
          results['source_file'] = path.split('/')[-1]
          all_results.append(results)
      except Exception as e:
        logger.error(f"Error processing {path}: {str(e)}")
        continue
    
    if progress_callback:
      progress_callback(90, 'Combining results...')
    
    if all_results:
      combined = pd.concat(all_results, ignore_index=True)
      if progress_callback:
        progress_callback(100, 'Processing complete!')
      return combined
    else:
      return pd.DataFrame()
  
  def convert_single(self, pdf_path: str) -> pd.DataFrame:
    """Process a single PDF using multiple extraction strategies."""
    results = []
    
    # Strategy 1: PDFPlumber (always available - most reliable)
    try:
      pdfplumber_results = self._extract_with_pdfplumber(pdf_path)
      if not pdfplumber_results.empty:
        results.append(pdfplumber_results)
    except Exception as e:
      logger.warning(f"PDFPlumber extraction failed: {str(e)}")
    
    # Strategy 2: Tabula (if available)
    if TABULA_AVAILABLE:
      try:
        tabula_results = self._extract_with_tabula(pdf_path)
        if not tabula_results.empty:
          results.append(tabula_results)
      except Exception as e:
        logger.warning(f"Tabula extraction failed: {str(e)}")
    
    # Strategy 3: Camelot (if available)
    if CAMELOT_AVAILABLE:
      try:
        camelot_results = self._extract_with_camelot(pdf_path)
        if not camelot_results.empty:
          results.append(camelot_results)
      except Exception as e:
        logger.warning(f"Camelot extraction failed: {str(e)}")
    
    # Strategy 4: Text pattern matching (always available)
    try:
      pattern_results = self._extract_with_patterns(pdf_path)
      if not pattern_results.empty:
        results.append(pattern_results)
    except Exception as e:
      logger.warning(f"Pattern extraction failed: {str(e)}")
    
    # Strategy 5: OCR (if available and no results from other methods)
    if not results and OCR_AVAILABLE:
      try:
        ocr_results = self._extract_with_ocr(pdf_path)
        if not ocr_results.empty:
          results.append(ocr_results)
          logger.info("OCR extraction successful")
      except Exception as e:
        logger.warning(f"OCR extraction failed: {str(e)}")
    
    # Combine and deduplicate results
    if results:
      combined = pd.concat(results, ignore_index=True)
      return self._deduplicate_rows(combined)
    else:
      return pd.DataFrame()
  
  def _extract_with_pdfplumber(self, pdf_path: str) -> pd.DataFrame:
    """Extract tables using PDFPlumber with improved table selection."""
    all_rows = []
    
    with pdfplumber.open(pdf_path) as pdf:
      for page_num, page in enumerate(pdf.pages):
        logger.info(f"Processing page {page_num + 1} with PDFPlumber")
        
        # Extract structured tables first
        tables = page.extract_tables()
        logger.info(f"Found {len(tables)} structured tables on page {page_num + 1}")
        
        # Process tables in order and prioritize main transaction table
        main_table_found = False
        
        for table_idx, table in enumerate(tables):
          if not table or len(table) < 2:
            continue
          
          logger.info(f"Examining table {table_idx + 1} with {len(table)} rows")
          
          # Check if this is likely the main transaction table
          header_row_idx = None
          for i, row in enumerate(table):
            if self._headers_match(row):
              header_row_idx = i
              logger.info(f"Found header row at index {i} in table {table_idx + 1}")
              break
          
          if header_row_idx is not None:
            # Extract data rows from this table
            data_rows_extracted = 0
            for row in table[header_row_idx + 1:]:
              if self._is_valid_transaction_row(row):
                clean_row = self._clean_row(row)
                if clean_row:
                  all_rows.append(clean_row)
                  data_rows_extracted += 1
            
            logger.info(f"Extracted {data_rows_extracted} data rows from table {table_idx + 1}")
            
            # If we found a good main transaction table, prioritize it
            if data_rows_extracted >= 3:  # Minimum threshold for main table
              main_table_found = True
              break
        
        # If no structured tables found or no main table identified, extract from text
        if not main_table_found:
          logger.info(f"No main transaction table found in structured tables, trying text extraction")
          text = page.extract_text() or ""
          if text:
            text_rows = self._extract_from_text_patterns(text)
            all_rows.extend(text_rows)
            logger.info(f"Extracted {len(text_rows)} rows from text patterns")
    
    logger.info(f"Total rows extracted from PDF: {len(all_rows)}")
    return self._create_dataframe(all_rows)
  
  def _extract_with_tabula(self, pdf_path: str) -> pd.DataFrame:
    """Extract tables using Tabula (if available)."""
    if not TABULA_AVAILABLE:
      return pd.DataFrame()
    
    all_rows = []
    
    try:
      # Extract all tables from all pages
      tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, 
                              pandas_options={'header': None})
      
      for table in tables:
        if table.empty or len(table) < 2:
          continue
        
        # Convert to list of lists for processing
        table_data = table.values.tolist()
        table_data = [[str(cell) if pd.notna(cell) else '' for cell in row] for row in table_data]
        
        # Find header row
        for i, row in enumerate(table_data):
          if self._headers_match(row):
            # Extract data rows after header
            for data_row in table_data[i+1:]:
              if self._is_valid_data_row(data_row):
                clean_row = self._clean_row(data_row)
                all_rows.append(clean_row)
            break
    
    except Exception as e:
      logger.warning(f"Tabula extraction error: {str(e)}")
    
    return self._create_dataframe(all_rows)
  
  def _extract_with_camelot(self, pdf_path: str) -> pd.DataFrame:
    """Extract tables using Camelot (if available)."""
    if not CAMELOT_AVAILABLE:
      return pd.DataFrame()
    
    all_rows = []
    
    try:
      # Try lattice method first (for tables with lines)
      tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
      
      # If no tables found, try stream method
      if len(tables) == 0:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
      
      for table in tables:
        if table.df.empty or len(table.df) < 2:
          continue
        
        # Convert to list of lists
        table_data = table.df.values.tolist()
        table_data = [[str(cell).strip() for cell in row] for row in table_data]
        
        # Find header row
        for i, row in enumerate(table_data):
          if self._headers_match(row):
            # Extract data rows after header
            for data_row in table_data[i+1:]:
              if self._is_valid_data_row(data_row):
                clean_row = self._clean_row(data_row)
                all_rows.append(clean_row)
            break
    
    except Exception as e:
      logger.warning(f"Camelot extraction error: {str(e)}")
    
    return self._create_dataframe(all_rows)
  
  def _extract_with_patterns(self, pdf_path: str) -> pd.DataFrame:
    """Extract data using text pattern matching for unstructured tables."""
    all_rows = []
    
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        text = page.extract_text() or ""
        rows = self._extract_from_text_patterns(text)
        all_rows.extend(rows)
    
    return self._create_dataframe(all_rows)
  
  def _extract_from_text_patterns(self, text: str) -> List[List[str]]:
    """Extract tabular data from unstructured text using patterns."""
    rows = []
    lines = text.split('\n')
    
    logger.info(f"Processing {len(lines)} lines of text for patterns")
    
    # Look for header line
    header_line_idx = None
    for i, line in enumerate(lines):
      if self._line_contains_headers(line):
        header_line_idx = i
        logger.info(f"Found potential header line at {i}: {line.strip()}")
        break
    
    if header_line_idx is None:
      logger.info("No header line found in text patterns")
      return rows
    
    # Extract data lines after header
    data_lines_processed = 0
    for i, line in enumerate(lines[header_line_idx + 1:], header_line_idx + 1):
      if not line.strip():
        continue
      
      # Skip lines that look like subtotals or summaries
      if any(keyword in line.lower() for keyword in ['total', 'subtotal', 'balance brought', 'carried forward', 'page']):
        continue
      
      # Try to parse as tabular data
      parsed_row = self._parse_text_line_to_row(line)
      if parsed_row and len(parsed_row) >= 3:  # At least date, description, amount
        if self._is_valid_data_row(parsed_row):
          # Pad row to match expected columns
          while len(parsed_row) < len(self.headers):
            parsed_row.append('')
          rows.append(parsed_row[:len(self.headers)])
          data_lines_processed += 1
          logger.info(f"Extracted row {data_lines_processed}: {parsed_row}")
    
    logger.info(f"Extracted {len(rows)} rows from text patterns")
    return rows
  
  def _line_contains_headers(self, line: str) -> bool:
    """Check if line contains the expected table headers."""
    line_lower = line.lower().strip()
    
    # Debug logging
    logger.info(f"Checking line for headers: {line_lower}")
    
    # Count how many expected headers are found
    found_headers = 0
    expected_count = len(self.normalized_headers)
    
    # Check for key transaction table headers specifically
    key_headers = ['date', 'description', 'amount', 'debit', 'credit', 'balance']
    
    for header in self.normalized_headers:
      header_clean = header.lower().strip()
      
      # More flexible matching for common bank statement headers
      if header_clean in ['date']:
        if any(date_word in line_lower for date_word in ['date', 'transaction date', 'post date']):
          found_headers += 1
          logger.info(f"Found date header in line: {line}")
      elif header_clean in ['description']:
        if any(desc_word in line_lower for desc_word in ['description', 'transaction', 'details', 'reference']):
          found_headers += 1
          logger.info(f"Found description header in line: {line}")
      elif header_clean in ['amount']:
        if any(amt_word in line_lower for amt_word in ['amount', 'debit', 'credit']):
          found_headers += 1
          logger.info(f"Found amount header in line: {line}")
      elif header_clean in ['debit']:
        if 'debit' in line_lower:
          found_headers += 1
          logger.info(f"Found debit header in line: {line}")
      elif header_clean in ['credit']:
        if 'credit' in line_lower:
          found_headers += 1
          logger.info(f"Found credit header in line: {line}")
      elif header_clean in ['balance']:
        if 'balance' in line_lower:
          found_headers += 1
          logger.info(f"Found balance header in line: {line}")
      elif header_clean in line_lower:
        found_headers += 1
        logger.info(f"Found exact header '{header_clean}' in line: {line}")
    
    # Require finding at least 60% of headers for main transaction table
    threshold = max(2, int(expected_count * 0.6))
    is_header_line = found_headers >= threshold
    
    # Additional check: avoid summary tables and other sections
    avoid_terms = ['summary', 'charges and fees', 'checks paid', 'deposits and other credits', 'withdrawals and other debits']
    if any(term in line_lower for term in avoid_terms):
      logger.info(f"Skipping line due to avoid terms: {line}")
      return False
    
    if is_header_line:
      logger.info(f"HEADER LINE FOUND with {found_headers}/{expected_count} headers: {line}")
    
    return is_header_line
  
  def _parse_text_line_to_row(self, line: str) -> Optional[List[str]]:
    """Parse a text line into a structured row with better date extraction."""
    line = line.strip()
    if not line:
      return None
    
    # Skip summary lines and non-data lines
    skip_terms = ['total', 'subtotal', 'balance brought', 'page', 'account #', 'continued', 'statement', 'summary']
    if any(term in line.lower() for term in skip_terms):
      return None
    
    logger.info(f"Parsing line: {line}")
    
    # Enhanced date pattern matching for MM/DD format commonly used in US bank statements
    date_patterns = [
      r'\b(1[0-2]|0?[1-9])/(3[01]|[12][0-9]|0?[1-9])\b',  # MM/DD
      r'\b(1[0-2]|0?[1-9])/(3[01]|[12][0-9]|0?[1-9])/(\d{2,4})\b',  # MM/DD/YY or MM/DD/YYYY
      r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'  # General date pattern
    ]
    
    # Enhanced amount pattern for monetary values
    amount_patterns = [
      r'\b\d+[.,]\d{2}\b',  # Standard monetary format
      r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b',  # With comma separators
      r'\$\s*\d+[.,]\d{2}\b'  # With dollar sign
    ]
    
    # Try to find date in the line
    date_found = None
    remaining_line = line
    
    for pattern in date_patterns:
      match = re.search(pattern, line)
      if match:
        date_found = match.group()
        remaining_line = line.replace(match.group(), '', 1).strip()
        logger.info(f"Found date: {date_found}")
        break
    
    if not date_found:
      # If no date pattern found, skip this line for transaction data
      logger.info(f"No date found in line, skipping: {line}")
      return None
    
    # Find amounts in the remaining line
    amounts = []
    amount_line = remaining_line
    
    for pattern in amount_patterns:
      found_amounts = re.findall(pattern, amount_line)
      for amount in found_amounts:
        amounts.append(amount.replace('$', '').replace(',', '').strip())
        amount_line = amount_line.replace(amount, '', 1)
    
    # Clean up description (what's left after removing date and amounts)
    description = remaining_line
    for amount in amounts:
      # Remove amount patterns from description
      description = re.sub(r'\b' + re.escape(amount) + r'\b', '', description)
      description = re.sub(r'\$\s*' + re.escape(amount) + r'\b', '', description)
    
    # Clean up description
    description = re.sub(r'\s+', ' ', description).strip()
    
    # Build the row based on expected headers
    row = []
    
    for header in self.normalized_headers:
      if header in ['date']:
        row.append(date_found)
      elif header in ['description']:
        row.append(description)
      elif header in ['debit']:
        # For debit, use first amount if available
        row.append(amounts[0] if amounts else '')
      elif header in ['credit']:
        # For credit, use second amount if available, otherwise empty
        row.append(amounts[1] if len(amounts) > 1 else '')
      elif header in ['balance']:
        # For balance, use last amount if available
        row.append(amounts[-1] if amounts else '')
      elif header in ['amount']:
        # For generic amount, use first available amount
        row.append(amounts[0] if amounts else '')
      else:
        # For any other header, try to extract relevant data
        row.append('')
    
    # Ensure we have meaningful data
    if date_found and (description or amounts):
      logger.info(f"Successfully parsed row: {row}")
      return row
    
    logger.info(f"Insufficient data in parsed row, skipping")
    return None
  
  def _smart_split_line(self, line: str) -> List[str]:
    """Intelligently split a line based on data patterns."""
    parts = []
    remaining = line.strip()
    
    logger.debug(f"Smart splitting line: '{remaining}'")
    
    # Common patterns for bank statement fields
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    amount_pattern = r'\b\d{1,3}(?:,\d{3})*\.?\d{0,2}\b'  # Handles amounts like 1,234.56 or 123
    
    # Try to extract date first
    date_match = re.search(date_pattern, remaining)
    if date_match:
      date_str = date_match.group().strip()
      parts.append(date_str)
      remaining = remaining[:date_match.start()] + ' ' + remaining[date_match.end():]
      logger.debug(f"Found date: {date_str}")
    
    # Find all amounts in the line
    amounts = []
    amount_matches = list(re.finditer(amount_pattern, remaining))
    
    # Remove amounts from remaining text (process in reverse order to maintain indices)
    for match in reversed(amount_matches):
      amount = match.group().strip()
      if len(amount) > 0 and (amount.replace(',', '').replace('.', '').isdigit()):
        amounts.append(amount)
        remaining = remaining[:match.start()] + ' ' + remaining[match.end():]
    
    amounts.reverse()  # Restore original order
    
    # Clean up remaining text (should be description)
    description = re.sub(r'\s+', ' ', remaining).strip()
    
    # Build final parts list
    if description:
      parts.append(description)
    
    # Add amounts
    parts.extend(amounts)
    
    logger.debug(f"Smart split result: {parts}")
    
    return parts
  
  def _headers_match(self, row: List[str]) -> bool:
    """Check if a row matches our target headers."""
    if not row:
      return False
    
    # Clean and normalize the row
    cleaned_row = []
    for cell in row:
      if cell is not None:
        cleaned_cell = str(cell).strip().lower()
        # Remove extra whitespace and special characters
        cleaned_cell = re.sub(r'[^\w\s]', '', cleaned_cell)
        cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
        if cleaned_cell:  # Only add non-empty cells
          cleaned_row.append(cleaned_cell)
    
    if not cleaned_row:
      return False
    
    logger.info(f"Checking row: {cleaned_row} against headers: {self.normalized_headers}")
    
    # Exact match (flexible length)
    if len(cleaned_row) >= len(self.normalized_headers):
      # Check if our headers are a subset of the row (in order)
      row_subset = cleaned_row[:len(self.normalized_headers)]
      if row_subset == self.normalized_headers:
        logger.info(f"Exact match found!")
        return True
    
    # Fuzzy match - check if most headers are present
    matches = 0
    total_headers = len(self.normalized_headers)
    
    for i, header in enumerate(self.normalized_headers):
      found_match = False
      
      # Check each cell in the row for this header
      for j, cell in enumerate(cleaned_row):
        if (header in cell or cell in header or 
            self._headers_similar(header, cell) or
            self._is_synonym(header, cell)):
          found_match = True
          matches += 1
          break
      
      if found_match:
        logger.info(f"Found match for header '{header}'")
    
    match_ratio = matches / total_headers
    logger.info(f"Match ratio: {match_ratio} ({matches}/{total_headers})")
    
    # Require at least 60% match (was 80% - too strict)
    is_match = match_ratio >= 0.6
    if is_match:
      logger.info(f"✅ Headers match! Ratio: {match_ratio}")
    else:
      logger.info(f"❌ Headers don't match. Ratio: {match_ratio}")
    
    return is_match
  
  def _headers_similar(self, header1: str, header2: str) -> bool:
    """Check if two headers are similar."""
    # Remove common words and check similarity
    common_words = {'date', 'description', 'amount', 'balance', 'debit', 'credit', 'ref', 'reference'}
    
    h1_words = set(header1.split()) - common_words
    h2_words = set(header2.split()) - common_words
    
    if not h1_words or not h2_words:
      return False
    
    intersection = h1_words.intersection(h2_words)
    union = h1_words.union(h2_words)
    
    return len(intersection) / len(union) > 0.5
  
  def _is_synonym(self, header1: str, header2: str) -> bool:
    """Check if two headers are synonyms."""
    synonyms = {
      'date': ['date', 'transaction date', 'trans date', 'dt'],
      'description': ['description', 'desc', 'details', 'particulars', 'transaction', 'reference'],
      'debit': ['debit', 'debit amount', 'withdrawal', 'withdrawals', 'out', 'dr'],
      'credit': ['credit', 'credit amount', 'deposit', 'deposits', 'in', 'cr'],
      'balance': ['balance', 'running balance', 'current balance', 'bal', 'amount'],
      'amount': ['amount', 'value', 'sum', 'total']
    }
    
    for key, synonym_list in synonyms.items():
      if header1 in synonym_list and header2 in synonym_list:
        return True
    
    return False
  
  def _is_valid_transaction_row(self, row: List[str]) -> bool:
    """Check if a row contains valid transaction data."""
    if not row or len(row) < 2:
      return False
    
    # Convert row to strings and clean
    str_row = [str(cell).strip() if cell is not None else '' for cell in row]
    
    # Skip empty rows
    if all(cell == '' for cell in str_row):
      return False
    
    # Skip header rows
    if self._headers_match(str_row):
      return False
    
    # Look for date pattern in first few columns
    has_date = False
    for i, cell in enumerate(str_row[:3]):  # Check first 3 columns for date
      if re.search(r'\b(1[0-2]|0?[1-9])/(3[01]|[12][0-9]|0?[1-9])\b', cell):
        has_date = True
        break
      if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', cell):
        has_date = True
        break
    
    # Look for amount pattern somewhere in the row
    has_amount = False
    for cell in str_row:
      if re.search(r'\b\d+[.,]\d{2}\b', cell):
        has_amount = True
        break
    
    # Must have either a date or an amount to be considered valid transaction data
    is_valid = has_date or has_amount
    
    if is_valid:
      logger.info(f"Valid transaction row: {str_row}")
    
    return is_valid

  def _is_valid_data_row(self, row: List[str]) -> bool:
    """Legacy method - redirects to _is_valid_transaction_row for backward compatibility."""
    return self._is_valid_transaction_row(row)
  
  def _clean_row(self, row: List[str]) -> List[str]:
    """Clean and normalize a data row."""
    cleaned = []
    for cell in row:
      if cell is None:
        cleaned.append('')
      else:
        # Clean cell content
        cell_str = str(cell).strip()
        # Remove extra whitespace
        cell_str = re.sub(r'\s+', ' ', cell_str)
        cleaned.append(cell_str)
    
    # Ensure row has correct number of columns
    while len(cleaned) < len(self.headers):
      cleaned.append('')
    
    return cleaned[:len(self.headers)]
  
  def _create_dataframe(self, rows: List[List[str]]) -> pd.DataFrame:
    """Create a DataFrame from extracted rows."""
    if not rows:
      return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=self.headers)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    df = df[~(df.astype(str).eq('').all(axis=1))]
    
    return df
  
  def _deduplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    if df.empty:
      return df
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Remove near-duplicates (optional, more sophisticated)
    # This could be implemented based on specific business rules
    
    return df.reset_index(drop=True)

  def _extract_with_ocr(self, pdf_path: str) -> pd.DataFrame:
    """Extract tables using OCR as a fallback method."""
    if not OCR_AVAILABLE:
      logger.warning("OCR not available - install pytesseract, pdf2image, and poppler")
      return pd.DataFrame()
    
    logger.info(f"Attempting OCR extraction for {pdf_path}")
    
    try:
      ocr_converter = OCRBankStatementConverter(self.headers)
      result = ocr_converter.convert_pdf_with_ocr(pdf_path)
      
      if not result.empty:
        logger.info(f"OCR extracted {len(result)} rows")
      else:
        logger.info("OCR extraction found no data")
      
      return result
      
    except Exception as e:
      logger.error(f"OCR extraction failed: {str(e)}")
      return pd.DataFrame() 