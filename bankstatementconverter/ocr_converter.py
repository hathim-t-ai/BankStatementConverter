"""OCR-based PDF converter for image-based PDFs."""

import os
import logging
from typing import List, Optional
import pandas as pd

# OCR imports with fallbacks
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRBankStatementConverter:
    """Converter that uses OCR for image-based PDFs."""
    
    def __init__(self, headers: List[str]):
        self.headers = [h.strip() for h in headers]
        self.normalized_headers = [h.strip().lower() for h in self.headers]
        
        if not OCR_AVAILABLE:
            logger.warning("OCR not available. Install: pip install pytesseract pdf2image")
    
    def convert_pdf_with_ocr(self, pdf_path: str) -> pd.DataFrame:
        """Convert image-based PDF using OCR."""
        
        if not OCR_AVAILABLE:
            logger.error("OCR libraries not available")
            return pd.DataFrame()
        
        logger.info(f"Starting OCR conversion of {pdf_path}")
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            logger.info(f"Converted PDF to {len(images)} images")
            
            all_rows = []
            
            for page_num, image in enumerate(images):
                logger.info(f"Processing page {page_num + 1} with OCR...")
                
                # Extract text using OCR
                text = pytesseract.image_to_string(image)
                logger.info(f"OCR extracted {len(text)} characters from page {page_num + 1}")
                
                if text.strip():
                    # Process extracted text
                    rows = self._extract_table_from_ocr_text(text)
                    all_rows.extend(rows)
                    logger.info(f"Extracted {len(rows)} rows from page {page_num + 1}")
            
            logger.info(f"Total rows extracted: {len(all_rows)}")
            return self._create_dataframe(all_rows)
            
        except Exception as e:
            logger.error(f"OCR conversion failed: {str(e)}")
            return pd.DataFrame()
    
    def _extract_table_from_ocr_text(self, text: str) -> List[List[str]]:
        """Extract table data from OCR text."""
        rows = []
        lines = text.split('\n')
        
        # Find header line
        header_line_idx = None
        for i, line in enumerate(lines):
            if self._line_contains_headers(line):
                header_line_idx = i
                logger.info(f"Found header line at {i}: {line.strip()}")
                break
        
        if header_line_idx is None:
            logger.info("No header line found in OCR text")
            return rows
        
        # Extract data lines
        for line in lines[header_line_idx + 1:]:
            if not line.strip():
                continue
            
            # Skip summary lines
            if any(keyword in line.lower() for keyword in ['total', 'subtotal', 'balance brought', 'page']):
                continue
            
            # Parse line as table row
            parsed_row = self._parse_ocr_line(line)
            if parsed_row and len(parsed_row) >= 3:
                # Ensure row matches expected columns
                while len(parsed_row) < len(self.headers):
                    parsed_row.append('')
                rows.append(parsed_row[:len(self.headers)])
        
        return rows
    
    def _line_contains_headers(self, line: str) -> bool:
        """Check if line contains table headers."""
        line_lower = line.lower()
        header_count = 0
        
        for header in self.normalized_headers:
            if header in line_lower:
                header_count += 1
        
        return header_count >= len(self.normalized_headers) * 0.6
    
    def _parse_ocr_line(self, line: str) -> Optional[List[str]]:
        """Parse OCR line into table columns with improved date extraction."""
        import re
        
        # Clean line
        line = line.strip()
        if not line:
            return None
        
        # Skip summary lines and non-data lines
        skip_terms = ['total', 'subtotal', 'balance brought', 'page', 'account #', 'continued', 'statement', 'summary']
        if any(term in line.lower() for term in skip_terms):
            return None
        
        logger.info(f"OCR parsing line: {line}")
        
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
                logger.info(f"OCR found date: {date_found}")
                break
        
        if not date_found:
            # If no date pattern found, skip this line for transaction data
            logger.info(f"OCR: No date found in line, skipping: {line}")
            return None
        
        # Find amounts in the remaining line
        amounts = []
        amount_line = remaining_line
        
        for pattern in amount_patterns:
            found_amounts = re.findall(pattern, amount_line)
            for amount in found_amounts:
                clean_amount = amount.replace('$', '').replace(',', '').strip()
                amounts.append(clean_amount)
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
            logger.info(f"OCR successfully parsed row: {row}")
            return row
        
        logger.info(f"OCR: Insufficient data in parsed row, skipping")
        return None
    
    def _create_dataframe(self, rows: List[List[str]]) -> pd.DataFrame:
        """Create DataFrame from extracted rows."""
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=self.headers)
        
        # Clean up
        df = df.dropna(how='all')
        df = df[~(df.astype(str).eq('').all(axis=1))]
        
        return df 