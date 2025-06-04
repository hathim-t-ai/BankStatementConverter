"""Extracts transaction tables from PDFs.

Only tables whose header row matches the given headers are included.
"""

class BankStatementConverter:
  def __init__(self, headers):
    self.headers = [h.strip().lower() for h in headers]

  def _headers_match(self, row):
    normalized = [c.strip().lower() for c in row]
    return normalized == self.headers

  def convert(self, pdf_paths, csv_path):
    """Process multiple PDF files and write matching tables to a CSV."""
    import csv
    import pdfplumber

    with open(csv_path, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(self.headers)
      for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
          for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
              if not table:
                continue
              if self._headers_match(table[0]):
                for row in table[1:]:
                  writer.writerow(row)

