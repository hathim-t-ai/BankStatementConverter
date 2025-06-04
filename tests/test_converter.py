import os
import tempfile
import unittest
from fpdf import FPDF
from bankstatementconverter.converter import BankStatementConverter


class ConverterTest(unittest.TestCase):
  def _create_pdf(self, path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 10, 'Date Description Debit Credit Balance', ln=True)
    pdf.cell(0, 10, '2021-01-01 Test 10 0 90', ln=True)
    pdf.output(path)

  def test_convert(self):
    with tempfile.TemporaryDirectory() as tmp:
      pdf_path = os.path.join(tmp, 'sample.pdf')
      self._create_pdf(pdf_path)
      csv_path = os.path.join(tmp, 'out.csv')
      headers = ['Date', 'Description', 'Debit', 'Credit', 'Balance']
      converter = BankStatementConverter(headers)
      converter.convert([pdf_path], csv_path)
      with open(csv_path) as f:
        first_line = f.readline().strip()
      self.assertEqual(first_line.lower(), ','.join(h.lower() for h in headers))


if __name__ == '__main__':
  unittest.main()

