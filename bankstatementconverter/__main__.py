import argparse
from .converter import BankStatementConverter


def main():
  parser = argparse.ArgumentParser(description='Extract bank statement tables')
  parser.add_argument('pdfs', nargs='+', help='Input PDF files')
  parser.add_argument('--headers', required=True, help='Comma separated column headers')
  parser.add_argument('--output', required=True, help='Output CSV file')
  args = parser.parse_args()

  headers = [h.strip() for h in args.headers.split(',')]
  converter = BankStatementConverter(headers)
  converter.convert(args.pdfs, args.output)


if __name__ == '__main__':
  main()

