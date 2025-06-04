# Bank Statement Converter

This project extracts tables from PDF bank statements and combines them into a single CSV.
It searches for tables whose first row matches user provided headers.

## Usage

```
python -m bankstatementconverter file1.pdf file2.pdf --headers "Date,Description,Debit,Credit,Balance" --output output.csv
```

## Development

Unit tests are located in the `tests` folder and can be run with:

```
python -m unittest
```
