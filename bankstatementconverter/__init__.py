"""
Bank Statement Converter Package

A sophisticated tool for extracting tabular data from PDF bank statements.
"""

from .converter import BankStatementConverter
from .advanced_converter import AdvancedBankStatementConverter

__version__ = "2.0.0"
__author__ = "Bank Statement Converter Team"

__all__ = [
    "BankStatementConverter",
    "AdvancedBankStatementConverter"
]

