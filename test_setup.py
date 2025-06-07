#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the bank statement converter is ready to run.
"""

import sys
import importlib

# List of required packages (core functionality)
REQUIRED_PACKAGES = [
    'flask',
    'pdfplumber', 
    'pandas',
    'tabula',
    'PyPDF2',
    'numpy',
    'werkzeug',
    'gunicorn',
    'PIL',  # Pillow
    'pdfminer',
    'jpype'  # JPype1
]

# Optional packages (enhanced functionality)
OPTIONAL_PACKAGES = [
    'cv2',  # opencv-python (for Camelot)
    'magic',  # python-magic (for file type detection)
    'camelot'  # camelot-py (for lattice table extraction)
]

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing core package imports...")
    
    failed_imports = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package} - {str(e)}")
            failed_imports.append(package)
    
    print("\n🔧 Testing optional package imports...")
    optional_available = []
    
    for package in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
            optional_available.append(package)
        except ImportError:
            print(f"⚠️  {package} (optional) - not available")
    
    if failed_imports:
        print(f"\n❌ Failed to import required packages: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        if optional_available:
            print(f"✨ Optional features available: {', '.join(optional_available)}")
        return True

def test_advanced_converter():
    """Test if the advanced converter can be imported."""
    print("\n🧪 Testing advanced converter...")
    
    try:
        from bankstatementconverter.advanced_converter import AdvancedBankStatementConverter
        
        # Test basic initialization
        converter = AdvancedBankStatementConverter(['Date', 'Description', 'Amount'])
        print("✅ Advanced converter imported and initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced converter test failed: {str(e)}")
        return False

def test_flask_app():
    """Test if the Flask app can be imported."""
    print("\n🧪 Testing Flask application...")
    
    try:
        import app
        print("✅ Flask application imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {str(e)}")
        return False

def test_pdf_processing():
    """Test basic PDF processing capabilities."""
    print("\n🧪 Testing PDF processing capabilities...")
    
    try:
        import pdfplumber
        import pandas as pd
        print("✅ Core PDF processing libraries ready!")
        
        # Test if we have Java for Tabula
        try:
            import jpype
            if not jpype.isJVMStarted():
                jpype.startJVM()
            print("✅ Java environment ready for Tabula!")
            return True
        except Exception as e:
            print(f"⚠️  Java/Tabula may have issues: {str(e)}")
            print("    PDFPlumber and pattern matching will still work!")
            return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🏦 Bank Statement Converter - Setup Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_advanced_converter,
        test_flask_app,
        test_pdf_processing
    ]
    
    all_passed = True
    
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To start the application:")
        print("1. source venv/bin/activate")
        print("2. python app.py")
        print("3. Open http://localhost:5000 in your browser")
        print("\n📝 Note: Some optional features may not be available, but core functionality works!")
    else:
        print("❌ Some critical tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 