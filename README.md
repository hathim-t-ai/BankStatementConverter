# 🏦 Advanced Bank Statement Converter

A sophisticated web-based tool that extracts tabular data from PDF bank statements using multiple AI-powered extraction strategies. This tool can handle both structured tables (with clear borders) and unstructured tabular data (column-based layout without lines).

## ✨ Features

- **🎯 Smart Header Detection**: Intelligent matching of user-specified column headers
- **🤖 Multiple Extraction Methods**: 
  - PDFPlumber for text-based tables
  - Tabula for structured tables (when Java is available)
  - Camelot for lattice/bordered tables (optional)
  - Pattern matching for unstructured data
- **🌐 Modern Web Interface**: Beautiful, responsive UI with drag-and-drop upload
- **📊 Real-time Processing**: Live progress updates and status tracking
- **📋 Data Preview**: Preview extracted data before downloading
- **🔄 Batch Processing**: Handle multiple PDF files simultaneously
- **🎨 Smart Deduplication**: Automatically remove duplicate entries
- **📱 Mobile Friendly**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- macOS, Linux, or Windows

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd BankStatementConverter
   ```

2. **Run the setup script (Recommended):**
   ```bash
   ./setup.sh
   ```

3. **Or manually set up:**
   ```bash
   # Create virtual environment
   python3.10 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### 🎬 Running the Application

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Start the web server:**
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:8080
   ```

## 📖 How to Use

1. **Specify Column Headers**: Enter the exact column headers as they appear in your bank statements (e.g., "Date, Description, Debit, Credit, Balance")

2. **Upload PDF Files**: 
   - Drag and drop PDF files onto the upload area, or
   - Click "Choose Files" to select files manually
   - Multiple files can be processed simultaneously

3. **Processing**: The tool will:
   - Analyze each PDF using multiple extraction strategies
   - Detect tables matching your specified headers
   - Extract and clean the data
   - Combine results from all files

4. **Download Results**: 
   - Preview the extracted data in a formatted table
   - Download the complete dataset as a CSV file

## 🏗️ Architecture

### Core Components

- **Flask Web Application** (`app.py`): Web server and API endpoints
- **Advanced Converter** (`bankstatementconverter/advanced_converter.py`): Multi-strategy PDF processing engine
- **Web Interface** (`templates/`): Modern, responsive frontend

### Extraction Strategies

1. **PDFPlumber**: Best for text-based tables and mixed content (always available)
2. **Tabula**: Excellent for well-structured tables with clear boundaries (requires Java)
3. **Camelot**: Specialized for lattice tables with borders (optional)
4. **Pattern Matching**: Handles unstructured tabular data using regex and smart parsing

## 🛠️ Development

### Project Structure
```
BankStatementConverter/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── setup.sh                       # Setup script
├── test_setup.py                  # Setup verification script
├── README.md                      # This file
├── bankstatementconverter/        # Core converter package
│   ├── __init__.py
│   ├── __main__.py               # CLI interface (legacy)
│   ├── converter.py              # Basic converter (legacy)
│   └── advanced_converter.py     # Advanced multi-strategy converter
├── templates/                    # HTML templates
│   ├── base.html                # Base template with styling
│   ├── index.html               # Main upload interface
│   └── preview.html             # Results preview page
├── uploads/                     # Temporary file storage
├── tests/                       # Unit tests
└── venv/                       # Virtual environment (created during setup)
```

### Running Tests
```bash
source venv/bin/activate
python test_setup.py  # Verify setup
python -m unittest discover tests  # Run unit tests
```

### CLI Usage (Legacy)
The original CLI interface is still available:
```bash
python -m bankstatementconverter file1.pdf file2.pdf --headers "Date,Description,Debit,Credit,Balance" --output output.csv
```

## 📊 Supported Bank Statement Formats

The tool can handle various bank statement formats:

- **Traditional Tables**: Clear rows and columns with borders
- **Unstructured Tables**: Column headers with space-separated data
- **Mixed Content**: Statements with multiple tables and text
- **Multi-page Documents**: Data spread across multiple pages

### Example Headers
- `Date, Description, Debit, Credit, Balance`
- `Transaction Date, Details, Amount, Running Balance`
- `Date, Reference, Particulars, Withdrawals, Deposits, Balance`

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development        # For development
export FLASK_DEBUG=1               # Enable debug mode
export MAX_CONTENT_LENGTH=50MB     # Maximum file upload size
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app

# Or using the built-in server (not recommended for production)
python app.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 5000 Occupied**: The app now uses port 8080 by default to avoid conflicts with AirTunes
2. **Installation Errors**: Ensure you're using Python 3.9+ and have activated the virtual environment
3. **PDF Processing Fails**: Some PDFs may be image-based or heavily formatted. Try different PDF files
4. **No Data Extracted**: Verify that your column headers exactly match those in the PDF
5. **Memory Issues**: For large files, increase system memory or process files individually
6. **OpenCV/Camelot Issues**: These are optional - core functionality works without them
7. **Java/Tabula Issues**: Tabula requires Java, but PDFPlumber provides excellent fallback functionality

### Optional Enhancements

To install optional features that may improve extraction quality:

```bash
# For better table detection (requires compilation time)
pip install opencv-python-headless camelot-py

# For Java-based table extraction (requires Java installation)
# Install Java from https://www.java.com first
```

### Getting Help
- Check the browser console for JavaScript errors
- Review the Flask application logs
- Ensure all dependencies are properly installed
- Run `python test_setup.py` to verify setup

## 🔮 Future Enhancements

- [ ] OCR support for image-based PDFs
- [ ] Machine learning-based table detection
- [ ] Support for additional file formats (Excel, CSV)
- [ ] Advanced data validation and cleaning
- [ ] Cloud deployment options
- [ ] User authentication and file management
- [ ] API endpoints for programmatic access

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for better financial data management**
