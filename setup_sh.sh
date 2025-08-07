#!/bin/bash

# Arabic DOCX to MP3 Converter - Setup Script
# This script sets up the environment in GitHub Codespaces or any Linux environment

echo "🎵 Arabic DOCX to MP3 Converter - Setup"
echo "======================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Update system packages
echo "📦 Updating system packages..."
sudo apt update -qq

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y ffmpeg python3-pip python3-venv

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_status "Python 3 is available"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p output
print_status "Output directory created"

# Check for Google Cloud credentials
echo "🔐 Checking Google Cloud credentials..."
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -z "$GOOGLE_TTS_API_KEY" ]; then
    print_warning "Google Cloud credentials not found"
    echo ""
    echo "Please set up Google Cloud credentials by:"
    echo "1. Setting GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON path)"
    echo "2. OR setting GOOGLE_TTS_API_KEY environment variable (API key)"
    echo ""
    echo "Example:"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account-key.json\""
    echo "OR"
    echo "export GOOGLE_TTS_API_KEY=\"your-api-key-here\""
else
    print_status "Google Cloud credentials found"
fi

# Check for input file
if [ ! -f "book.docx" ]; then
    print_warning "Input file 'book.docx' not found"
    echo "Please place your Arabic DOCX file in the project directory and name it 'book.docx'"
else
    print_status "Input file 'book.docx' found"
fi

# Test imports
echo "🧪 Testing Python imports..."
python3 -c "
try:
    from docx import Document
    from google.cloud import texttospeech
    from pydub import AudioSegment
    import tqdm
    print('✅ All Python packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_status "All dependencies are working correctly"
else
    print_error "Some dependencies failed to import"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Set up Google Cloud credentials (if not done already)"
echo "2. Place your Arabic DOCX file as 'book.docx'"
echo "3. Run: python main.py"
echo ""
echo "For detailed instructions, see README.md"
