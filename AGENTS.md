# AGENTS.md

## Project overview

The Legal Document Analyzer is an AI-powered tool designed to help users understand, evaluate risks, and improve legal contracts and agreements. This project leverages modern AI technologies including LangChain, LangGraph, and Ollama to provide comprehensive legal document analysis.

### Key Features
- **Multi-agent workflow**: Uses LangGraph to orchestrate multiple AI agents for comprehensive analysis
- **Document format support**: Handles PDF and TXT files with potential for additional formats
- **Long document processing**: Uses chunking and map-reduce strategies for documents exceeding context limits
- **Risk identification**: Automatically detects legal risks with severity assessments
- **Improvement suggestions**: Provides specific clause-level recommendations
- **Comprehensive reporting**: Generates detailed Markdown reports with all analysis sections

## Preferences and dependencies

### Technical Requirements
1. **Python Version**: Use Python 3.12 or later
2. **Dependency Management**: Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**: Load environment variables from a `.env` file for API keys and configurations.

4. **LLM Backend**: Use Ollama as the local LLM backend with fallback to OpenAI.

### Recommended AI Models
```Python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

# Primary Ollama models
llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')

# Fallback OpenAI model
llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=500)
```

### Required Dependencies
- `streamlit`: Web application framework
- `langchain`: AI workflow orchestration
- `langchain-ollama`: Ollama integration
- `langchain-openai`: OpenAI integration
- `langgraph`: Multi-agent workflow management
- `python-dotenv`: Environment variable management
- `pypdf2`: PDF processing
- `python-docx`: DOCX processing
- `pytesseract`: OCR for scanned documents
- `pillow`: Image processing
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `pymupdf4llm`: Advanced PDF processing

## Project structure

```
Legal_Document_Analyzer/
├── app.py                  # Original application version
├── app-v2.py               # Enhanced version with advanced features
├── app-v3.py               # Latest version (if available)
├── utils.py                # Utility functions and helpers
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── AGENTS.md               # Agent configuration and guidelines
├── README.md               # Project documentation
├── LICENSE                 # License information
├── Dockerfile              # Container configuration
├── assets/                 # Sample documents and outputs
│   ├── Lease_Agreement.pdf  # Example legal document
│   ├── test.pdf             # Test document
│   └── output.md            # Sample analysis output
└── scratch.py              # Development/testing scripts
```

## Key files and their purposes

### Core Application Files
- **app.py**: Original implementation of the legal document analyzer
- **app-v2.py**: Enhanced version with improved features and multi-agent workflow
- **app-v3.py**: Latest version with additional capabilities
- **utils.py**: Contains utility functions for document processing, text extraction, and helper methods

### Configuration Files
- **requirements.txt**: Lists all Python dependencies required for the project
- **.env**: Environment variables for API keys, model configurations, and settings
- **Dockerfile**: Container configuration for deployment

### Documentation and Assets
- **AGENTS.md**: Agent configuration guidelines and project standards (this file)
- **README.md**: Comprehensive project documentation and usage instructions
- **LICENSE**: Legal licensing information (MIT License)
- **Lease_Agreement.pdf**: Sample legal document for testing
- **test.pdf**: Additional test document
- **output.md**: Sample analysis output format

## Build and test commands

### Installation and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Legal_Document_Analyzer.git
cd Legal_Document_Analyzer

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to configure your settings

# Install required Ollama models
ollama pull granite4:350m
ollama pull llama3.2
```

### Running the Application
```bash
# Run the main application
streamlit run app-v2.py

# Run specific versions
streamlit run app.py          # Original version
streamlit run app-v3.py       # Latest version
```

### Testing and Development
```bash
# Run development server with auto-reload
streamlit run app-v2.py --server.runOnSave true

# Test document processing
python scratch.py

# Run specific analysis components
python -c "from utils import process_document; process_document('test.pdf')"
```

### Docker Deployment
```bash
# Build Docker image
docker build -t legal-document-analyzer .

# Run container
docker run -p 8501:8501 -v $(pwd):/app legal-document-analyzer
```
