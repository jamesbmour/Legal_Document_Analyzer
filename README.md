# Legal Document Analyzer

An AI-powered legal document analysis tool that helps users understand, evaluate risks, and improve legal contracts and agreements.

## 📋 Overview

The Legal Document Analyzer is a Streamlit application that uses LangChain, Ollama, and LangGraph to provide comprehensive analysis of legal documents. It can:

- **Summarize** legal documents concisely
- **Identify risks** and potential liabilities
- **Suggest improvements** for better legal protection
- **Handle long documents** using map-reduce strategies
- **Generate comprehensive reports** in Markdown format

## 🚀 Features

### Core Capabilities
- **Multi-agent workflow**: Uses LangGraph to orchestrate multiple AI agents for comprehensive analysis
- **Document format support**: Handles PDF and TXT files
- **Long document processing**: Uses chunking and map-reduce strategies for documents exceeding context limits
- **Customizable AI settings**: Adjust model parameters and temperatures for different analysis types
- **Interactive interface**: Clean Streamlit UI with tabbed results

### Analysis Components
1. **Document Summary**: Concise executive summary highlighting key obligations and terms
2. **Risk Analysis**: Identifies legal risks with severity and mitigation suggestions
3. **Improvement Suggestions**: Specific clause-level recommendations organized by topic
4. **Full Report Generation**: Comprehensive markdown report with all analysis sections

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Backend**: Ollama (local LLM) with LangChain integration
- **Workflow Orchestration**: LangGraph
- **Document Processing**: PyMuPDF4LLM for PDF-to-markdown conversion
- **Configuration**: Python-dotenv for environment variables

## 📁 Project Structure

```
Legal_Document_Analyzer/
├── app.py                  # Original application version
├── app-v2.py               # Enhanced version with advanced features
├── app-v3.py               # Latest version (if available)
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── README.md               # This file
├── LICENSE                 # License information
└── assets/                 # Sample documents and outputs
```

## 🔧 Installation

### Prerequisites
- Python 3.12 or later
- Ollama installed and running locally
- Required Ollama models (e.g., `granite4:350m`)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Legal_Document_Analyzer.git
   cd Legal_Document_Analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env to configure your settings
   ```

4. **Install required Ollama models**:
   ```bash
   ollama pull granite4:350m
   ```

5. **Run the application**:
   ```bash
   streamlit run app-v2.py
   ```

## 🎯 Usage

### Basic Workflow
1. **Upload a document**: PDF or TXT file containing legal content
2. **Configure settings** (optional): Adjust model parameters and analysis settings
3. **Click "Analyze Document"**: Let the AI process your document
4. **Review results**: Explore summary, risks, suggestions, and full report
5. **Download report**: Save the comprehensive analysis as a Markdown file

### Advanced Features
- **Long document handling**: Automatically condenses large documents using map-reduce
- **Temperature control**: Adjust creativity/precision for different analysis types
- **Model selection**: Choose different Ollama models based on your needs
- **Result caching**: Analysis results are cached by document hash for efficiency

## 📊 Analysis Process

The application uses a multi-agent workflow:

1. **Document Loading**: Extract text from PDF/TXT files
2. **Text Condensation** (for long documents): Map-reduce summarization
3. **Parallel Analysis**:
   - Summary generation
   - Risk identification  
   - Improvement suggestions
4. **Report Compilation**: Combine all analyses into final report

## 🔧 Configuration

### Environment Variables

Create a `.env` file with these variables:

```env
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# Default model
DEFAULT_MODEL=granite4:350m

# Document processing limits
MAX_INPUT_CHARS=10000
CHUNK_SIZE=3500
CHUNK_OVERLAP=250
```

### Model Selection

The application supports any Ollama-compatible model. Popular choices:
- `granite4:350m` - Lightweight, fast model (default)
- `llama3.2` - More capable but larger
- `mistral` - Balanced performance

## 📈 Performance Considerations

- **Document size**: Larger documents take longer to process
- **Model selection**: Larger models provide better quality but require more resources
- **Temperature settings**: Higher temperatures increase creativity but may reduce consistency
- **Caching**: Results are cached by document hash to avoid reprocessing

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the excellent UI framework
- LangChain and LangGraph for AI workflow orchestration
- Ollama for making local LLMs accessible
- PyMuPDF for PDF processing capabilities

## 📎 Sample Documents

The repository includes sample legal documents for testing:
- `Lease_Agreement.pdf` - Example lease agreement
- `test.pdf` - Additional test document

## 🚨 Disclaimer

This tool provides AI-assisted analysis but does not constitute legal advice. Always consult with a qualified attorney for professional legal guidance.

## 🔮 Future Enhancements

- Support for additional document formats (DOCX, scanned PDFs with OCR)
- Multi-document comparison and analysis
- Custom template support for specific document types
- Integration with legal databases for clause validation
- Collaborative review features

---

**Legal Document Analyzer** - Making legal analysis accessible through AI 🚀