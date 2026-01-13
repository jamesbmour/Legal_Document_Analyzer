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
- **Multi-agent workflow**: Uses LangGraph to orchestrate multiple specialized AI agents for comprehensive analysis ([see Agent Architecture](agents.md))
- **Document format support**: Handles PDF and TXT files with intelligent text extraction
- **Long document processing**: Uses chunking and map-reduce strategies for documents exceeding context limits
- **Customizable AI settings**: Adjust model parameters and temperatures for different analysis types
- **Interactive interface**: Clean Streamlit UI with tabbed results
- **Result caching**: Smart caching prevents redundant processing of the same documents

### Analysis Components
1. **Document Summary**: Concise executive summary highlighting key obligations and terms
2. **Risk Analysis**: Identifies legal risks with severity, likelihood, and mitigation suggestions
3. **Improvement Suggestions**: Specific clause-level recommendations organized by topic
4. **Full Report Generation**: Comprehensive markdown report with all analysis sections combined

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
├── app-v2.py               # Enhanced version with map-reduce and advanced settings
├── app-v3.py               # Latest simplified version
├── utils.py                # Utility functions (workflow visualization)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── README.md               # This file
├── agents.md               # Agent architecture documentation
├── LICENSE                 # License information
└── assets/                 # Sample documents and outputs
```

## 🔧 Installation

### Prerequisites
- **Python 3.12 or later**
- **Ollama** installed and running locally ([Install Ollama](https://ollama.ai/download))
- Required Ollama models (e.g., `granite4:350m`, `llama3.2:1b`)

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

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to configure your settings
   ```

4. **Install required Ollama models**:
   ```bash
   # Lightweight fast model (recommended for testing)
   ollama pull granite4:350m

   # Or use a more capable model
   ollama pull llama3.2:1b
   ```

5. **Run the application**:
   ```bash
   # Run the enhanced version with all features
   streamlit run app-v2.py

   # Or run the simplified version
   streamlit run app-v3.py
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

The application uses a **multi-agent workflow** powered by LangGraph:

1. **Document Loading**: Extract text from PDF/TXT files using PyMuPDF
2. **Text Condensation** (for long documents): Map-reduce summarization to handle context limits
3. **Sequential Agent Analysis**:
   - **Summarize Agent**: Generates executive summary (5-12 bullet points)
   - **Risk Analysis Agent**: Identifies legal risks with severity/likelihood ratings
   - **Suggestions Agent**: Recommends improvements organized by topic
4. **Report Compilation**: Assembles all analyses into a structured markdown report

For detailed information about the agent architecture, workflow execution, and extending the system, see **[Agent Architecture Documentation](agents.md)**.

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
- `granite4:350m` - Lightweight, fast model (default for app-v2.py)
- `llama3.2:1b` - Fast and capable, good balance (default for app-v3.py)
- `llama3.2:3b` - More capable for complex analysis
- `mistral` - Strong performance for legal text
- `qwen2.5` - Excellent instruction following

Configure the model in the sidebar or via the `.env` file.

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

### Planned Features
- Support for additional document formats (DOCX, scanned PDFs with OCR)
- Multi-document comparison and analysis
- Custom template support for specific document types (NDA, MSA, etc.)
- Integration with legal databases for clause validation
- Collaborative review features

### Agent System Enhancements
- Clause extraction and structuring agent
- Compliance checking agent for regulatory requirements
- Entity recognition agent (parties, dates, amounts)
- Q&A agent for document-specific questions
- Parallel agent execution for faster processing

See **[agents.md](agents.md)** for detailed enhancement opportunities and implementation guidance.

---

## 📚 Documentation

- **[README.md](README.md)** - This file, project overview and setup
- **[agents.md](agents.md)** - Detailed agent architecture and workflow documentation
- **[LICENSE](LICENSE)** - MIT License information

---

**Legal Document Analyzer** - Making legal analysis accessible through AI 🚀