import os
import tempfile
import hashlib
from textwrap import dedent
from typing import TypedDict, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

from langchain_text_splitters import CharacterTextSplitter

from utils import draw_mermaid_png

################################ Configuration & Setup ################################

# Load environment variables from a .env file
load_dotenv()

# Define the default model to use for AI operations
DEFAULT_MODEL = "granite4:350m"
# Set the base URL for the Ollama service, using an environment variable or a default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Define default parameters for document processing
DEFAULT_MAX_INPUT_CHARS = 10_000
DEFAULT_CHUNK_CHARS = 3_500
DEFAULT_CHUNK_OVERLAP = 250

################################ Data Structures ################################

class AgentState(TypedDict, total=False):
    """
    Define the shared state structure for the LangGraph agent.
    Each key represents a piece of information passed between nodes.
    """
    original_text: str
    condensed_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str

################################ Core Utilities ################################

def file_hash(data: bytes) -> str:
    """
    Compute SHA256 hash of file data.
    Use the hash to uniquely identify and cache document processing results.
    """
    return hashlib.sha256(data).hexdigest()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a large text into smaller, overlapping chunks.
    Prepare text for map-reduce summarization to handle LLM context limits.
    """
    # Initialize a character-based text splitter
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    # Split the document text into manageable chunks
    return splitter.split_text(text)

@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    """
    Retrieve or create a cached Ollama language model instance.
    Avoid re-initializing the LLM for every call to improve performance.
    """
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)

def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    """
    Invoke the Ollama LLM with a given prompt and configuration.
    Encapsulate the LLM interaction for consistency.
    """
    llm = get_llm(model, base_url, temperature)
    # Call the LLM with the provided prompt
    resp = llm.invoke(prompt)
    # Extract and clean the LLM's response content
    return (resp.content or "").strip()

################################ Text Processing Functions ################################

def map_reduce_condense(
    text: str,
    model: str,
    base_url: str,
    temperature: float,
    chunk_chars: int,
    overlap: int,
) -> str:
    """
    Condense long text using a map-reduce strategy.
    Split text into chunks, summarize each chunk, then consolidate the summaries.
    """
    # Divide the input text into smaller, overlapping segments
    chunks = chunk_text(text, chunk_chars, overlap)

    partials = []
    # Summarize each individual text chunk
    for i, c in enumerate(chunks, start=1):
        # Construct a prompt for summarizing a single chunk
        prompt = dedent(f"""
        You are an expert legal assistant.
        Summarize this portion of a legal document. Preserve obligations, deadlines, termination, payment, liability, indemnity, IP, confidentiality, governing law, dispute resolution.
        Output as concise bullet points.

        [Chunk {i}/{len(chunks)}]
        {c}
        """)
        # Call the LLM to get a summary for the current chunk
        partials.append(call_ollama(prompt, model, base_url, temperature))

    # Consolidate all individual chunk summaries into one comprehensive summary
    reduce_prompt = dedent(f"""
    You are an expert legal assistant.
    Consolidate the following chunk summaries into a single coherent, non-redundant summary.
    Use short sections with headings:
    - Parties & Purpose
    - Key Obligations
    - Money (fees, payment terms, penalties)
    - Term, Renewal, Termination
    - Liability, Indemnity, Insurance
    - IP & Confidentiality
    - Disputes & Governing Law
    - Important Deadlines

    Chunk summaries:
    {"\n\n".join(partials)}
    """)
    # Call the LLM to perform the final reduction
    return call_ollama(reduce_prompt, model, base_url, temperature)


################################ AI Agent Nodes ################################

def summarize_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a concise executive summary of the legal document.
    Focus on practical meaning and major commitments.
    """
    # Use condensed text if available, otherwise use the original text
    condensed = state.get("condensed_text") or state["original_text"]
    # Construct a prompt to request an executive summary
    prompt = dedent(f"""
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.

    Document:
    {condensed}
    """)
    # Call the LLM to generate the summary
    return {
        "summary": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_summary
        )
    }

def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    """
    Identify potential legal risks and liabilities within the document.
    Provide details on each risk's severity, likelihood, and mitigation.
    """
    # Use condensed text if available, otherwise use the original text
    condensed = state.get("condensed_text") or state["original_text"]
    # Construct a prompt to request risk analysis
    prompt = dedent(f"""
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)

    Document:
    {condensed}
    """)
    # Call the LLM to analyze and describe the risks
    return {
        "risks": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_risks
        )
    }

def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    """
    Suggest improvements or identify missing protections in the legal document.
    Focus on specific clause-level suggestions, categorized by topic.
    """
    # Use condensed text if available, otherwise use the original text
    condensed = state.get("condensed_text") or state["original_text"]
    # Construct a prompt to request improvement suggestions
    prompt = dedent(f"""
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity, confidentiality, IP, dispute resolution, warranties).

    Document:
    {condensed}
    """)
    # Call the LLM to generate suggestions for improvement
    return {
        "suggestions": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            st.session_state.temp_suggestions,
        )
    }

def compile_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Assemble the final analysis report from individual agent outputs.
    Format the report with a clear structure and disclaimer.
    """
    # Combine the summary, risks, and suggestions into a single markdown report
    report = dedent(f"""
    # Legal Document Analysis (AI-Assisted)

    > Disclaimer: This is not legal advice. Consult a qualified attorney before acting.

    ## 📝 Document Summary
    {state.get("summary","")}

    ## ⚠️ Identified Risks
    {state.get("risks","")}

    ## 💡 Suggestions for Improvement
    {state.get("suggestions","")}
    """).strip()
    # Store the complete report in the agent state
    return {"final_report": report}

################################ Workflow Management ################################

@st.cache_resource
def get_workflow():
    """
    Define and compile the LangGraph workflow for legal document analysis.
    Cache the workflow to avoid re-creation on every run.
    """
    # Initialize the StateGraph with the defined agent state
    workflow = StateGraph(AgentState)
    # Add each analysis function as a node in the graph
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    # Set the starting point of the graph
    workflow.set_entry_point("summarize")
    # Define the sequence of execution for the nodes
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    # Mark the end of the workflow execution
    workflow.add_edge("compile_report", END)
    # Compile the graph for execution
    graph = workflow.compile()
    # Generate a Mermaid diagram for visualization of the workflow
    draw_mermaid_png(graph)
    return graph

################################ Document Handling ################################

def load_doc(uploaded_file) -> str:
    """
    Load content from an uploaded document, supporting TXT and PDF formats.
    Convert PDFs to markdown text using pymupdf4llm.
    """
    # Extract file extension to determine processing method
    suffix = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()

    # Handle plain text files directly
    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    tmp_path = None
    # Process PDF files by converting them to markdown
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Convert the temporary PDF file to markdown text
        text = pymupdf4llm.to_markdown(tmp_path)
        return text.strip()
    finally:
        # Ensure the temporary file is deleted after processing
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def analyze_document(doc_text: str) -> Dict[str, Any]:
    """
    Initiate the legal document analysis workflow with the provided text.
    Invoke the pre-defined LangGraph application.
    """
    # Retrieve the compiled LangGraph workflow
    app = get_workflow()
    # Execute the workflow with the original and potentially condensed text
    return app.invoke({"original_text": doc_text, "condensed_text": st.session_state.condensed_text})

################################ Streamlit Application ################################

def main():
    """
    Set up the main Streamlit application interface for legal document analysis.
    Handle file uploads, model configuration, analysis execution, and results display.
    """
    # Configure the Streamlit page title and layout
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")

    # Inject custom CSS for styling report boxes
    st.markdown(
        """
        <style>
        .report-box {
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Define the content for the Streamlit sidebar
    with st.sidebar:
        st.header("Upload Document")
        # Allow users to upload PDF or TXT files
        uploaded_file = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

        st.divider()
        st.subheader("Model Settings")
        # Allow users to configure the LLM model name
        st.session_state.model = st.text_input("Model", value=DEFAULT_MODEL)
        # Control generation temperature for summary
        st.session_state.temp_summary = st.slider("Summary temperature", 0.0, 1.0, 0.2, 0.05)
        # Control generation temperature for risks
        st.session_state.temp_risks = st.slider("Risks temperature", 0.0, 1.0, 0.2, 0.05)
        # Control generation temperature for suggestions
        st.session_state.temp_suggestions = st.slider("Suggestions temperature", 0.0, 1.0, 0.3, 0.05)

        st.divider()
        st.subheader("Long Document Handling")
        # Set the maximum number of characters to extract for preview
        max_chars = st.number_input("Max extracted chars (preview)", value=DEFAULT_MAX_INPUT_CHARS, step=1000)
        # Configure chunk size for map-reduce condensation
        chunk_chars = st.number_input("Chunk size (chars)", value=DEFAULT_CHUNK_CHARS, step=500)
        # Configure overlap between chunks for map-reduce condensation
        overlap = st.number_input("Chunk overlap (chars)", value=DEFAULT_CHUNK_OVERLAP, step=50)

    # Set the main title of the application
    st.title("⚖️ AI Legal Document Analyzer")

    # Process the document if a file has been uploaded
    if uploaded_file:
        data = uploaded_file.getvalue()
        # Compute a hash of the file content for caching results
        h = file_hash(data)
        # Initialize session state to store analysis results by file hash
        st.session_state.setdefault("results_by_hash", {})

        # Trigger document analysis when the button is clicked
        if st.button("Analyze Document"):
            # Display a progress bar during analysis
            prog = st.progress(0, text="Reading document...")
            # Load the document content
            doc_text = load_doc(uploaded_file)

            # Show a preview of the extracted text
            preview_text = doc_text[: int(max_chars)]
            with st.expander("Extracted text preview"):
                st.markdown(preview_text)

            prog.progress(25, text="Condensing long document...")
            # Condense the document if it's too long for direct LLM processing
            condensed = map_reduce_condense(
                doc_text,
                model=st.session_state.model,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2, # Use a fixed temperature for condensation
                chunk_chars=int(chunk_chars),
                overlap=int(overlap),
            )
            # Store the condensed text in session state
            st.session_state.condensed_text = condensed

            prog.progress(55, text="Running analysis agents...")
            # Run the LangGraph agent workflow for analysis
            result = analyze_document(doc_text)

            prog.progress(100, text="Done.")
            # Cache the analysis results using the file hash
            st.session_state.results_by_hash[h] = result

        # Display analysis results if they exist for the current file hash
        if h in st.session_state.results_by_hash:
            result = st.session_state.results_by_hash[h]

            st.markdown("## 📊 Analysis Report")
            # Create tabs to organize different sections of the report
            summary_tab, risks_tab, suggestions_tab, report_tab = st.tabs(
                ["Summary", "Risks", "Suggestions", "Full report"]
            )

            # Display the document summary in its dedicated tab
            with summary_tab:
                st.markdown("### 📝 Document Summary")
                st.markdown(f"<div class='report-box'>{result.get('summary','')}</div>", unsafe_allow_html=True)

            # Display the identified risks in its dedicated tab
            with risks_tab:
                st.markdown("### ⚠️ Identified Risks")
                st.markdown(f"<div class='report-box'>{result.get('risks','')}</div>", unsafe_allow_html=True)

            # Display suggestions for improvement in its dedicated tab
            with suggestions_tab:
                st.markdown("### 💡 Suggestions for Improvement")
                st.markdown(f"<div class='report-box'>{result.get('suggestions','')}</div>", unsafe_allow_html=True)

            # Display the full compiled report in its dedicated tab
            with report_tab:
                st.markdown(result.get("final_report", ""))

            # Provide a button to download the full report
            st.download_button("Download Report", result.get("final_report", ""), file_name="legal_analysis.md")

    # Prompt the user to upload a document if none is present
    else:
        st.warning("Upload a PDF or TXT document to begin.")

# Ensure the main application function runs when the script is executed
if __name__ == "__main__":
    main()
