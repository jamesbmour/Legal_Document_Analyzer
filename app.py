import os
import tempfile
from textwrap import dedent
from typing import TypedDict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm
from langchain_text_splitters import CharacterTextSplitter

#%%
################################ Configuration & Setup ################################

load_dotenv()

# Define model parameters and connection strings
llm_model = "granite4:350m"  # Using a lightweight model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_INPUT_CHARS = 10_000  # Prevent overly long prompts


def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    """Invoke the configured Ollama model via LangChain and return text content."""
    llm = ChatOllama(
        model=llm_model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
    response = llm.invoke(prompt)
    content = response.content
    return content.strip()


def clamp_text(text: str) -> str:
    """Limit the amount of text sent to the model to avoid context overflow."""
    return text if len(text) <= MAX_INPUT_CHARS else text[:MAX_INPUT_CHARS]


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


# Define the shared state schema for the LangGraph agents
class AgentState(TypedDict):
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str

#%%
################################ Graph Node Functions ################################

def summarize_node(state: AgentState):
    """Summarizes the legal document."""
    text = clamp_text(state["original_text"])
    prompt = dedent(
        f"""
        You are an expert legal assistant. Summarize the following legal document concisely:

        {text}
        """
    )

    return {"summary": call_ollama(prompt)}


def analyze_risks_node(state: AgentState):
    """Identifies potential risks in the document."""
    text = clamp_text(state["original_text"])
    prompt = dedent(
        f"""
        You are an expert legal assistant. Identify key legal risks and liabilities in this document:

        {text}
        """
    )
    return {"risks": call_ollama(prompt)}


def suggest_improvements_node(state: AgentState):
    """Suggests improvements for the document."""
    text = clamp_text(state["original_text"])
    prompt = dedent(
        f"""
        You are an expert legal assistant. Suggest clause improvements or missing protections for this document:

        {text}
        """
    )
    return {"suggestions": call_ollama(prompt, temperature=0.3)}


def compile_report_node(state: AgentState):
    """Compiles the final markdown report."""
    # Aggregate outputs from all previous nodes into a single Markdown string
    report = dedent(
        f"""
        ### 📝 Document Summary
        {state["summary"]}

        ### ⚠️ Identified Risks
        {state["risks"]}

        ### 💡 Suggestions for Improvement
        {state["suggestions"]}
        """
    )
    return {"final_report": report}

#%%
################################ Workflow Construction ################################

def create_workflow():
    # Initialize the state machine
    workflow = StateGraph(AgentState)

    # Register processing nodes
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    # Define the sequential execution path
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()

#%%
################################ Helper Functions ################################

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


def analyze_document(doc_text):
    """
    Analyzes the document and returns the final report.
    """
    app = create_workflow()
    initial_state = {"original_text": doc_text}

    # Execute the graph workflow
    result = app.invoke(initial_state)


    st.markdown("## 📊 Analysis Report")

    # Present results in dedicated tabs instead of a single block
    summary_tab, risks_tab, suggestions_tab = st.tabs(
        ["Summary", "Risks", "Suggestions"]
    )
    with summary_tab:
        st.markdown("### 📝 Document Summary")
        st.markdown(
            f"<div class='report-box'>{result['summary']}</div>",
            unsafe_allow_html=True,
        )

    with risks_tab:
        st.markdown("### ⚠️ Identified Risks")
        st.markdown(
            f"<div class='report-box'>{result['risks']}</div>",
            unsafe_allow_html=True,
        )

    with suggestions_tab:
        st.markdown("### 💡 Suggestions for Improvement")
        st.markdown(
            f"<div class='report-box'>{result['suggestions']}</div>",
            unsafe_allow_html=True,
        )

    # Provide a download option for the generated analysis
    st.download_button(
        "Download Report", result["final_report"], file_name="legal_analysis.md"
    )

#%%
################################ Main App Interface ################################

def main():
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")




    # Configure the sidebar for file uploads
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Upload Legal Document (PDF or TXT)", type=["pdf", "txt"]
        )

        st.text(f"Model: {llm_model}")

    st.title("⚖️ AI Legal Document Analyzer")
    st.markdown(
        "Upload a contract or legal document to get an instant AI-powered analysis."
    )
    with st.expander("Session State"):
        st.json(st.session_state, expanded=False)

    # Trigger processing only when a file exists and the button is clicked
    if uploaded_file and st.button("Analyze Document"):
        with st.spinner("Reading document..."):
            doc_text = load_doc(uploaded_file)

        # Validate that text was successfully extracted
        if not doc_text.strip():
            st.error("Could not extract text from document.")
        else:
            with st.spinner("Analyzing with AI Agents..."):
                analyze_document(doc_text)


if __name__ == "__main__":
    main()
