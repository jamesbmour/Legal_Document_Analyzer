import os
import tempfile
import hashlib
from textwrap import dedent
from typing import TypedDict, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import pymupdf4llm

################## Configuration & Setup ##################
# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "llama3.2:1b"
# Retrieve OLLAMA_BASE_URL from environment, default to localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUMMARY_TEMPERATURE = 0.2
RISKS_TEMPERATURE = 0.2
SUGGESTIONS_TEMPERATURE = 0.3

def ensure_session_defaults():
    """Ensure default values are set in Streamlit session state.
    """
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("temp_summary", SUMMARY_TEMPERATURE)
    st.session_state.setdefault("temp_risks", RISKS_TEMPERATURE)
    st.session_state.setdefault("temp_suggestions", SUGGESTIONS_TEMPERATURE)
    
################## Data Structures ###############


class AgentState(TypedDict, total=False):
    """
    Represents the state of the agent during processing.
    Contains the original text and generated outputs at each step.
    """
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str

#################### Core Utilities ###################


def file_hash(data: bytes) -> str:
    """ Generate a SHA256 hash for the given binary data.

    Args:
        data (bytes): _description_

    Returns:
        str: _description_
    """


@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    """Initialize and cache Ollama LLM instance."""


def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    """Invoke LLM with prompt and return response content."""


######## AI Agent Nodes #######################

def summarize_node(state: AgentState) -> Dict[str, Any]:
    """Generate executive summary of legal document."""


def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    """Identify legal risks and liabilities in document."""


def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    """Suggest improvements and missing protections for document."""


def compile_report_node(state: AgentState) -> Dict[str, Any]:
    """Compile final analysis report from summary, risks, and suggestions."""


################################ Workflow Management ################################


@st.cache_resource
def get_workflow():
    """Initialize and configure LangGraph workflow for document analysis."""


################################ Document Handling ################################


def load_doc(uploaded_file) -> str:
    """Extract text from uploaded PDF or TXT file."""


def analyze_document(doc_text: str) -> Dict[str, Any]:
    """Run AI workflow to analyze document text."""


################################ Streamlit Application ################################


def display_analysis_results(h):
    """Display analysis results in organized tabs with download option."""


def analyze_uploaded_file(uploaded_file, h):
    """Process uploaded file and run AI analysis."""


def render_sidebar_inputs():
    """Render sidebar with file upload and model settings."""


def main():
    """Main Streamlit application entry point."""


# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()