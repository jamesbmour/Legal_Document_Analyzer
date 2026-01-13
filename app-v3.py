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

################################ Configuration & Setup ################################

load_dotenv()

DEFAULT_MODEL = "granite4:350m"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

################################ Data Structures ################################

class AgentState(TypedDict, total=False):
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str

################################ Core Utilities ################################

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)

def call_ollama(prompt: str, model: str, base_url: str, temperature: float) -> str:
    llm = get_llm(model, base_url, temperature)
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()


################################ AI Agent Nodes ################################

def summarize_node(state: AgentState) -> Dict[str, Any]:
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.

    Document:
    {text}
    """)
    return {
        "summary": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_summary
        )
    }

def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)

    Document:
    {text}
    """)
    return {
        "risks": call_ollama(
            prompt, st.session_state.model, OLLAMA_BASE_URL, st.session_state.temp_risks
        )
    }

def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    text = state["original_text"]
    prompt = dedent(f"""
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity, confidentiality, IP, dispute resolution, warranties).

    Document:
    {text}
    """)
    return {
        "suggestions": call_ollama(
            prompt,
            st.session_state.model,
            OLLAMA_BASE_URL,
            st.session_state.temp_suggestions,
        )
    }

def compile_report_node(state: AgentState) -> Dict[str, Any]:
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
    return {"final_report": report}

################################ Workflow Management ################################

@st.cache_resource
def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()

################################ Document Handling ################################

def load_doc(uploaded_file) -> str:
    suffix = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()

    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    text = pymupdf4llm.to_markdown(tmp_path)
    os.remove(tmp_path)
    return text.strip()

def analyze_document(doc_text: str) -> Dict[str, Any]:
    app = get_workflow()
    return app.invoke({"original_text": doc_text})

################################ Streamlit Application ################################

def main():
    st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")

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

    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

        st.divider()
        st.subheader("Model Settings")
        st.session_state.model = st.text_input("Model", value=DEFAULT_MODEL)
        st.session_state.temp_summary = st.slider("Summary temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state.temp_risks = st.slider("Risks temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state.temp_suggestions = st.slider("Suggestions temperature", 0.0, 1.0, 0.3, 0.05)

    st.title("⚖️ AI Legal Document Analyzer")

    if uploaded_file:
        data = uploaded_file.getvalue()
        h = file_hash(data)
        st.session_state.setdefault("results_by_hash", {})

        if st.button("Analyze Document"):
            prog = st.progress(0, text="Reading document...")
            doc_text = load_doc(uploaded_file)

            prog.progress(50, text="Running analysis agents...")
            result = analyze_document(doc_text)

            prog.progress(100, text="Done.")
            st.session_state.results_by_hash[h] = result

        if h in st.session_state.results_by_hash:
            result = st.session_state.results_by_hash[h]

            st.markdown("## 📊 Analysis Report")
            summary_tab, risks_tab, suggestions_tab, report_tab = st.tabs(
                ["Summary", "Risks", "Suggestions", "Full report"]
            )

            with summary_tab:
                st.markdown("### 📝 Document Summary")
                st.markdown(result.get('summary',''))

            with risks_tab:
                st.markdown("### ⚠️ Identified Risks")
                st.markdown(result.get('risks',''))

            with suggestions_tab:
                st.markdown("### 💡 Suggestions for Improvement")
                st.markdown(result.get('suggestions',''))
            with report_tab:
                st.markdown(result.get("final_report", ""))

            st.download_button("Download Report", result.get("final_report", ""), file_name="legal_analysis.md")

    else:
        st.warning("Upload a PDF or TXT document to begin.")

if __name__ == "__main__":
    main()
