# Agent Architecture Documentation

## Overview

The Legal Document Analyzer uses a **multi-agent architecture** built on [LangGraph](https://langchain-ai.github.io/langgraph/) to orchestrate specialized AI agents for comprehensive legal document analysis. Each agent is responsible for a specific analysis task, and they work together in a sequential workflow to produce a complete legal analysis report.

## Architecture Pattern

This application implements a **Sequential Agent Pipeline** pattern where:
- Each agent specializes in one aspect of legal analysis
- Agents execute in a predetermined sequence
- All agents share a common state object (`AgentState`)
- The workflow is deterministic and reproducible

## Agent State

All agents share a common state structure defined by the `AgentState` TypedDict:

```python
class AgentState(TypedDict, total=False):
    original_text: str      # Raw extracted document text
    condensed_text: str     # Map-reduced condensed version (for long docs)
    summary: str            # Executive summary output
    risks: str              # Risk analysis output
    suggestions: str        # Improvement suggestions output
    final_report: str       # Compiled markdown report
```

### State Flow
- **Input**: `original_text` (and optionally `condensed_text`)
- **Intermediate**: Each agent adds its output to the state
- **Output**: `final_report` containing all analysis sections

## Agent Nodes

### 1. Summarize Agent (`summarize_node`)

**Purpose**: Generate a concise executive summary of the legal document

**Responsibilities**:
- Extract key obligations and terms
- Highlight major commitments
- Provide practical meaning of the agreement

**Implementation** (`app-v2.py:145-166`, `app-v3.py:71-89`):
```python
def summarize_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = """
    You are an expert legal assistant.
    Produce a concise executive summary of this legal document (5–12 bullet points max).
    Focus on practical meaning and major commitments.
    """
    return {"summary": call_ollama(prompt, model, base_url, temp_summary)}
```

**Configuration**:
- Temperature: 0.2 (low for factual, consistent summaries)
- Model: User-configurable (default: `granite4:350m` or `llama3.2:1b`)

**Output Format**: 5-12 bullet points highlighting key document aspects

---

### 2. Risk Analysis Agent (`analyze_risks_node`)

**Purpose**: Identify and assess potential legal risks and liabilities

**Responsibilities**:
- Identify legal risks in the document
- Assess severity and likelihood of each risk
- Provide brief mitigation strategies

**Implementation** (`app-v2.py:168-193`, `app-v3.py:92-111`):
```python
def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = """
    You are an expert legal assistant.
    Identify key legal risks and liabilities. For each item include:
    - Risk
    - Why it matters
    - Severity: Low/Med/High
    - Likelihood: Low/Med/High
    - Suggested mitigation (1 line)
    """
    return {"risks": call_ollama(prompt, model, base_url, temp_risks)}
```

**Configuration**:
- Temperature: 0.2 (low for factual, reliable risk assessment)
- Model: User-configurable

**Output Format**: Structured risk entries with severity, likelihood, and mitigation

---

### 3. Improvement Suggestions Agent (`suggest_improvements_node`)

**Purpose**: Recommend enhancements and identify missing legal protections

**Responsibilities**:
- Suggest specific clause-level improvements
- Identify missing protections
- Organize recommendations by legal topic

**Implementation** (`app-v2.py:195-219`, `app-v3.py:114-132`):
```python
def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    condensed = state.get("condensed_text") or state["original_text"]
    prompt = """
    You are an expert legal assistant.
    Suggest improvements or missing protections. Prefer specific clause-level suggestions.
    Organize by topic (payment, termination, limitation of liability, indemnity,
    confidentiality, IP, dispute resolution, warranties).
    """
    return {"suggestions": call_ollama(prompt, model, base_url, temp_suggestions)}
```

**Configuration**:
- Temperature: 0.3 (slightly higher for creative suggestions)
- Model: User-configurable

**Output Format**: Topic-organized improvement suggestions

**Topics Covered**:
- Payment terms
- Termination clauses
- Limitation of liability
- Indemnification
- Confidentiality
- Intellectual property
- Dispute resolution
- Warranties

---

### 4. Report Compilation Agent (`compile_report_node`)

**Purpose**: Assemble all analysis outputs into a final markdown report

**Responsibilities**:
- Combine summary, risks, and suggestions
- Format as structured markdown
- Add disclaimer and proper headings

**Implementation** (`app-v2.py:221-242`, `app-v3.py:135-151`):
```python
def compile_report_node(state: AgentState) -> Dict[str, Any]:
    report = f"""
    # Legal Document Analysis (AI-Assisted)

    > Disclaimer: This is not legal advice. Consult a qualified attorney before acting.

    ## 📝 Document Summary
    {state.get("summary","")}

    ## ⚠️ Identified Risks
    {state.get("risks","")}

    ## 💡 Suggestions for Improvement
    {state.get("suggestions","")}
    """
    return {"final_report": report}
```

**Configuration**: No LLM calls (pure templating)

**Output Format**: Markdown-formatted complete analysis report

---

## Workflow Execution

### LangGraph Workflow Definition

The workflow is defined using LangGraph's `StateGraph` (`app-v2.py:246-272`, `app-v3.py:158-172`):

```python
@st.cache_resource
def get_workflow():
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("suggest_improvements", suggest_improvements_node)
    workflow.add_node("compile_report", compile_report_node)

    # Define execution sequence
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze_risks")
    workflow.add_edge("analyze_risks", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()
```

### Execution Flow Diagram

```
┌─────────────────┐
│  User Upload    │
│   Document      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Extract │
│      Text       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Map-Reduce     │◄─── (Only for long documents)
│  Condensation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   START AGENT   │
│    WORKFLOW     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Summarize Node │──── Generates executive summary
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Risks   │──── Identifies legal risks
│     Node        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Suggest       │──── Recommends improvements
│ Improvements    │
│     Node        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compile Report  │──── Assembles final report
│     Node        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   END (Output   │
│  Final Report)  │
└─────────────────┘
```

### Sequential vs Parallel Execution

**Current Implementation**: **Sequential**
- Agents execute one after another in a fixed order
- Each agent completes before the next begins
- State is passed linearly through the pipeline

**Rationale for Sequential Design**:
- Ensures consistent, predictable results
- Simpler debugging and state management
- All agents work on the same condensed text
- Report compilation requires all previous outputs

**Potential for Parallelization**:
The three analysis agents (summarize, analyze_risks, suggest_improvements) could theoretically run in parallel since they don't depend on each other's outputs. However, the current sequential design prioritizes:
- Simplicity
- Deterministic execution order
- Resource efficiency (one LLM call at a time)

---

## Document Processing Pipeline

### Long Document Handling

For documents exceeding the LLM context limit, the application uses a **map-reduce condensation strategy** before agent execution:

#### Map Phase (`app-v2.py:92-141`)
1. **Split** document into overlapping chunks
2. **Summarize** each chunk individually with legal-specific prompts
3. **Preserve** critical legal elements (obligations, deadlines, liability, etc.)

#### Reduce Phase
1. **Consolidate** chunk summaries into a coherent whole
2. **Organize** by structured legal topics:
   - Parties & Purpose
   - Key Obligations
   - Money (fees, payment, penalties)
   - Term, Renewal, Termination
   - Liability, Indemnity, Insurance
   - IP & Confidentiality
   - Disputes & Governing Law
   - Important Deadlines

**Configuration** (adjustable in sidebar):
- `max_chars`: Maximum characters for preview (default: 10,000)
- `chunk_chars`: Characters per chunk (default: 3,500)
- `overlap`: Overlap between chunks (default: 250)

---

## LLM Integration

### Ollama Backend

All agents use **Ollama** for local LLM execution via LangChain's `ChatOllama`:

```python
@st.cache_resource
def get_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)
```

**Benefits of Local LLM**:
- Privacy: Documents never leave your machine
- Cost: No API fees
- Control: Full control over model selection
- Speed: Low latency for local inference

### Temperature Settings

Different agents use different temperature settings to optimize for their task:

| Agent | Temperature | Rationale |
|-------|------------|-----------|
| Condensation | 0.2 | Factual, preserves key information |
| Summary | 0.2 | Consistent, accurate summaries |
| Risk Analysis | 0.2 | Reliable, conservative risk assessment |
| Suggestions | 0.3 | Slightly creative for improvement ideas |

**User Control**: All temperatures are adjustable via sidebar sliders in `app-v2.py`

---

## Caching Strategy

### Result Caching

Analysis results are cached by document hash to avoid redundant processing:

```python
def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# Cache results in session state
st.session_state.results_by_hash[hash] = result
```

**Benefits**:
- Instant retrieval for previously analyzed documents
- Reduced LLM compute costs
- Improved user experience

### Workflow Caching

The LangGraph workflow is cached using Streamlit's `@st.cache_resource`:
- Workflow compiled once per session
- Reduces initialization overhead
- Shared across all document analyses

---

## Extending the Agent System

### Adding New Agents

To add a new agent to the workflow:

1. **Define the agent function**:
```python
def new_agent_node(state: AgentState) -> Dict[str, Any]:
    text = state.get("condensed_text") or state["original_text"]
    prompt = "Your custom prompt here"
    return {"new_output_key": call_ollama(prompt, model, base_url, temperature)}
```

2. **Update AgentState**:
```python
class AgentState(TypedDict, total=False):
    # ... existing fields
    new_output_key: str
```

3. **Add to workflow**:
```python
workflow.add_node("new_agent", new_agent_node)
workflow.add_edge("some_node", "new_agent")
workflow.add_edge("new_agent", "next_node")
```

4. **Update report compilation** to include new output

### Parallel Agent Execution

To run agents in parallel (e.g., summarize, analyze_risks, suggest_improvements):

```python
from langgraph.graph import END, StateGraph
from typing import List

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("summarize", summarize_node)
workflow.add_node("analyze_risks", analyze_risks_node)
workflow.add_node("suggest_improvements", suggest_improvements_node)
workflow.add_node("compile_report", compile_report_node)

# Set entry point
workflow.set_entry_point("summarize")

# Use conditional edges for parallel execution
workflow.add_conditional_edges(
    "summarize",
    lambda x: ["analyze_risks", "suggest_improvements"],
    {
        "analyze_risks": "analyze_risks",
        "suggest_improvements": "suggest_improvements"
    }
)

# Compile waits for all branches before continuing
workflow.add_edge("analyze_risks", "compile_report")
workflow.add_edge("suggest_improvements", "compile_report")
workflow.add_edge("compile_report", END)
```

---

## Performance Considerations

### Execution Time

Typical execution time for a medium-sized legal document (10-20 pages):
- **Document loading**: 1-2 seconds
- **Map-reduce condensation**: 10-30 seconds (if needed)
- **Summarize agent**: 5-15 seconds
- **Risk analysis agent**: 5-15 seconds
- **Suggestions agent**: 5-15 seconds
- **Report compilation**: <1 second
- **Total**: ~25-75 seconds

**Factors affecting performance**:
- Document length
- Model size (larger models are slower but more capable)
- Local hardware (CPU/GPU)
- Chunk size (larger chunks = fewer LLM calls but longer per call)

### Resource Usage

- **Memory**: Primarily determined by model size and Streamlit caching
- **CPU/GPU**: LLM inference is the primary bottleneck
- **Disk**: Minimal (temporary PDF files, cached workflow graph)

---

## Monitoring and Debugging

### Workflow Visualization

The application can generate a Mermaid diagram of the workflow:

```python
from utils import draw_mermaid_png

graph = workflow.compile()
png_bytes = draw_mermaid_png(graph)
```

This creates `graph.png` showing the complete agent execution flow.

### State Inspection

During development, you can inspect the agent state at any point:

```python
result = app.invoke({"original_text": doc_text})
print(result.keys())  # ['original_text', 'summary', 'risks', 'suggestions', 'final_report']
```

### Progress Tracking

The Streamlit UI shows real-time progress:
- Reading document... (0%)
- Condensing long document... (25%)
- Running analysis agents... (55%)
- Done (100%)

---

## Best Practices

### For Users

1. **Model Selection**: Start with smaller models (`granite4:350m`) for speed, use larger models for quality
2. **Temperature Tuning**: Keep temperatures low (0.1-0.3) for legal analysis
3. **Document Preparation**: Clean PDFs produce better results than scanned documents
4. **Chunk Settings**: Increase chunk size for better context, but watch for token limits

### For Developers

1. **Agent Isolation**: Keep agents stateless and independent
2. **Prompt Engineering**: Be specific about desired output format in prompts
3. **Error Handling**: Wrap LLM calls in try-except for robustness
4. **Testing**: Use sample documents to validate agent outputs
5. **Versioning**: Cache workflow by configuration to handle setting changes

---

## Future Enhancements

### Potential Agent Additions

1. **Clause Extraction Agent**: Identify and structure key clauses
2. **Compliance Checking Agent**: Verify against regulatory requirements
3. **Comparison Agent**: Compare multiple versions of a document
4. **Contract Classification Agent**: Identify document type (NDA, MSA, etc.)
5. **Entity Recognition Agent**: Extract parties, dates, monetary values
6. **Q&A Agent**: Answer specific questions about the document

### Workflow Improvements

1. **Conditional Branching**: Route to specialized agents based on document type
2. **Iterative Refinement**: Allow agents to critique and improve each other's outputs
3. **Human-in-the-Loop**: Add approval gates between agents
4. **Multi-Document Analysis**: Extend state to handle multiple documents
5. **Streaming Outputs**: Show agent results as they complete

---

## References

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangChain Documentation**: https://python.langchain.com/
- **Ollama Models**: https://ollama.ai/library
- **Streamlit Documentation**: https://docs.streamlit.io/

---

## Glossary

- **Agent**: A specialized AI function that performs one specific task
- **Node**: A LangGraph agent function that processes state
- **State**: Shared data structure passed between agents
- **Workflow**: The complete graph of agents and their execution order
- **Map-Reduce**: A pattern for processing large documents by chunking and consolidating
- **Temperature**: LLM parameter controlling randomness (0=deterministic, 1=creative)
- **Condensation**: Summarizing long documents to fit within LLM context limits

---

**Legal Document Analyzer Agent System** - Orchestrating specialized AI agents for comprehensive legal analysis 🤖⚖️
