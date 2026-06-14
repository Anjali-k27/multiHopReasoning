# Newton SkillUP Autonomous Agent (v4.0 - DSPy ReAct)

A local Streamlit dashboard powered by a **DSPy ReAct agent** that autonomously
reasons across three simulated data sources — a SQL financial database, a vector
document store, and a knowledge graph — to produce boardroom-ready executive briefs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [How DSPy ReAct Works](#3-how-dspy-react-works)
4. [The Three Tools Explained](#4-the-three-tools-explained)
5. [Project Structure](#5-project-structure)
6. [Setup & Installation](#6-setup--installation)
7. [Running the Dashboard](#7-running-the-dashboard)
8. [Test Query & Expected Answer](#8-test-query--expected-answer)
9. [Running the Automated Test](#9-running-the-automated-test)
10. [How the Multi-Hop Reasoning Works Step by Step](#10-how-the-multi-hop-reasoning-works-step-by-step)

---

## 1. Project Overview

Traditional AI pipelines answer questions by making a single call to a language
model. This project demonstrates a fundamentally different paradigm: a **ReAct
(Reasoning + Acting) agent** that breaks a complex question into sub-tasks,
selects the right tool for each sub-task, observes the result, and iterates —
all autonomously, without any hard-coded decision logic.

The agent is built on top of **DSPy**, Anthropic/Stanford's declarative framework
for programming language models. The LLM backbone is **Google Gemini 2.5 Flash**.

**What makes this "multi-hop"?**  
To answer the test query, the agent must cross three knowledge boundaries:

- Hop 1 — Financial database → Microsoft's cash position  
- Hop 2 — Knowledge graph → Alpha AI's competitor (NeuralNet Labs)  
- Hop 3 — Document store → NeuralNet Labs' strategic weakness  

No single tool can answer the question alone. The agent must chain them.

---

## 2. Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Text Area → "Deploy Agent" Button → Status Spinner  │   │
│  │  Executive Brief (st.info)                           │   │
│  │  Audit Log Expander (trajectory steps)               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │ master_agent(command=...)
┌────────────────────────────▼────────────────────────────────┐
│               AlphaFundAgent (agent_engine.py)              │
│                    dspy.Module subclass                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              dspy.ReAct engine                       │   │
│  │  signature: strategic_command →                      │   │
│  │             comprehensive_executive_brief            │   │
│  │  max_iters: 5                                        │   │
│  └───────────┬──────────────────────────────────────────┘   │
└──────────────┼──────────────────────────────────────────────┘
               │ iterative tool calls
    ┌──────────┼────────────────────────────────┐
    ▼          ▼                                ▼
┌────────┐ ┌────────┐                  ┌────────────┐
│  SQL   │ │ Vector │                  │  Knowledge │
│ Tool   │ │  Doc   │                  │   Graph    │
│        │ │ Search │                  │   Tool     │
└────────┘ └────────┘                  └────────────┘
```

### Key components

| Component | File | Role |
|-----------|------|------|
| `AlphaFundAgent` | `agent_engine.py` | DSPy Module wrapper around the ReAct engine |
| `dspy.ReAct` | (DSPy library) | Orchestrates the Thought → Tool → Observation loop |
| `dspy.Predict` | (DSPy library) | Powers each single reasoning step inside ReAct |
| `dspy.ChainOfThought` | (DSPy library) | Synthesises the final brief from the completed trajectory |
| `master_agent` | `agent_engine.py` | Module-level singleton shared by both the UI and the test |

---

## 3. How DSPy ReAct Works

DSPy's `ReAct` module implements the [ReAct paper](https://arxiv.org/abs/2210.03629)
as a declarative, signature-driven loop. Here is exactly what happens on each
iteration:

```
┌─────────────────────────────────────────────────────────────┐
│  Iteration N                                                │
│                                                             │
│  INPUT:  strategic_command + trajectory so far             │
│                                                             │
│  dspy.Predict produces three outputs:                       │
│    next_thought   → free-form reasoning text               │
│    next_tool_name → one of the registered tool names       │
│    next_tool_args → JSON dict of arguments for the tool    │
│                                                             │
│  DSPy executes: tools[next_tool_name](**next_tool_args)     │
│                                                             │
│  OUTPUT: observation appended to trajectory                │
│                                                             │
│  Loop continues until next_tool_name == "finish"           │
└─────────────────────────────────────────────────────────────┘
```

After the loop, `dspy.ChainOfThought` reads the full trajectory and synthesises
the `comprehensive_executive_brief`.

The model never hard-codes which tool to call or in which order — it reasons
its way there on every run.

---

## 4. The Three Tools Explained

### Tool 1 — `execute_sql_financials(entity_name: str)`
**Docstring trigger:** "exact mathematical figures, market caps, revenue, or cash"  
Simulates a SQL query against a financial database. Returns hard numbers such as
Microsoft's $50 Billion liquid cash reserve. The docstring deliberately avoids
qualitative language so the agent does not use this tool for strategic analysis.

### Tool 2 — `search_unstructured_documents(semantic_query: str)`
**Docstring trigger:** "qualitative information, CEO quotes, or strategic weaknesses"  
Simulates a semantic vector search across internal research documents. Returns
prose excerpts such as analyst notes on a company's operational weaknesses.

### Tool 3 — `traverse_knowledge_graph(entity_name: str)`
**Docstring trigger:** "multi-hop relationships or 'who competes with who'"  
Simulates a graph traversal (think Neo4j / Cypher). Returns edges in a
property-graph notation, e.g. `(Alpha AI) --[COMPETES_WITH]--> (NeuralNet Labs)`.

**Why separate tools?**  
Real hedge-fund or enterprise AI systems store data across multiple specialised
stores. Using three tools with targeted docstrings teaches the agent *when* to
reach for each source — a critical skill for production RAG architectures.

---

## 5. Project Structure

```
pythonProject/
├── .env                  # GEMINI_API_KEY (not committed)
├── .gitignore
├── requirements.txt      # streamlit, dspy-ai, python-dotenv
├── agent_engine.py       # Tools, AlphaFundAgent, master_agent singleton
├── app.py                # Streamlit dashboard
└── test_agent.py         # Automated pass/fail verification script
```

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### Step 1 — Clone / open the project
```bash
cd multiHopReasoning/pythonProject
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

The `agent_engine.py` loads this automatically via `python-dotenv` and maps it
to `GOOGLE_API_KEY` as required by the LiteLLM/Gemini backend that DSPy uses
internally.

---

## 7. Running the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. You will see:

```
Newton SkillUP Autonomous Agent (v4.0 - DSPy ReAct)
────────────────────────────────────────────────────
[Strategic Command text area]
[Deploy Autonomous Agent button]
```

---

## 8. Test Query & Expected Answer

Copy the query below into the **Strategic Command** text area and click
**Deploy Autonomous Agent**.

### Query

```
Should Microsoft use its cash to acquire Alpha AI or NeuralNet Labs?
Use financial data for Microsoft, find out who Alpha AI competes with,
and check the documents for any weaknesses regarding the competitor
to make your decision.
```

### What the agent should do (Cognitive Loop)

| Step | Thought | Tool Called | Observation |
|------|---------|-------------|-------------|
| 1 | "I need Microsoft's financial capacity before evaluating an acquisition." | `execute_sql_financials("Microsoft")` | `SQL RESULT: Microsoft Liquid cash available: $50 Billion.` |
| 2 | "Now I need to understand Alpha AI's competitive landscape." | `traverse_knowledge_graph("Alpha AI")` | `GRAPH TOPOLOGY: (Alpha AI) --[COMPETES_WITH]--> (NeuralNet Labs).` |
| 3 | "The competitor is NeuralNet Labs. Let me find its weaknesses." | `search_unstructured_documents("weaknesses of NeuralNet Labs")` | `DOCUMENT CHUNK: 'NeuralNet Labs suffers from massive cloud compute overhead.'` |
| 4 | "I have all the data I need." | `finish` | `Completed.` |

### Expected Executive Brief

The final `comprehensive_executive_brief` should contain all three of the
following synthesised facts:

> **Recommendation: Acquire Alpha AI**
>
> Microsoft holds **$50 Billion** in liquid cash, providing ample capacity for a
> major acquisition. Intelligence from the knowledge graph confirms that
> **Alpha AI directly competes with NeuralNet Labs**. Document analysis reveals
> that **NeuralNet Labs suffers from massive cloud compute overhead** — a
> structural weakness that Microsoft, with its Azure cloud infrastructure, is
> uniquely positioned to exploit by acquiring and empowering Alpha AI.
> Acquiring Alpha AI is therefore the strategically superior move: it strengthens
> a competitor against a financially burdened rival without inheriting that
> burden.

### What to look for in the UI

1. **Status spinner** — shows "Agent Deployed. Initiating Cognitive Loop..." while
   the agent is reasoning.
2. **Executive Brief** — displayed in a blue `st.info` block once the loop
   completes.
3. **Audit Log expander** (🔍) — expand it to see every Thought, Tool Call,
   and Observation step-by-step. This is the full ReAct trajectory and proves
   the agent did not hallucinate — every fact in the brief is traceable to a
   specific tool observation.

---

## 9. Running the Automated Test

The `test_agent.py` script bypasses the UI and runs the agent directly, then
checks two hard pass/fail criteria:

```bash
python test_agent.py
```

**Criterion 1** — All three tools must be invoked during the reasoning loop:
- `execute_sql_financials`
- `search_unstructured_documents`
- `traverse_knowledge_graph`

**Criterion 2** — The final brief must contain all three key facts:
- `$50 Billion`
- `NeuralNet Labs`
- `cloud compute overhead`

A passing run prints:

```
OVERALL RESULT: ✅  ALL TESTS PASSED
```

If a criterion fails, refine the tool docstrings or the ReAct signature in
`agent_engine.py` and re-run until all checks pass.

---

## 10. How the Multi-Hop Reasoning Works Step by Step

This section walks through the internals so you can reason about what DSPy is
doing under the hood.

### Step 1 — Signature compilation
When `AlphaFundAgent.__init__` runs, `dspy.ReAct` reads the string signature
`"strategic_command -> comprehensive_executive_brief"` and builds a typed
`dspy.Signature` object. It appends a `trajectory` input field and three
output fields (`next_thought`, `next_tool_name`, `next_tool_args`) to create
the internal `react_signature` used by `dspy.Predict`.

### Step 2 — Tool registration
Each Python function is wrapped in a `dspy.Tool` object. DSPy extracts the
function name, type hints, and docstring to build the tool description shown
to the LLM. A special `finish` tool is added automatically.

### Step 3 — Prompt construction
On the first call, the trajectory is empty. DSPy constructs a prompt that
lists all tools with their descriptions and asks the model to produce
`next_thought`, `next_tool_name`, and `next_tool_args`.

### Step 4 — Iteration
After each model call:
1. The thought, tool name, and args are appended to the trajectory dict.
2. The chosen tool function is called with the provided args.
3. The return value is stored as `observation_N` in the trajectory.
4. The updated trajectory is passed to the model on the next iteration.

### Step 5 — Extraction
Once the model calls `finish`, `dspy.ChainOfThought` reads the entire
trajectory alongside the original `strategic_command` and synthesises the
`comprehensive_executive_brief` with an accompanying `rationale`.

### Step 6 — Return
`dspy.Prediction(trajectory=trajectory, **extract)` is returned. The UI reads
`comprehensive_executive_brief` for the brief and the `trajectory` dict for
the audit log.

---

*Built with DSPy 3.x · Gemini 2.5 Flash · Streamlit*
