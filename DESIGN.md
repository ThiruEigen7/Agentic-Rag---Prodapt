# Design Document: Agentic Financial RAG

This document outlines the architecture, tool specifications, and safety mechanisms of the Agentic Financial RAG system designed for analyzing Indian IT company financials.

---

## 1. Agent Reasoning Loop (Step-by-Step)

The agent operates as a multi-step autonomous reasoner, moving from raw input to a cited answer through the following stages:

### Phase A: Pre-Processing (Deterministic & Filtered)
1.  **Query Normalization**: The raw user query is cleaned. Common entity aliases (e.g., "TCS" or "CTS") are mapped to canonical names like "Cognizant" to ensure tool consistency.
2.  **Gatekeeping**: An LLM-based classifier evaluates the intent. 
    *   **TRIVIAL**: Answers directly (e.g., "Hello").
    *   **REFUSE**: Declines out-of-scope or sensitive requests (e.g., "Who won the IPL?" or "Give me stock tips").
    *   **PROCEED**: Pass-through to the main agent loop.

### Phase B: The Main Execution Loop (Max 8 Iterations)
The agent maintains a **Stateful Context**, a growing string that records every tool call, its specific input (e.g., generated SQL), and its raw output. This loop handles the "missing data" problem through multi-step reasoning:

3.  **Instructional Planning**: Before the first tool call, the agent generates a high-level strategy (the "Plan"). If the query is complex (e.g., a comparison), the planner explicitly schedules multiple tools (e.g., "Get numbers from SQL, then strategy from Vector").
4.  **Action Selection (The Brain)**: The LLM analyzes the *entire* context of previous steps. It is instructed to never repeat a query that returned "No results." If a tool fails, the selection logic forces a "pivot" to an alternative tool (e.g., switching from SQL to semantic search if a column is missing).
5.  **Tool Execution**: The selected tool runs autonomously. Structured tools (`query_data`) perform natural-language-to-SQL conversion, while unstructured tools (`search_docs`) perform FAISS vector retrieval.
6.  **Sufficiency Evaluation (The Gate)**: After *every* step, the agent performs a dual check:
    *   **Heuristic Guard**: Hardcoded rules ensure comparisons always proceed to qualitative enrichment, even if numeric data is found in Step 1.
    *   **Reasoning Guard**: The LLM evaluates if the current gathered facts are sufficient to fulfill the "Plan." If not, it returns to Step 4 to select the next action.

### Phase C: Answer Synthesis
7.  **Final Composition**: Once data is sufficient, the LLM synthesizes a coherent narrative, ensuring all numerical facts are cited from the retrieved context.

---

## 2. Tool Schemas

The agent has access to three specialized tools:

| Tool Name | Description | Input (JSON/Str) | Output (JSON) |
| :--- | :--- | :--- | :--- |
| **`query_data`** | **Structured Logic.** Natural Language to SQL converter for numeric data. | `{"question": "string"}` | `{"result": [...], "sql": "query", "source": "db"}` |
| **`search_docs`** | **Semantic Logic.** FAISS-based vector search over annual report text. | `{"query": "string", "company": "optional"}` | `{"chunks": [{"text": "...", "score": 0.8}, ...]}` |
| **`web_search`** | **Real-time Logic.** Bridge to Tavily API for news and live stock prices. | `{"query": "string"}` | `{"results": [{"title": "...", "url": "...", "content": "..."}]}` |

---

## 3. Prevention of Infinite Loops & Stalls

To ensure the agent remains performant and doesn't enter "hallucination loops," three layers of protection are implemented:

### 1. The Hard Cap (Token of Safety)
The core loop is wrapped in a `for` loop with a constant `MAX_STEPS = 8`. If the agent cannot reach a conclusion within 8 tool calls, it is forcibly terminated with a `HardCapExceeded` exception. It then provides a partial answer based on what it found, ensuring the user gets a result even if incomplete.

### 2. Failure-Driven Routing (Avoiding Retries)
If a tool returns an error or empty results (e.g., a SQL query finds no rows), this status is explicitly fed back into the context. The `Select Tool` prompt instructs the LLM that **repeating a failed query is prohibited**. Instead, the agent must pivot to an alternative source (e.g., if `query_data` fails, try `search_docs`).

### 3. Contextual Progression Tracking
Every tool call is recorded with its original input and raw output. The LLM is forced to review this "execution trace" at every step. This prevents "state-less stalls" where an agent might ask for the same data twice because it "forgot" it already tried.

### 4. Deterministic Model Fallbacks
To prevent stalling due to API rate limits, the system detects `429 Too Many Requests` errors and automatically switches to a secondary, faster model (e.g., switching from Llama-3.3-70b to Llama-3.1-8b). This ensures the reasoning process continues even during high-load periods.

---

## 4. Traceability & Response Format

The system is designed for maximum transparency. Every run must produce a structured execution trace in the terminal and save a deep JSON log.

### Terminal Trace Format
The terminal output follows a strict audit sequence:
1.  **Step-by-Step Execution**: Prints `Step N: [tool] input='...'` and `result=...` for every action.
2.  **TOOL CALLED**: A header summarizing all tools used.
3.  **Numbered Synthesis Block**:
    -   `1)QUESTION`: The input.
    -   `2)PLAN`: The reasoning intent.
    -   `3)FINAL ANSWER`: The response with inline `[tool -> source]` citations.
    -   `4)CITATIONS`: Numbered list of unique sources.
    -   `5)OVERALL SCORE`: Confidence metric.
    -   `6)STEPS`: Efficiency metric (N/8).

### Citation Standards
The agent is prohibited from using generic citations like "search_docs". It must name the specific resource:
-   **Structured**: `financials.db (table/row description)`.
-   **Unstructured**: `[filename].md (section/page)`.
-   **Web**: `Original Source URL`.
