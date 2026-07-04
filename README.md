<p align="center">
  <h1 align="center">🧠 StackMind</h1>
  <p align="center"><strong>An open-source, multi-step reasoning AI agent for complex problem solving.</strong></p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> · <a href="#how-it-works">How It Works</a> · <a href="#how-is-stackmind-different-from-chatgpt--gemini">Comparison</a> · <a href="#project-structure">Structure</a>
  </p>
</p>

---

## What is StackMind?

StackMind is an **AI problem-solving engine** — not a chatbot.

It takes a problem as input, breaks it into steps, reasons through each step, verifies its own answer, and returns a structured solution with full transparency into *how* and *why* the answer was reached.

Under the hood, it uses LangChain with pluggable LLM backends (Mistral, OpenAI) and connects them to a planning system, a knowledge graph, a memory layer, and domain-specific solvers.

### Key Features

| Feature | Description |
|---|---|
| **Multi-Step Planning** | Decomposes complex problems into sub-tasks with dependency tracking |
| **Chain-of-Thought Reasoning** | Shows every reasoning step, not just the final answer |
| **Self-Verification** | Automatically checks and validates its own solutions |
| **Persistent Memory** | Remembers context across sessions using ChromaDB |
| **Knowledge Graph** | Builds causal entity maps to understand relationships between concepts |
| **Domain-Specific Solvers** | Specialized engines for math, code, science, and general problems |
| **Multi-Modal Input** | Accepts text, images, voice, and code files |
| **Pluggable LLMs** | Swap between Mistral, OpenAI, or any LangChain-compatible model |
| **Dual Interface** | CLI for local use, REST API for integration |

---

## How is StackMind Different from ChatGPT / Gemini?

This is the most common question — and the answer is architectural, not cosmetic.

| | ChatGPT / Gemini | StackMind |
|---|---|---|
| **Architecture** | Monolithic model behind an API | Modular agent with planning, reasoning, memory, and verification layers |
| **Transparency** | Black box — you see the answer, not the process | Full chain-of-thought — every reasoning step is exposed |
| **Verification** | No self-check — may hallucinate confidently | Built-in self-verification with confidence scoring |
| **Memory** | Session-only (or limited) context | Persistent long-term memory across sessions (ChromaDB) |
| **Knowledge Graph** | None | Builds causal entity maps to track concept relationships |
| **Planning** | Single-pass generation | Multi-step planning with dependency graphs and alternative paths |
| **Domain Routing** | One model handles everything | Specialized solvers for math, code, science with tailored prompts |
| **Ownership** | Cloud-hosted, proprietary | Fully local, open-source, your data stays on your machine |
| **Customization** | Limited to system prompts | Full control — swap models, add solvers, extend the pipeline |

**In short:** ChatGPT is a *conversational interface* to a large language model. StackMind is a *problem-solving engine* that uses LLMs as one component in a larger reasoning pipeline.

---

## Quickstart

### Prerequisites
- Python 3.10+
- A Mistral API key (or OpenAI API key)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/StackMind.git
cd StackMind

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### Usage

**Interactive CLI:**
```bash
python main.py --mode cli
```

**REST API Server:**
```bash
python main.py --mode api
```

**One-Shot Problem Solving:**
```bash
python main.py --mode cli --problem "Find all prime numbers less than 50"
```

### One-Click Launch on Windows

Double-click [Start-StackMind.bat](Start-StackMind.bat) to start the backend API, start the frontend dev server, and open the UI in your browser.

---

## How It Works

```
Input (text / image / voice / code)
        │
        ▼
┌─────────────────┐
│  Input Processor │  ← Parses and classifies the input
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Domain Detection │  ← Routes to the right solver (math, code, science...)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Planning System  │  ← Breaks the problem into sub-tasks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Domain Solver    │  ← Solves each sub-task with specialized logic
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reasoning Engine │  ← Generates chain-of-thought for transparency
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Verification   │  ← Self-checks the solution for correctness
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Memory + KG      │  ← Stores results and updates the knowledge graph
└────────┬────────┘
         │
         ▼
     Solution
 (with confidence score, reasoning steps, and explanation)
```

---

## Project Structure

```
StackMind/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
│
├── config/ps://contribution.usercontent.google.com/download?
│   └── settings.py          # Centralized configuration management
│
├── core/                    # The brain
│   ├── agent.py             # Main orchestration agent
│   ├── reasoning.py         # Chain-of-thought reasoning engine
│   ├── planning.py          # Multi-step planning system
│   ├── memory.py            # Persistent memory (ChromaDB + session)
│   ├── knowledge_graph.py   # Causal entity graph (NetworkX)
│   ├── symbolic.py          # Symbolic computation (SymPy)
│   └── code_execution.py    # Sandboxed code runner
│
├── domains/                 #  Specialized solvers
│   ├── general_solver.py    # General-purpose problem solver
│   ├── math_solver.py       # Mathematics & symbolic math
│   ├── code_solver.py       # Code debugging, generation, analysis
│   └── science_solver.py    # Scientific reasoning
│
├── processors/              # Input processing
│   ├── text_processor.py    # Text analysis & entity extraction
│   ├── image_processor.py   # Image analysis & OCR
│   ├── voice_processor.py   # Speech-to-text transcription
│   └── code_processor.py    # Code parsing & language detection
│
├── integrations/            # External services
│   ├── mistral_integration.py  # Mistral AI LLM provider
│   ├── openai_integration.py   # OpenAI LLM provider
│   └── data_integration.py     # Web search, financial data, news
│
├── interfaces/              # User interfaces
│   ├── cli.py               # Interactive command-line interface
│   └── api.py               # FastAPI REST server
│
├── utils/                   # Utilities
│   ├── helpers.py           # Common helper functions
│   ├── validators.py        # Input validation
│   └── visualization.py     # Graph & data visualization
│
├── tests/                   # Test suite
│   ├── test_core.py
│   ├── test_interfaces.py
│   ├── test_processors.py
│   └── test_solvers.py
│
└── data/                    # Data 
    └── README.md
```

---

## Configuration

StackMind is configured via:
1. **`.env`** — API keys and environment-specific settings
2. **`config/settings.py`** — Application defaults (LLM provider, memory, input processing)

To switch LLM providers, edit the `llm` section in `settings.py`:
```python
"llm": {
    "provider": "mistral",       # or "openai"
    "model": "mistral-large-latest",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

---

## API Endpoints

When running in API mode (`python main.py --mode api`), the following endpoints are available:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/solve/text` | Solve a text-based problem |
| `POST` | `/solve/image` | Solve from an image (base64) |
| `POST` | `/solve/voice` | Solve from audio (base64) |
| `POST` | `/solve/code` | Analyze/debug/generate code |
| `POST` | `/feedback` | Submit feedback on a solution |
| `GET` | `/sessions/{id}` | Get session information |
| `GET` | `/health` | Health check |

Interactive API docs at: `http://localhost:8000/docs`

---

## Roadmap

- [ ] **Streaming Responses** — Real-time reasoning output
- [ ] **Google Gemini Integration** — As an additional LLM provider
- [ ] **Web UI** — Streamlit/Gradio dashboard for browser-based use
- [ ] **Plugin System** — Drop-in custom solvers
- [ ] **RAG** — Upload documents and query them
- [ ] **Multi-Agent Collaboration** — LangGraph-powered agent teams

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
