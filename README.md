# 🧠 ResearchGPT Assistant

**ResearchGPT Assistant** is an intelligent research tool that leverages advanced AI techniques to help researchers process academic documents, generate insights, and automate research workflows.  
It integrates document retrieval, natural language reasoning, and AI agents into a modular, testable system.

---

## 🚀 Quick Start

Once installed and configured, you can run the complete demo and evaluation suite:

```bash
# Run the end-to-end demonstration
python main.py

# Run the full test and evaluation suite
python test_system.py
```

All outputs (summaries, reports, performance metrics) will be saved automatically to the `results/` directory.

---

## ✨ Features

### Core Capabilities
- **Document Processing** – Extract and process text from PDF research papers  
- **Intelligent Search** – TF-IDF–based similarity search for relevant document retrieval  
- **Advanced Prompting** – Chain-of-Thought, Self-Consistency, and ReAct prompting strategies  
- **AI Agents** – Specialized agents for summarization, question-answering, and workflow orchestration  
- **Verification & Editing** – Automatic answer checking and improvement  

### Advanced Prompting Techniques
- **Chain-of-Thought Reasoning** – Structured multi-step reasoning for complex queries  
- **Self-Consistency** – Multiple reasoning paths with consensus-based answer selection  
- **ReAct Workflows** – Iterative Thought–Action–Observation research cycles with local fallbacks  

### Multi-Agent Architecture
- **Summarizer Agent** – Generates document overviews  
- **QA Agent** – Answers factual or analytical questions  
- **Research Workflow Agent** – Runs complete research sessions  
- **Agent Orchestrator** – Routes and coordinates AI tasks  

---

## 🧩 Technical Architecture

### Technology Stack
- Python 3.8+
- Mistral API (LLM integration)
- scikit-learn – TF-IDF & cosine similarity
- PyPDF2 – PDF text extraction
- NLTK – Sentence tokenization
- pandas / numpy – Data processing
- python-dotenv – Configuration management and .env loading

### Project Structure
```
research_gpt_assistant/
├── README.md
├── requirements.txt
├── config.py
├── document_processor.py
├── research_assistant.py
├── research_agents.py
├── main.py
├── test_system.py
├── .env.example
├── data/
│   ├── sample_papers/
│   └── processed/
├── results/
│   ├── summaries/
│   ├── analyses/
│   ├── demo_report.md
│   ├── evaluation_report.md
│   └── test_results.json
└── prompts/
    └── prompt_templates.txt
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher  
- A valid [Mistral API key](https://console.mistral.ai/api-keys)  
- Git  

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-gpt-assistant.git
   cd research-gpt-assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and replace the placeholder with your **Mistral API key**:

   ```bash
   MISTRAL_API_KEY=sk-your-real-key
   MODEL_NAME=mistral-medium-latest
   MODEL_FALLBACKS=mistral-small-latest, mistral-tiny-latest
   TEMPERATURE=0.2
   MAX_TOKENS=384
   API_MAX_RETRIES=4
   API_THROTTLE_SEC=0.75
   API_CAPACITY_COOLDOWN_SEC=2.5
   ```

   > 📝 **Note:**  
   > The project no longer stores API keys in `config.py`.  
   > You **must** set them in `.env` before running any scripts.

5. **Prepare directories**
   ```bash
   mkdir -p data/sample_papers data/processed results
   ```

6. **Add sample research papers**
   Place your PDF files into `data/sample_papers/`. They will be processed and indexed automatically.

---

## 🧠 Usage

### Basic Demo
```bash
python main.py
```
This:
- Processes all PDFs in `data/sample_papers/`
- Builds a searchable index
- Demonstrates all prompting strategies
- Saves results to the `results/` directory

### Testing & Evaluation
```bash
python test_system.py
```
This:
- Tests all components  
- Benchmarks performance and accuracy  
- Generates reports under `results/`

---

## 🧩 Advanced Examples

### Programmatic Usage
```python
from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

config = Config()
docs = DocumentProcessor(config)
assistant = ResearchGPTAssistant(config, docs)

docs.process_document("data/sample_papers/ai_paper.pdf")
docs.build_search_index()

response = assistant.answer_research_question(
    "What are the main limitations of current machine learning approaches?",
    use_cot=True,
    use_verification=True
)

print(response['answer'])
```

### Using AI Agents
```python
from research_agents import AgentOrchestrator

orchestrator = AgentOrchestrator(assistant)
qa_result = orchestrator.route_task('qa', {'question': 'How do transformer models work?'})
print(qa_result)
```

---

## ⚙️ Configuration Reference

All configuration values are loaded from `.env` at runtime.  
You can override them by editing that file or setting environment variables.

| Parameter | Description | Default |
|------------|-------------|----------|
| `MISTRAL_API_KEY` | Your API key | — |
| `MODEL_NAME` | Primary model | `mistral-medium-latest` |
| `MODEL_FALLBACKS` | Backup models (CSV) | `mistral-small-latest, mistral-tiny-latest` |
| `TEMPERATURE` | Output randomness | `0.2` |
| `MAX_TOKENS` | Max response length | `384` |
| `API_MAX_RETRIES` | Retry attempts | `4` |
| `API_THROTTLE_SEC` | Delay between calls | `0.75` |
| `API_CAPACITY_COOLDOWN_SEC` | Cooldown on 429s | `2.5` |

---

## 📊 Results and Evaluation

After running demos or tests, results appear in `results/`:

| File | Description |
|------|-------------|
| `cot_response.json` | Chain-of-Thought reasoning example |
| `self_consistency_response.txt` | Self-Consistency output |
| `react_workflow.json` | ReAct workflow |
| `verification_result.json` | Answer verification |
| `demo_report.md` | Summary of demo run |
| `evaluation_report.md` | Comprehensive evaluation |
| `test_results.json` | Detailed performance metrics |

---

## 🧪 Performance

| Operation | Avg. Time |
|------------|------------|
| Document processing | ~1–2 s per page |
| Query response | 2–5 s |
| QA throughput | ~0.7 s per query |

**System Requirements:**
- ≥ 4 GB RAM (8 GB recommended)
- 1 GB disk space
- Internet access for API calls

---

## 🛠️ Troubleshooting

| Issue | Resolution |
|--------|-------------|
| **Missing API key** | Add `MISTRAL_API_KEY` to `.env`. |
| **429 “capacity exceeded” errors** | Increase `API_THROTTLE_SEC` to 1.0 or wait before retrying. |
| **PDF not processed** | Ensure file is text-based (not scanned images). |
| **Import errors** | Run `pip install -r requirements.txt`. |
| **Empty search results** | Verify index creation with `build_search_index()`. |

Logs are written to `logs/app.log` for detailed diagnostics.

---

## 🧩 Development & Contribution

1. Fork the repository  
2. Create a branch:
   ```bash
   git checkout -b feature-name
   ```
3. Implement and test your changes  
4. Submit a pull request with context

**Code style:**  
- Follow PEP 8  
- Include docstrings and type hints  
- Add tests for new features

---

## 📚 Educational Context

This project was developed as part of the **Code Kentucky AI/ML Pathway Capstone Project**.  
It demonstrates:
- Natural Language Processing with NLTK  
- TF-IDF & cosine similarity search  
- Large Language Model integration (Mistral API)  
- Multi-agent AI architecture & workflow automation  
- Applied software engineering practices for AI systems  

---

## 🧾 License

Originally developed by **Rama Kattunga** for educational use in the Code Kentucky AI/ML Capstone.  
Extended and refined for continued research and educational purposes.

---

## 🏁 Version History

| Version | Description |
|----------|--------------|
| v1.0.0 | Full system integration, testing, and documentation |
| v1.1.0 | Environment variable migration, fallback models, cooldown handling, improved logging |
