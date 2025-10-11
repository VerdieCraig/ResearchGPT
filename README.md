# ResearchGPT Assistant

ResearchGPT Assistant is an intelligent research tool that leverages advanced AI techniques to help researchers process academic documents, generate insights, and automate research workflows.  
This project demonstrates the integration of machine learning fundamentals, natural language processing, advanced prompting strategies, and AI agents in a practical research assistance application.

---

## 🚀 Quick Start

Once installed and configured, you can run the complete demo and evaluation suite:

```bash
# Run the end-to-end demonstration
python main.py

# Run the full test and evaluation suite
python test_system.py


All outputs (summaries, reports, performance metrics) will be saved automatically to the results/ directory.

✨ Features
Core Capabilities

Document Processing – Extract and process text from PDF research papers

Intelligent Search – TF-IDF–based similarity search for relevant document retrieval

Advanced Prompting – Chain-of-Thought, Self-Consistency, and ReAct prompting strategies

AI Agents – Specialized agents for summarization, question-answering, and research workflows

Research Automation – Multi-step session management with integrated verification and reporting

Advanced Prompting Techniques

Chain-of-Thought Reasoning – Step-by-step logical reasoning for complex questions

Self-Consistency – Multiple reasoning paths with consensus-based answers

ReAct Workflows – Structured Thought–Action–Observation research cycles

Verification and Editing – Answer quality checking and improvement mechanisms

AI Agents

Summarizer Agent – Document and literature overview generation

QA Agent – Factual and analytical question answering

Research Workflow Agent – Complete research session orchestration

Agent Orchestrator – Multi-agent task coordination and routing

🧩 Technical Architecture
Technology Stack

Python 3.8+

Mistral API – Large Language Model integration

scikit-learn – TF-IDF and cosine similarity

PyPDF2 – PDF text extraction

NLTK – Sentence tokenization

pandas / numpy – Data manipulation and numerical computing

python-dotenv – Configuration management

Project Structure
research_gpt_assistant/
├── README.md
├── requirements.txt
├── config.py
├── document_processor.py
├── research_assistant.py
├── research_agents.py
├── main.py
├── test_system.py
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

⚙️ Installation
Prerequisites

Python 3.8 or higher

A valid Mistral API key

Git

Setup Instructions

Clone the repository

git clone https://github.com/yourusername/research-gpt-assistant.git
cd research-gpt-assistant


Create and activate a virtual environment

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows


Install dependencies

pip install -r requirements.txt


Configure API settings

Copy .env.example to .env

Add your Mistral API key:

MISTRAL_API_KEY=sk-your-real-key
MODEL_NAME=mistral-medium-latest
MODEL_FALLBACKS=mistral-small-latest, mistral-tiny-latest
TEMPERATURE=0.2
MAX_TOKENS=384


Adjust retry/backoff parameters if desired.

Prepare directories

mkdir -p data/sample_papers data/processed results


Add sample research papers
Place PDF files in data/sample_papers/.
These will be processed and indexed automatically.

🧠 Usage
Basic Usage
python main.py


This:

Processes all PDFs in data/sample_papers/

Builds a searchable index

Demonstrates all prompting strategies

Saves results to the results/ directory

Testing and Evaluation
python test_system.py


This:

Tests all components

Benchmarks response speed and accuracy

Generates reports under results/

🧩 Advanced Examples
Custom Research Query
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

Using AI Agents
from research_agents import AgentOrchestrator

orchestrator = AgentOrchestrator(assistant)

# Summarize
summary = orchestrator.route_task('summarizer', {'doc_id': 'paper_id'})

# Question answering
qa_result = orchestrator.route_task('qa', {
    'question': 'How do transformer models work?',
    'type': 'analytical'
})

# Full research workflow
session = orchestrator.route_task('workflow', {
    'research_topic': 'natural language processing trends'
})

⚙️ Configuration Guide

Edit .env or config.py to adjust:

Parameter	Description	Default
MISTRAL_API_KEY	Your API key	—
MODEL_NAME	Model to use	mistral-medium-latest
MODEL_FALLBACKS	Backup models	mistral-small-latest
TEMPERATURE	Output randomness	0.2
MAX_TOKENS	Max response length	384
CHUNK_SIZE	Text chunk size	1000
OVERLAP	Overlap between chunks	100
API_MAX_RETRIES	Retry attempts	4
API_THROTTLE_SEC	Delay between calls	0.75
📊 Results and Evaluation

After running demos or tests, results appear in results/:

File	Description
cot_response.json	Chain-of-Thought reasoning example
self_consistency_response.txt	Self-Consistency prompting output
react_workflow.json	ReAct workflow demonstration
verification_result.json	Answer verification
demo_report.md	Summary of demo run
evaluation_report.md	Comprehensive system evaluation
test_results.json	Detailed performance metrics

These collectively form the Results Documentation required for the final milestone.

🧪 Performance

System Requirements

Memory: ≥ 4 GB (8 GB recommended)

Disk: 1 GB free space

Network: Stable internet connection for API calls

Typical Metrics

Operation	Time
Document processing	~1–2 s per page
Query response	2–5 s (avg.)
QA throughput	~0.7 s per query (local benchmark)
🛠️ Troubleshooting
Issue	Resolution
Missing API key	Add MISTRAL_API_KEY to .env.
429 “capacity exceeded” errors	Increase API_THROTTLE_SEC to 1.0 or wait a minute before retrying.
PDF not processed	Ensure file is extractable (not scanned images).
Import errors	Reinstall dependencies: pip install -r requirements.txt.
Empty search results	Verify documents are indexed via build_search_index().

Logs are stored in logs/app.log for detailed diagnostics.

🧩 Development & Contribution

Fork the repository

Create a branch

git checkout -b feature-name


Implement and test changes

Submit a pull request with description and context

Code style:

Follow PEP 8

Use docstrings and type hints

Add tests for new functionality

📚 Educational Context

This project was developed as part of the AI/ML Capstone Project for the Code Kentucky Artificial Intelligence Pathway.

It demonstrates:

Natural Language Processing with NLTK

Machine Learning fundamentals (TF-IDF, cosine similarity)

Large Language Model integration via Mistral API

Multi-agent AI architecture and workflow automation

Software engineering best practices for AI systems

🧾 License

This project was originally created for educational purposes by Rama Kattunga
for the Code Kentucky AI/ML Capstone. All code and documentation are open for academic use and learning.

🏁 Version History

v1.0.0 – Initial release with full system integration, testing, and documentation