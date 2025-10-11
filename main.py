"""
Main execution script for ResearchGPT Assistant

Functionality:
1. Load configuration and initialize system components
2. Process sample documents (PDFs)
3. Build TF-IDF search index
4. Demonstrate prompting strategies and agent coordination
5. Save comprehensive demo results
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

# Agents are optional but recommended for full demo
try:
    from research_agents import AgentOrchestrator
except Exception:
    AgentOrchestrator = None


def main():
    """
    Execute end-to-end demonstration:
    1) Initialize components
    2) Ingest sample PDFs
    3) Build search index
    4) Demonstrate capabilities via demonstrate_all_capabilities()
    5) Save results
    """
    print("=== ResearchGPT Assistant Demo ===")

    args = _parse_args()

    # 1) Initialize
    print("\n1. Initializing system...")
    config = Config()
    logger = getattr(config, "logger", None)
    if logger:
        logger.info("Starting ResearchGPT demo")

    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    orchestrator = AgentOrchestrator(research_assistant) if AgentOrchestrator is not None else None

    # 2) Ingest PDFs
    print("\n2. Processing sample documents...")
    sample_papers_dir = args.sample_dir or config.SAMPLE_PAPERS_DIR
    pdf_files = _collect_pdfs(sample_papers_dir)
    if not pdf_files:
        print(f"   No PDF files found in: {sample_papers_dir}")
        print("   Add 2–3 PDFs to proceed.")
        return

    to_process = pdf_files[: args.limit] if args.limit else pdf_files
    for pdf_file in to_process:
        pdf_path = os.path.join(sample_papers_dir, pdf_file)
        print(f"   Processing: {pdf_file}")
        try:
            doc_id = doc_processor.process_document(pdf_path)
            print(f"   Processed as doc_id: {doc_id}")
        except Exception as e:
            print(f"   Error processing {pdf_file}: {e}")

    # 3) Index
    print("\n3. Building search index...")
    doc_processor.build_search_index()
    stats = doc_processor.get_document_stats()
    print(f"   Documents processed: {stats}")

    # 4) Full capabilities demonstration
    print("\n4. Running full capabilities demonstration...")
    demo_summary = demonstrate_all_capabilities(
        config=config,
        doc_processor=doc_processor,
        assistant=research_assistant,
        orchestrator=orchestrator,
        pdf_files=to_process,
        args=args,
    )

    # 5) Final report
    print("\n5. Generating final demonstration report...")
    try:
        final_report = _generate_demo_report(config, doc_processor)
        _save_result("demo_report.md", final_report, config, is_text=True)
    except Exception as e:
        print(f"   Report generation failed: {e}")

    print("\n=== Demo Complete ===")
    print(f"Results saved in: {config.RESULTS_DIR}")
    for f in demo_summary.get("saved_files", []):
        print(f"- {f}")


def demonstrate_all_capabilities(
    *,
    config: Config,
    doc_processor: DocumentProcessor,
    assistant: ResearchGPTAssistant,
    orchestrator: Any,
    pdf_files: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Demonstrate:
    - Document processing (already performed)
    - Chain-of-Thought reasoning
    - Self-Consistency prompting
    - ReAct workflow
    - Agent coordination (Summarizer, QA, Research Workflow)
    - Verification and result saving
    """
    saved_files: List[str] = []

    # Similarity search (sanity)
    test_query = args.query or "machine learning algorithms"
    print(f"   Similarity search query: '{test_query}'")
    try:
        similar_chunks = doc_processor.find_similar_chunks(test_query, top_k=3)
        print(f"   Relevant chunks found: {len(similar_chunks)}")
    except Exception as e:
        print(f"   Similarity search failed: {e}")

    # Chain-of-Thought
    print("\n   Chain-of-Thought reasoning...")
    cot_query = args.cot_query or "What are the main advantages and limitations of deep learning?"
    try:
        cot_response = assistant.answer_research_question(
            cot_query, use_cot=True, use_verification=False
        )
        _save_result("cot_response.json", cot_response, config)
        print("   Saved: cot_response.json")
        saved_files.append("cot_response.json")
    except Exception as e:
        print(f"   CoT demonstration failed: {e}")

    # Self-Consistency
    print("\n   Self-Consistency prompting...")
    sc_query = args.sc_query or "How do neural networks learn?"
    try:
        relevant_chunks = doc_processor.find_similar_chunks(sc_query, top_k=5)
        sc_response = assistant.self_consistency_generate(sc_query, relevant_chunks, num_attempts=3)
        _save_result("self_consistency_response.txt", sc_response, config, is_text=True)
        print("   Saved: self_consistency_response.txt")
        saved_files.append("self_consistency_response.txt")
    except Exception as e:
        print(f"   Self-consistency demonstration failed: {e}")

    # ReAct workflow
    print("\n   ReAct research workflow...")
    react_query = args.react_query or "What are the current trends in natural language processing?"
    try:
        react_response = assistant.react_research_workflow(react_query)
        _save_result("react_workflow.json", react_response, config)
        print("   Saved: react_workflow.json")
        saved_files.append("react_workflow.json")
    except Exception as e:
        print(f"   ReAct demonstration failed: {e}")

    # Agent coordination (if available)
    if orchestrator is not None:
        print("\n   Agent coordination...")

        # Summarizer Agent (first doc)
        try:
            first_doc_id = os.path.splitext(pdf_files[0])[0]
            summary_result = orchestrator.route_task("summarizer", {"doc_id": first_doc_id})
            _save_result("document_summary.json", summary_result, config)
            print("   Saved: document_summary.json")
            saved_files.append("document_summary.json")
        except Exception as e:
            print(f"   Summarizer agent failed: {e}")

        # QA Agent
        try:
            qa_task = {"question": "What methodology was used in the research?", "type": "analytical"}
            qa_result = orchestrator.route_task("qa", qa_task)
            _save_result("qa_response.json", qa_result, config)
            print("   Saved: qa_response.json")
            saved_files.append("qa_response.json")
        except Exception as e:
            print(f"   QA agent failed: {e}")

        # Research Workflow Agent
        try:
            workflow_task = {"research_topic": "artificial intelligence applications"}
            workflow_result = orchestrator.route_task("workflow", workflow_task)
            _save_result("research_workflow.json", workflow_result, config)
            print("   Saved: research_workflow.json")
            saved_files.append("research_workflow.json")
        except Exception as e:
            print(f"   Workflow agent failed: {e}")
    else:
        print("   Agent orchestrator unavailable; skipping agent demos.")

    # Verification demonstration
    print("\n   Answer verification...")
    try:
        test_answer = "Neural networks are computational models inspired by biological neural networks."
        test_query_for_verification = "What are neural networks?"
        verification_result = assistant.verify_and_edit_answer(
            test_answer, test_query_for_verification, "Sample context"
        )
        _save_result("verification_result.json", verification_result, config)
        print("   Saved: verification_result.json")
        saved_files.append("verification_result.json")
    except Exception as e:
        print(f"   Verification demonstration failed: {e}")

    return {"saved_files": saved_files}


# ---------- Helpers ----------

def _parse_args():
    parser = argparse.ArgumentParser(description="ResearchGPT Assistant Demo")
    parser.add_argument("--sample-dir", type=str, help="Directory of sample PDFs (defaults to config.SAMPLE_PAPERS_DIR)")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to ingest (e.g., 3)")
    parser.add_argument("--query", type=str, help="Similarity search test query")
    parser.add_argument("--cot-query", type=str, help="Chain-of-Thought demo query")
    parser.add_argument("--sc-query", type=str, help="Self-consistency demo query")
    parser.add_argument("--react-query", type=str, help="ReAct demo query")
    return parser.parse_args()


def _collect_pdfs(dir_path: str) -> List[str]:
    if not os.path.exists(dir_path):
        print(f"   Sample papers directory not found: {dir_path}")
        return []
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    files.sort()
    return files


def _save_result(filename, data, config, is_text: bool = False):
    """
    Save result to results directory as JSON or plain text.
    """
    try:
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)

        if is_text:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        if getattr(config, "logger", None):
            config.logger.info("Saved result: %s", filepath)
    except Exception as e:
        print(f"   Error saving {filename}: {e}")
        if getattr(config, "logger", None):
            config.logger.error("Failed to save %s: %s", filename, e)


def _generate_demo_report(config, doc_processor):
    """
    Generate a simple Markdown report summarizing the demo run.
    """
    doc_stats = doc_processor.get_document_stats()
    report = f"""# ResearchGPT Assistant - Demonstration Report

## System Overview
- Model: {config.MODEL_NAME}
- Temperature: {config.TEMPERATURE}
- Max Tokens: {config.MAX_TOKENS}

## Documents Processed
{json.dumps(doc_stats, indent=2)}

## Capabilities Demonstrated
1. Document processing (PDF extraction, cleaning, NLTK chunking, TF-IDF search)
2. Similarity search and context-grounded QA
3. Chain-of-Thought (structured reasoning)
4. Self-consistency (multiple candidates, selection)
5. ReAct-style research workflow
6. Answer verification

## Paths
- Results directory: {config.RESULTS_DIR}
- Sample papers directory: {config.SAMPLE_PAPERS_DIR}
"""
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Check the configuration and input documents, then retry.")
