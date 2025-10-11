"""
Testing and Evaluation Script for ResearchGPT Assistant

Covers:
1) Unit tests for document processing
2) Integration tests for prompting strategies (CoT, Self-Consistency, ReAct, basic QA)
3) Agent evaluations (Summarizer, QA, Workflow) with orchestrator if available
4) Performance benchmarks (timings; simple efficiency metrics)
5) Quality scoring (lightweight relevance/coherence proxies)
6) Consolidated evaluation report + JSON artifact
"""

from __future__ import annotations

import os
import re
import time
import json
from typing import Dict, Any, List, Tuple

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

# Orchestrator is optional; tests will adapt if unavailable
try:
    from research_agents import AgentOrchestrator
except Exception:
    AgentOrchestrator = None


class ResearchGPTTester:
    def __init__(self):
        # Initialize system components
        self.config = Config()
        self._ensure_dirs([self.config.RESULTS_DIR])
        self.doc_processor = DocumentProcessor(self.config)
        self.assistant = ResearchGPTAssistant(self.config, self.doc_processor)
        self.orchestrator = AgentOrchestrator(self.assistant) if AgentOrchestrator is not None else None

        # Test set
        self.test_queries: List[str] = [
            "What are the main advantages of machine learning?",
            "How do neural networks process information?",
            "What are the limitations of current AI systems?",
            "Compare supervised and unsupervised learning approaches.",
            "What are the ethical considerations in AI development?",
        ]

        # Accumulators
        self.evaluation_results: Dict[str, Any] = {
            "response_times": [],
            "response_lengths": [],
            "prompt_strategy_comparison": {},
            "agent_performance": {},
            "document_processing": {},
            "performance_benchmark": {},
            "overall_scores": {},
        }

        # Ingest sample PDFs if available; otherwise seed a synthetic doc
        self._prepare_corpus()

    # ---------------------- Corpus Prep ----------------------

    def _prepare_corpus(self) -> None:
        sample_dir = self.config.SAMPLE_PAPERS_DIR
        pdfs = [f for f in os.listdir(sample_dir)] if os.path.exists(sample_dir) else []
        pdfs = [f for f in pdfs if f.lower().endswith(".pdf")]

        if pdfs:
            for name in pdfs:
                try:
                    self.doc_processor.process_document(os.path.join(sample_dir, name))
                except Exception as e:
                    print(f"Warning: failed to process {name}: {e}")
        else:
            # Fallback synthetic doc to enable tests without PDFs
            synthetic_text = (
                "Title: An Introductory Overview of Machine Learning\n\n"
                "Machine learning (ML) provides methods for learning patterns from data. "
                "Supervised learning uses labeled examples to train models for classification and regression. "
                "Unsupervised learning discovers structure such as clusters or latent factors without labels. "
                "Neural networks compose nonlinear functions to learn representations. "
                "Common limitations include data quality, generalization, and interpretability. "
                "Ethical considerations include bias, privacy, transparency, and accountability."
            )
            # Build a pseudo-document entry directly
            doc_id = "synthetic_ml_overview"
            chunks = self.doc_processor.chunk_text(synthetic_text, chunk_size=200, overlap=40)
            self.doc_processor.documents[doc_id] = {
                "title": "An Introductory Overview of Machine Learning",
                "chunks": chunks,
                "metadata": {
                    "title": "An Introductory Overview of Machine Learning",
                    "num_chars": len(synthetic_text),
                    "num_words": len(synthetic_text.split()),
                    "num_chunks": len(chunks),
                    "source_path": "(synthetic)",
                },
            }

        # Build TF-IDF index
        self.doc_processor.build_search_index()

    # ---------------------- Document Processing Tests ----------------------

    def test_document_processing(self) -> Dict[str, Any]:
        """
        Unit checks for preprocessing, chunking, indexing, and similarity search.
        """
        print("\n=== Testing Document Processing ===")
        results = {
            "text_preprocessing": False,
            "chunking": False,
            "index_built": False,
            "similarity_search": False,
            "stats": {},
            "errors": [],
        }

        try:
            raw = "This  is    a   line.\n\n\nPage 12\nAnother-line\nwith hyphen-\nation."
            cleaned = self.doc_processor.preprocess_text(raw)
            results["text_preprocessing"] = bool(cleaned and "Page 12" not in cleaned)

            chunks = self.doc_processor.chunk_text(
                "Sentence one. Sentence two. Sentence three, which is longer.", chunk_size=60, overlap=10
            )
            results["chunking"] = len(chunks) >= 2

            results["index_built"] = self.doc_processor.document_vectors is not None

            # Similarity search sanity
            sim = self.doc_processor.find_similar_chunks("neural networks", top_k=3)
            results["similarity_search"] = isinstance(sim, list)
            results["stats"] = self.doc_processor.get_document_stats()

            print("   ✓ Document processing: PASS")
        except Exception as e:
            results["errors"].append(str(e))
            print(f"   ✗ Document processing error: {e}")

        self.evaluation_results["document_processing"] = results
        return results

    # ---------------------- Prompting Strategies ----------------------

    def test_prompting_strategies(self) -> Dict[str, Any]:
        """
        Compare Chain-of-Thought, Self-Consistency, ReAct, and basic QA using timing and length.
        """
        print("\n=== Testing Prompting Strategies ===")
        comp = {"chain_of_thought": [], "self_consistency": [], "react_workflow": [], "basic_qa": []}

        for i, query in enumerate(self.test_queries[:3]):
            print(f"   Query {i+1}: {query}")
            try:
                # Chain-of-Thought
                start = time.time()
                chunks = self.doc_processor.find_similar_chunks(query, top_k=5)
                cot = self.assistant.chain_of_thought_reasoning(query, chunks)
                t_cot = time.time() - start
                comp["chain_of_thought"].append(
                    {"query": query, "time": t_cot, "len": len(cot or ""), "quality": self.evaluate_response_quality(cot, query)}
                )

                # Self-Consistency
                start = time.time()
                sc = self.assistant.self_consistency_generate(query, chunks, num_attempts=3)
                t_sc = time.time() - start
                comp["self_consistency"].append(
                    {"query": query, "time": t_sc, "len": len(sc or ""), "quality": self.evaluate_response_quality(sc, query)}
                )

                # ReAct
                start = time.time()
                react = self.assistant.react_research_workflow(query)
                t_react = time.time() - start
                comp["react_workflow"].append(
                    {
                        "query": query,
                        "time": t_react,
                        "steps": len(react.get("workflow_steps", [])),
                        "len": len((react.get("final_answer") or "")),
                        "quality": self.evaluate_response_quality(react.get("final_answer", ""), query),
                    }
                )

                # Basic QA
                start = time.time()
                qa = self.assistant.answer_simple_question(query, top_k=5)
                t_qa = time.time() - start
                comp["basic_qa"].append(
                    {"query": query, "time": t_qa, "len": len(qa or ""), "quality": self.evaluate_response_quality(qa, query)}
                )

                print("   ✓ Completed")
            except Exception as e:
                print(f"   ✗ Strategy error: {e}")

        self.evaluation_results["prompt_strategy_comparison"] = comp
        return comp

    # ---------------------- Agent Performance ----------------------

    def test_agent_performance(self) -> Dict[str, Any]:
        """
        Exercise Summarizer, QA, and Research Workflow agents (if orchestrator available).
        """
        print("\n=== Testing AI Agents ===")
        results = {"summarizer_agent": {}, "qa_agent": {}, "workflow_agent": {}, "orchestrator": bool(self.orchestrator)}

        if not self.orchestrator:
            print("   Orchestrator unavailable; skipping agent tests.")
            self.evaluation_results["agent_performance"] = results
            return results

        try:
            # Choose a doc_id to summarize
            any_doc_id = next(iter(self.assistant.doc_processor.documents.keys()), None)
            if any_doc_id:
                sum_res = self.orchestrator.route_task("summarizer", {"doc_id": any_doc_id})
                results["summarizer_agent"] = {"doc_id": any_doc_id, "summary_len": len((sum_res.get("summary") or ""))}
                print("   ✓ Summarizer agent")
            else:
                results["summarizer_agent"] = {"error": "No documents available"}

            qa_res = self.orchestrator.route_task(
                "qa", {"question": "What is supervised learning?", "type": "factual"}
            )
            results["qa_agent"] = {"answer_len": len((qa_res.get("answer") or qa_res.get("analysis") or ""))}
            print("   ✓ QA agent")

            wf_res = self.orchestrator.route_task("workflow", {"research_topic": "neural networks applications"})
            results["workflow_agent"] = {
                "questions": len(wf_res.get("generated_questions", [])) if isinstance(wf_res, dict) else 0
            }
            print("   ✓ Workflow agent")

        except Exception as e:
            results["error"] = f"Agent error: {e}"
            print(f"   ✗ Agent error: {e}")

        self.evaluation_results["agent_performance"] = results
        return results

    # ---------------------- Performance Benchmark ----------------------

    def run_performance_benchmark(self) -> Dict[str, Any]:
        """
        Measure response times for key calls and compute simple efficiency metrics.
        """
        print("\n=== Running Performance Benchmark ===")
        bench = {"document_processing_time": 0.0, "query_response_times": [], "system_efficiency": {}}

        # Measure similarity search time
        start = time.time()
        _ = self.assistant.doc_processor.find_similar_chunks("baseline timing test", top_k=3)
        bench["document_processing_time"] = time.time() - start

        # Measure QA response times
        for q in self.test_queries[:2]:
            start = time.time()
            try:
                res = self.assistant.answer_research_question(q, use_cot=False, use_verification=False)
                elapsed = time.time() - start
                bench["query_response_times"].append(
                    {"query": q, "response_time": elapsed, "response_length": len(res.get("answer", ""))}
                )
            except Exception as e:
                bench["query_response_times"].append({"query": q, "error": str(e), "response_time": None})

        # Aggregate metrics
        times = [r["response_time"] for r in bench["query_response_times"] if r.get("response_time") is not None]
        avg = sum(times) / len(times) if times else 0.0
        bench["system_efficiency"] = {
            "average_response_time": round(avg, 3),
            "queries_per_minute": round((60.0 / avg) if avg > 0 else 0.0, 2),
        }

        print(f"   Average QA response time: {bench['system_efficiency']['average_response_time']} s")
        self.evaluation_results["performance_benchmark"] = bench
        return bench

    # ---------------------- Quality Scoring ----------------------

    def evaluate_response_quality(self, response: str, query: str) -> Dict[str, float]:
        """
        Lightweight proxy metrics:
        - length_score: normalized length (0..1)
        - keyword_relevance: Jaccard overlap of stem-like tokens between query and response
        - coherence_score: penalty for over-fragmentation (heuristic)
        - completeness_score: presence of concluding tokens/phrases
        """
        response = (response or "").strip()
        query = (query or "").strip()

        # Token sets (simple, lowercase words of length >= 3)
        def toks(t: str) -> List[str]:
            return re.findall(r"[A-Za-z]{3,}", t.lower())

        q_set, r_set = set(toks(query)), set(toks(response))
        jacc = (len(q_set & r_set) / len(q_set | r_set)) if (q_set | r_set) else 0.0

        # Length proxy
        length_score = min(len(response) / 300.0, 1.0)

        # Coherence proxy: penalize too many very short lines
        lines = [ln for ln in response.splitlines() if ln.strip()]
        short_lines = sum(1 for ln in lines if len(ln) < 25)
        coherence_score = max(0.0, 1.0 - (short_lines / max(1, len(lines)))) if lines else 0.0

        # Completeness proxy: look for concluding markers
        completeness_score = 1.0 if re.search(r"\btherefore\b|\bin summary\b|\bconclusion\b|\boverall\b", response, re.I) else 0.6

        overall = (length_score + jacc + coherence_score + completeness_score) / 4.0

        return {
            "length_score": round(length_score, 3),
            "keyword_relevance": round(jacc, 3),
            "coherence_score": round(coherence_score, 3),
            "completeness_score": round(completeness_score, 3),
            "overall_score": round(overall, 3),
        }

    # ---------------------- Report ----------------------

    def generate_evaluation_report(self) -> str:
        """
        Assemble a markdown report summarizing tests and metrics.
        """
        comp = self.evaluation_results.get("prompt_strategy_comparison", {})
        agents = self.evaluation_results.get("agent_performance", {})
        bench = self.evaluation_results.get("performance_benchmark", {})
        docs = self.evaluation_results.get("document_processing", {})

        report = f"""# ResearchGPT Assistant – Evaluation Report

## 1. Document Processing
- Preprocessing: {docs.get('text_preprocessing')}
- Chunking: {docs.get('chunking')}
- Index built: {docs.get('index_built')}
- Similarity search: {docs.get('similarity_search')}
- Stats:
{json.dumps(docs.get('stats', {}), indent=2)}
## 2. Prompting Strategy Comparison
{json.dumps(comp, indent=2)}
## 3. Agent Performance
{json.dumps(agents, indent=2)}
## 4. Performance Benchmark
{json.dumps(bench, indent=2)}

## 5. Notes
- Scores are heuristic proxies to support quick iteration.
- For rigorous evaluation, consider human judgments and task-specific gold sets.

"""
        return report

    # ---------------------- Runner ----------------------

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute the complete suite and save artifacts.
        """
        print("Starting ResearchGPT Assistant Test Suite...")

        doc_results = self.test_document_processing()
        comp_results = self.test_prompting_strategies()
        agent_results = self.test_agent_performance()
        bench_results = self.run_performance_benchmark()

        # Persist results
        self.evaluation_results.update(
            {
                "document_processing": doc_results,
                "prompt_strategy_comparison": comp_results,
                "agent_performance": agent_results,
                "performance_benchmark": bench_results,
            }
        )

        report_md = self.generate_evaluation_report()
        self._ensure_dirs([self.config.RESULTS_DIR])
        with open(os.path.join(self.config.RESULTS_DIR, "evaluation_report.md"), "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(os.path.join(self.config.RESULTS_DIR, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        print("\n=== Test Suite Complete ===")
        print("Results saved:")
        print("- evaluation_report.md")
        print("- test_results.json")
        return self.evaluation_results

    # ---------------------- Utilities ----------------------

    @staticmethod
    def _ensure_dirs(paths: List[str]) -> None:
        for p in paths:
            os.makedirs(p, exist_ok=True)


if __name__ == "__main__":
    tester = ResearchGPTTester()
    _ = tester.run_all_tests()
