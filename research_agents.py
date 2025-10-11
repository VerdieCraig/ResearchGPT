"""
AI Research Agents for Specialized Tasks

Implements:
- Base agent class with common functionality
- SummarizerAgent for single/multi-document summaries
- QAAgent for factual and analytical QA
- ResearchWorkflowAgent for end-to-end research sessions
- AgentOrchestrator for task routing and coordination
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json
import re


class BaseAgent:
    """
    Common interface and utilities for all agents.
    """

    def __init__(self, research_assistant):
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Each agent must implement execute_task().")

    # ---- Shared helpers ----

    def _get_doc_text(self, doc_id: str) -> str:
        """
        Retrieve full text of a processed document by concatenating its chunks.
        """
        docs = self.assistant.doc_processor.documents
        if doc_id not in docs:
            raise ValueError(f"Unknown document id: {doc_id}")
        return "\n\n".join(docs[doc_id]["chunks"])

    def _format_context_from_doc_ids(self, doc_ids: List[str], max_chars: int = 6000) -> str:
        """
        Build a compact context block from multiple docs (concatenated chunks with headers).
        """
        parts: List[str] = []
        used = 0
        for did in doc_ids:
            text = self._get_doc_text(did)
            header = f"[{did}]"
            block = f"{header}\n{text}"
            if used + len(block) + 2 > max_chars:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts) if parts else "No context available."

    def _format_context_from_chunks(self, chunk_tuples: List[Tuple[str, float, str]], max_chars: int = 4000) -> str:
        """
        Convert [(chunk, score, doc_id), ...] into a compact context string.
        """
        parts: List[str] = []
        used = 0
        for chunk, score, doc_id in chunk_tuples:
            block = f"[{doc_id}] (sim={score:.3f})\n{chunk}"
            if used + len(block) + 2 > max_chars:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts) if parts else "No relevant context available."

    def _mean_similarity(self, chunk_tuples: List[Tuple[str, float, str]]) -> float:
        """
        Compute mean cosine similarity from (chunk, score, doc_id) tuples.
        """
        if not chunk_tuples:
            return 0.0
        return float(sum(s for _c, s, _d in chunk_tuples) / len(chunk_tuples))


class SummarizerAgent(BaseAgent):
    """
    Document summarization agent (single and multi-document).
    """

    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Summarize a single document: key findings, methodology, limitations, conclusions.
        """
        try:
            text = self._get_doc_text(doc_id)
        except Exception as e:
            return {"error": f"Failed to load document '{doc_id}': {e}"}

        prompt = self.assistant.prompts.get("document_summary") or (
            "Summarize the following document excerpts focusing on key findings, methodology, "
            "limitations, and conclusions.\n\nExcerpts:\n{context}\n\nSummary:"
        )
        summary = self.assistant._call_mistral(prompt.format(context=text), temperature=0.2)

        summary_data = {
            "doc_id": doc_id,
            "summary": summary,
            "word_count": len(summary.split()),
            "key_topics": self._extract_key_topics(summary),
        }
        return summary_data

    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Create a multi-document overview: common themes, differences, gaps, directions.
        """
        indiv: List[Dict[str, Any]] = []
        for did in doc_ids:
            indiv.append(self.summarize_document(did))

        # Build a compact JSON of individual summaries for the model to analyze.
        summaries_json = json.dumps(
            [{"doc_id": s.get("doc_id"), "summary": s.get("summary")} for s in indiv],
            ensure_ascii=False,
            indent=2,
        )

        overview_prompt = (
            "You are given summaries of multiple research papers in JSON. "
            "Produce a concise literature overview covering: common themes, differing methodologies, "
            "consistent findings vs. contradictions, research gaps, and future directions.\n\n"
            f"Summaries JSON:\n{summaries_json}\n\nOverview:"
        )
        overview = self.assistant._call_mistral(overview_prompt, temperature=0.2)

        return {
            "overview": overview,
            "papers_analyzed": len(doc_ids),
            "individual_summaries": indiv,
        }

    def _extract_key_topics(self, text: str, max_topics: int = 8) -> List[str]:
        """
        Simple heuristic keyword extraction from the summary (lowercase, deduplicate).
        """
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "over", "such",
            "are", "was", "were", "using", "based", "paper", "study", "approach", "method",
            "methods", "results", "conclusion", "conclusions", "findings"
        }
        freq: Dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in ranked[:max_topics]]

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            ids = task_input.get("doc_ids") or []
            return self.create_literature_overview(ids)
        return {"error": "Invalid task input for SummarizerAgent"}


class QAAgent(BaseAgent):
    """
    Question answering agent (factual and analytical).
    """

    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"

    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        """
        Factual QA grounded in local document context. Returns answer, sources, and confidence.
        """
        relevant = self.assistant.doc_processor.find_similar_chunks(question, top_k=5)
        context = self._format_context_from_chunks(relevant)

        prompt = self.assistant.prompts.get("qa_with_context") or (
            "Answer the question only using the context below. If information is insufficient, say so.\n\n"
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer with brief citations in brackets (doc_id)."
        )
        answer = self.assistant._call_mistral(
            prompt.format(question=question, context=context),
            temperature=0.2,
            system="Constrain to context and provide concise citations like [doc_id].",
        )

        confidence = max(0.0, min(1.0, self._mean_similarity(relevant)))
        return {
            "question": question,
            "answer": answer,
            "sources": [doc_id for (_c, _s, doc_id) in relevant],
            "confidence": round(confidence, 3),
        }

    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        """
        Analytical QA using chain-of-thought style prompting (concise steps + synthesized answer).
        """
        chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=5)
        response = self.assistant.chain_of_thought_reasoning(question, chunks)
        return {
            "question": question,
            "analysis": response,
            "reasoning_type": "chain_of_thought",
            "sources": [doc_id for (_c, _s, doc_id) in chunks],
        }

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        question = task_input.get("question", "")
        qtype = task_input.get("type", "factual")
        if not question:
            return {"error": "Missing 'question' for QAAgent"}
        if qtype == "analytical":
            return self.answer_analytical_question(question)
        return self.answer_factual_question(question)


class ResearchWorkflowAgent(BaseAgent):
    """
    End-to-end research session agent.
    """

    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        self.summarizer = SummarizerAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)

    def _generate_research_questions(self, topic: str, k_min: int = 3, k_max: int = 5) -> List[str]:
        prompt = (
            "Generate between {kmin} and {kmax} specific, answerable research questions covering different aspects "
            "(what, how, why, implications) of the topic below. Return them as a numbered list only.\n\n"
            f"Topic: {topic}\n\nQuestions:"
        ).format(kmin=k_min, kmax=k_max)
        raw = self.assistant._call_mistral(prompt, temperature=0.2)
        # Extract lines that look like numbered bullets
        qs = [re.sub(r"^\s*\d+\s*[\).\-\:]\s*", "", ln).strip() for ln in raw.splitlines()]
        qs = [q for q in qs if len(q.split()) >= 3]
        return qs[:k_max] if qs else []

    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        """
        Workflow:
        1) Generate questions
        2) Retrieve relevant docs and create literature overview
        3) Answer generated questions (analytical)
        4) Identify gaps and future directions
        """
        results: Dict[str, Any] = {
            "research_topic": research_topic,
            "generated_questions": [],
            "document_analysis": {},
            "answers": [],
            "research_gaps": "",
            "future_directions": "",
        }

        # 1) Questions
        questions = self._generate_research_questions(research_topic)
        results["generated_questions"] = questions

        # 2) Relevant documents and overview
        relevant_chunks = self.assistant.doc_processor.find_similar_chunks(research_topic, top_k=10)
        doc_ids = list(dict.fromkeys([doc_id for (_c, _s, doc_id) in relevant_chunks]))
        if doc_ids:
            overview = self.summarizer.create_literature_overview(doc_ids)
            results["document_analysis"] = overview

        # 3) Answer questions (analytical)
        answers: List[Dict[str, Any]] = []
        for q in questions:
            ans = self.qa_agent.answer_analytical_question(q)
            answers.append(ans)
        results["answers"] = answers

        # 4) Gaps and future directions
        overview_text = overview["overview"] if doc_ids else ""
        answers_text = "\n\n".join([f"Q: {a['question']}\nA: {a.get('analysis','')}" for a in answers])
        gaps_prompt = (
            "Based on the literature overview and the answers, identify key research gaps and propose concrete "
            "future research directions (3–5 items each).\n\nOverview:\n{ov}\n\nAnswers:\n{ans}\n\n"
            "Format:\nGaps:\n- ...\n- ...\n\nFuture Directions:\n- ...\n- ..."
        ).format(ov=overview_text, ans=answers_text)
        gaps_text = self.assistant._call_mistral(gaps_prompt, temperature=0.2)

        # Split into gaps and directions if possible
        gaps, directions = self._split_gaps_and_directions(gaps_text)
        results["research_gaps"] = gaps
        results["future_directions"] = directions

        return results

    def _split_gaps_and_directions(self, text: str) -> Tuple[str, str]:
        m = re.split(r"\bFuture\s*Directions\s*:\b", text, flags=re.IGNORECASE)
        if len(m) == 2:
            return m[0].strip(), m[1].strip()
        return text.strip(), ""

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "research_topic" in task_input:
            return self.conduct_research_session(task_input["research_topic"])
        return {"error": "Invalid task input for ResearchWorkflowAgent"}


class AgentOrchestrator:
    """
    Orchestrates multiple agents and routes tasks.
    """

    def __init__(self, research_assistant):
        self.assistant = research_assistant
        self.agents: Dict[str, BaseAgent] = {
            "summarizer": SummarizerAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant),
        }

    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to an appropriate agent based on task_type.
        Supported types: 'summarizer', 'qa', 'workflow'
        """
        agent = self.agents.get(task_type)
        if not agent:
            return {"error": f"Unknown task type: {task_type}"}
        try:
            return agent.execute_task(task_input)
        except Exception as e:
            return {"error": f"Agent '{task_type}' failed: {e}"}

    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """
        Minimal multi-agent coordination:
        - If description includes 'summarize', try summarizer
        - If includes 'qa', try QA
        - If includes 'workflow', run research workflow

        This is a simple placeholder you can extend with a real planner.
        """
        desc = workflow_description.lower()
        results: Dict[str, Any] = {"workflow_description": workflow_description, "steps_executed": [], "final_result": {}}

        try:
            if "workflow" in desc:
                results["steps_executed"].append("workflow")
                results["final_result"]["workflow"] = self.route_task("workflow", {"research_topic": workflow_description})

            if "summarize" in desc:
                # As a simple heuristic, summarize top-1 relevant doc for the topic
                rel = self.assistant.doc_processor.find_similar_chunks(workflow_description, top_k=1)
                if rel:
                    doc_id = rel[0][2]
                    results["steps_executed"].append("summarizer")
                    results["final_result"]["summary"] = self.route_task("summarizer", {"doc_id": doc_id})

            if "qa" in desc or "answer" in desc or "question" in desc:
                results["steps_executed"].append("qa")
                results["final_result"]["qa"] = self.route_task("qa", {"question": workflow_description, "type": "analytical"})

        except Exception as e:
            results["error"] = f"Complex workflow failed: {e}"

        return results
