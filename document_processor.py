"""
Document Processing Module for ResearchGPT Assistant (NLTK version)
1) PDF text extraction and cleaning
2) Text preprocessing and chunking (NLTK sent_tokenize)
3) Basic similarity search using TF-IDF
4) Document metadata extraction
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Any

import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK setup ---
import nltk
from nltk.tokenize import sent_tokenize

def _ensure_nltk_data(logger: logging.Logger):
    """Ensure punkt models are available without crashing offline."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK data: punkt")
        nltk.download("punkt", quiet=True)
    # punkt_tab is used by newer NLTK for sentence boundary exceptions
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            logger.info("Downloading NLTK data: punkt_tab")
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            # Safe to ignore if not available in current NLTK version
            pass


class DocumentProcessor:
    def __init__(self, config):
        """
        Initialize Document Processor

        - Store configuration
        - Defer TF-IDF vectorizer initialization (adaptive in build_search_index)
        - Create empty document storage
        - Ensure NLTK data is available
        """
        self.config = config
        self.logger = getattr(config, "logger", logging.getLogger(__name__))
        _ensure_nltk_data(self.logger)

        # Allow language override via config/.env (default: english)
        self.sent_lang = getattr(config, "SENT_TOKENIZE_LANG", os.getenv("SENT_TOKENIZE_LANG", "english"))

        # Defer vectorizer construction so we can adapt params to corpus size
        self.vectorizer: TfidfVectorizer | None = None

        self.documents: Dict[str, Dict[str, Any]] = {}
        self._all_chunks: List[str] = []
        self._chunk_index_map: List[Tuple[str, int]] = []
        self.document_vectors = None

    # ---------------- PDF Extraction ----------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text_parts: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    self.logger.warning("Failed extracting page %d from %s: %s", i, pdf_path, e)
                    page_text = ""
                text_parts.append(page_text)

        raw_text = "\n".join(text_parts)

        # First-pass cleanup for PDF artifacts
        cleaned = raw_text
        cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)  # dehyphenate line breaks
        cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)        # normalize newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)              # collapse blank lines
        cleaned = cleaned.replace("\xa0", " ")
        cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)

        return cleaned.strip()

    # ---------------- Preprocessing ----------------

    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"\bPage\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\d+\s+of\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*[-–—_=]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # ---------------- Chunking (NLTK) ----------------

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks using NLTK sentence tokenization.
        - Packs as many whole sentences as will fit into ~chunk_size chars.
        - Carries character overlap from the tail of the previous chunk.
        """
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if overlap is None:
            overlap = self.config.OVERLAP

        if not text.strip():
            return []

        # Use NLTK for robust sentence boundaries
        sentences = sent_tokenize(text, language=self.sent_lang)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current = ""

        for sent in sentences:
            # If a single sentence is longer than chunk_size, hard-wrap by words
            if len(sent) > chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                words = sent.split()
                buf = ""
                for w in words:
                    add = (w if not buf else " " + w)
                    if len(buf) + len(add) > chunk_size:
                        chunks.append(buf)
                        # char-based overlap tail
                        if overlap > 0 and len(buf) > overlap:
                            buf = buf[-overlap:]
                        else:
                            buf = ""
                        buf += w
                    else:
                        buf += add
                if buf:
                    chunks.append(buf)
                continue

            addition = (sent if not current else " " + sent)
            if len(current) + len(addition) <= chunk_size:
                current += addition
            else:
                if current:
                    chunks.append(current.strip())
                if overlap > 0 and len(current) > overlap:
                    tail = current[-overlap:]
                    current = (tail + " " + sent).strip()
                else:
                    current = sent

        if current:
            chunks.append(current.strip())

        return chunks

    # ---------------- Pipeline ----------------

    def process_document(self, pdf_path: str) -> str:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = os.path.basename(pdf_path).rsplit(".", 1)[0]
        self.logger.info("Processing document: %s", doc_id)

        raw = self.extract_text_from_pdf(pdf_path)
        text = self.preprocess_text(raw)
        chunks = self.chunk_text(text)

        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first_line if len(first_line.split()) >= 3 else doc_id

        metadata = {
            "title": title,
            "num_chars": len(text),
            "num_words": len(text.split()),
            "num_chunks": len(chunks),
            "source_path": pdf_path,
        }

        self.documents[doc_id] = {"title": title, "chunks": chunks, "metadata": metadata}

        self.logger.info("Document %s: %d chars, %d words, %d chunks",
                         doc_id, metadata["num_chars"], metadata["num_words"], metadata["num_chunks"])
        return doc_id

    # ---------------- Indexing (Adaptive TF-IDF) ----------------

    def build_search_index(self) -> None:
        """
        Build a TF-IDF index with parameters that adapt to small corpora.
        Falls back to permissive settings if sklearn raises min_df/max_df conflicts.
        """
        self._all_chunks = []
        self._chunk_index_map = []

        for doc_id, doc in self.documents.items():
            for i, ch in enumerate(doc.get("chunks", [])):
                ch = (ch or "").strip()
                if ch:
                    self._all_chunks.append(ch)
                    self._chunk_index_map.append((doc_id, i))

        if not self._all_chunks:
            self.document_vectors = None
            self.vectorizer = None
            self.logger.warning("No chunks available to index.")
            return

        n = len(self._all_chunks)
        self.logger.info("Building TF-IDF index over %d chunks", n)

        # Choose parameters based on corpus size
        def _make_vec(permissive: bool) -> TfidfVectorizer:
            if permissive:
                return TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,     # allow terms that appear in only 1 chunk
                    max_df=1.0,   # do not filter high-frequency terms
                    norm="l2",
                )
            else:
                return TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,     # require terms appear in >=2 chunks
                    max_df=0.9,   # drop very common terms
                    norm="l2",
                )

        # Use permissive settings for tiny corpora
        use_permissive = n < 5
        self.vectorizer = _make_vec(permissive=use_permissive)

        try:
            self.document_vectors = self.vectorizer.fit_transform(self._all_chunks)
        except ValueError as e:
            # Fallback if sklearn complains about min_df/max_df relationship
            if "max_df corresponds to < documents than min_df" in str(e):
                self.logger.info("TF-IDF fallback to permissive settings due to small corpus.")
                self.vectorizer = _make_vec(permissive=True)
                self.document_vectors = self.vectorizer.fit_transform(self._all_chunks)
            else:
                raise

        self.logger.info("TF-IDF index built. Matrix shape: %s", getattr(self.document_vectors, "shape", None))

    # ---------------- Search ----------------

    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        if self.document_vectors is None or self.vectorizer is None or len(self._all_chunks) == 0:
            raise RuntimeError("Search index is empty. Did you call build_search_index()?")

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.document_vectors)[0]
        top_idx = np.argsort(-sims)[:top_k]

        results: List[Tuple[str, float, str]] = []
        for idx in top_idx:
            chunk_text = self._all_chunks[idx]
            score = float(sims[idx])
            doc_id, _local_idx = self._chunk_index_map[idx]
            results.append((chunk_text, score, doc_id))
        return results

    # ---------------- Stats ----------------

    def get_document_stats(self) -> Dict[str, Any]:
        num_docs = len(self.documents)
        total_chunks = sum(d["metadata"]["num_chunks"] for d in self.documents.values()) if num_docs else 0
        total_words = sum(d["metadata"]["num_words"] for d in self.documents.values()) if num_docs else 0
        avg_len_words = (total_words / num_docs) if num_docs else 0.0
        titles = [d["title"] for d in self.documents.values()]
        return {
            "num_documents": num_docs,
            "total_chunks": total_chunks,
            "average_document_length_words": round(avg_len_words, 2),
            "titles": titles,
        }
