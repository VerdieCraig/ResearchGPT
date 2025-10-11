"""
Configuration file for ResearchGPT Assistant
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

class Config:
    """
    Reference implementation for the AI Research Agent configuration.

    - 1) Mistral API configuration (with required key)
    - 2) Processing parameters (chunk size, overlap)
    - 3) Directory paths for data and results
    - 4) Model parameters (temperature, max tokens)
    - 5) Logging configuration
    - 6) API robustness knobs (retries, backoff, throttling, model fallbacks)
    """

    def __init__(self):
        # Load environment variables from .env (if present) and system env
        load_dotenv()

        # 1) Mistral API configuration
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")  # REQUIRED (validated below)

        # 4) Model parameters
        self.MODEL_NAME  = os.getenv("MODEL_NAME", "mistral-medium")
        self.TEMPERATURE = self._to_float(os.getenv("TEMPERATURE", "0.1"))
        self.MAX_TOKENS  = self._to_int(os.getenv("MAX_TOKENS", "1000"))

        # 6) API robustness knobs (used by research_assistant.py)
        self.API_MAX_RETRIES         = self._to_int(os.getenv("API_MAX_RETRIES", "4"))
        self.API_RETRY_BACKOFF_BASE  = self._to_float(os.getenv("API_RETRY_BACKOFF_BASE", "1.5"))
        self.API_RETRY_BACKOFF_MAX   = self._to_float(os.getenv("API_RETRY_BACKOFF_MAX", "12"))
        self.API_THROTTLE_SEC        = self._to_float(os.getenv("API_THROTTLE_SEC", "0.75"))
        # Comma-separated list of fallback models, e.g. "mistral-small-latest,mistral-tiny-latest"
        self.MODEL_FALLBACKS = self._parse_list(os.getenv("MODEL_FALLBACKS", "mistral-small-latest,mistral-tiny-latest"))

        # 2) Processing parameters
        self.CHUNK_SIZE = self._to_int(os.getenv("CHUNK_SIZE", "1000"))
        self.OVERLAP    = self._to_int(os.getenv("OVERLAP", "100"))
        # Optional: NLTK sentence tokenizer language (used by DocumentProcessor)
        self.SENT_TOKENIZE_LANG = os.getenv("SENT_TOKENIZE_LANG", "english")

        # 3) Directory paths
        self.DATA_DIR            = os.getenv("DATA_DIR", "data")
        self.SAMPLE_PAPERS_DIR   = os.getenv("SAMPLE_PAPERS_DIR", os.path.join(self.DATA_DIR, "sample_papers"))
        self.PROCESSED_DIR       = os.getenv("PROCESSED_DIR", os.path.join(self.DATA_DIR, "processed"))
        self.RESULTS_DIR         = os.getenv("RESULTS_DIR", "results")

        # 5) Logging configuration
        self.LOG_DIR   = os.getenv("LOG_DIR", "logs")
        self.LOG_FILE  = os.getenv("LOG_FILE", "app.log")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        # Initialize logging before validation so errors are visible
        self.logger = self._configure_logger()

        # Ensure required directories exist
        self._ensure_dirs()

        # Validate configuration (required + ranges)
        self.validate_required()
        self.validate_ranges()

        self.logger.info("Configuration initialized successfully.")
        self.logger.debug("Effective configuration: %s", self.as_dict(redact_secrets=True))

    # ---------- Public API ----------

    def validate_required(self) -> None:
        """Validate presence of required values (API key, etc.)."""
        if not self.MISTRAL_API_KEY or not self.MISTRAL_API_KEY.strip():
            msg = (
                "Missing required environment variable: MISTRAL_API_KEY\n"
                "Add it to your .env file, e.g.\n\n    MISTRAL_API_KEY=sk-your-real-key\n"
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def validate_ranges(self) -> None:
        """Validate numeric ranges and relationships."""
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
        if self.MAX_TOKENS <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not (0 <= self.OVERLAP < self.CHUNK_SIZE):
            raise ValueError("OVERLAP must be >= 0 and strictly less than CHUNK_SIZE")
        if self.API_MAX_RETRIES < 1:
            raise ValueError("API_MAX_RETRIES must be >= 1")
        if self.API_RETRY_BACKOFF_BASE <= 0 or self.API_RETRY_BACKOFF_MAX <= 0:
            raise ValueError("API_RETRY_BACKOFF_BASE and API_RETRY_BACKOFF_MAX must be positive")
        if self.API_THROTTLE_SEC < 0:
            raise ValueError("API_THROTTLE_SEC must be >= 0")

    def as_dict(self, *, redact_secrets: bool = False) -> dict:
        """Return the effective config (optionally redacting secrets)."""
        d = {
            "MODEL_NAME": self.MODEL_NAME,
            "TEMPERATURE": self.TEMPERATURE,
            "MAX_TOKENS": self.MAX_TOKENS,
            "DATA_DIR": self.DATA_DIR,
            "SAMPLE_PAPERS_DIR": self.SAMPLE_PAPERS_DIR,
            "PROCESSED_DIR": self.PROCESSED_DIR,
            "RESULTS_DIR": self.RESULTS_DIR,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "OVERLAP": self.OVERLAP,
            "SENT_TOKENIZE_LANG": self.SENT_TOKENIZE_LANG,
            "LOG_DIR": self.LOG_DIR,
            "LOG_FILE": self.LOG_FILE,
            "LOG_LEVEL": self.LOG_LEVEL,
            # API robustness
            "API_MAX_RETRIES": self.API_MAX_RETRIES,
            "API_RETRY_BACKOFF_BASE": self.API_RETRY_BACKOFF_BASE,
            "API_RETRY_BACKOFF_MAX": self.API_RETRY_BACKOFF_MAX,
            "API_THROTTLE_SEC": self.API_THROTTLE_SEC,
            "MODEL_FALLBACKS": self.MODEL_FALLBACKS,
        }
        d["MISTRAL_API_KEY"] = "***redacted***" if redact_secrets else self.MISTRAL_API_KEY
        return d

    # ---------- Internal helpers ----------

    def _configure_logger(self) -> logging.Logger:
        os.makedirs(self.LOG_DIR, exist_ok=True)

        logger = logging.getLogger("ai_research_agent")
        if logger.handlers:
            # If re-instantiated, just update level and reuse handlers
            logger.setLevel(getattr(logging, self.LOG_LEVEL, logging.INFO))
            return logger

        level = getattr(logging, self.LOG_LEVEL, logging.INFO)
        logger.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Rotating file
        fh = RotatingFileHandler(
            filename=os.path.join(self.LOG_DIR, self.LOG_FILE),
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    def _ensure_dirs(self) -> None:
        for d in [self.DATA_DIR, self.SAMPLE_PAPERS_DIR, self.PROCESSED_DIR, self.RESULTS_DIR, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _to_int(val: str) -> int:
        try:
            return int(val)
        except ValueError as e:
            raise ValueError(f"Expected integer but got '{val}'") from e

    @staticmethod
    def _to_float(val: str) -> float:
        try:
            return float(val)
        except ValueError as e:
            raise ValueError(f"Expected float but got '{val}'") from e

    @staticmethod
    def _parse_list(val: str) -> list:
        if not val:
            return []
        return [x.strip() for x in val.split(",") if x.strip()]
