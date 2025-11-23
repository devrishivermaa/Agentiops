# agents/worker_agent.py

"""
WorkerAgent for ResidualAgent Option B.
Supports:
1. Receiving global context from ResidualAgent/SubMaster
2. Using global context in page-level LLM analysis
"""

import os
import time
import ray
from typing import Dict, Any, Optional

from utils.logger import get_logger
from utils.llm_helper import LLMProcessor, analyze_page

logger = get_logger("WorkerAgent")


@ray.remote
class WorkerAgent:

    def __init__(self, worker_id: str, llm_model: str = None, processing_requirements: list = None):
        self.worker_id = worker_id
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mistral-small-latest")
        self.processing_requirements = processing_requirements or []
        self.llm_processor = None
        self.global_context = {}   # <-- new storage

        logger.info(f"[{worker_id}] WorkerAgent initialized | model={self.llm_model}")

    # ----------------------------------------------------------
    # Receive global context from SubMaster or ResidualAgent
    # ----------------------------------------------------------
    def set_global_context(self, context: Dict[str, Any]):
        self.global_context = context or {}
        logger.info(f"[{self.worker_id}] Global context received")
        return {"worker_id": self.worker_id, "status": "context_received"}

    # ----------------------------------------------------------
    def initialize(self):
        """Initialize local LLM processor."""
        try:
            self.llm_processor = LLMProcessor(
                model=self.llm_model,
                temperature=0.3,
                max_retries=5,
                caller_id=self.worker_id
            )
            logger.info(f"[{self.worker_id}] LLM processor initialized")
            return {"worker_id": self.worker_id, "status": "ready"}

        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM initialization failed: {e}")
            return {
                "worker_id": self.worker_id,
                "status": "error",
                "error": str(e)
            }

    # ----------------------------------------------------------
    def process_page(self, page_num: int, text: str, role: str,
                     section_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process one page using LLM analysis.
        Enhanced with global_context injection.
        """
        start_time = time.time()
        logger.info(f"[{self.worker_id}] Processing page {page_num}")

        result = {
            "page": page_num,
            "section": section_name or "Unknown",
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "char_count": len(text),
            "worker_id": self.worker_id,
            "global_context_used": bool(self.global_context)
        }

        if not self.llm_processor:
            logger.warning(f"[{self.worker_id}] LLM not initialized")
            result.update({
                "summary": "[LLM not available]",
                "entities": [],
                "keywords": [],
                "status": "error"
            })
            return result

        if len(text.strip()) < 30:
            result.update({
                "summary": "[Text too short for analysis]",
                "entities": [],
                "keywords": [],
                "status": "skipped"
            })
            return result

        try:
            # Pass global_context to analyze_page
            analysis = analyze_page(
                llm_processor=self.llm_processor,
                role=role,
                text=text,
                page_num=page_num,
                section_name=section_name,
                processing_requirements=self.processing_requirements,
                global_context=self.global_context     # <-- key fix
            )

            result.update(analysis)
            result["status"] = "success"

            logger.info(
                f"[{self.worker_id}] Page {page_num} processed: "
                f"{len(result.get('entities', []))} entities, "
                f"{len(result.get('keywords', []))} keywords"
            )

        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM failed for page {page_num}: {e}")
            result.update({
                "summary": "[LLM failed]",
                "entities": [],
                "keywords": [],
                "llm_error": str(e),
                "status": "error"
            })

        result["processing_time"] = time.time() - start_time
        return result
