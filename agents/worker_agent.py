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
        self.global_context = {}

        # -------------------------------------------------
        # Worker-level memory cache (LRU style)
        # -------------------------------------------------
        self.cache = {}              # key → result
        self.cache_access = {}       # key → timestamp
        self.cache_max_items = 500   # Before eviction

        logger.info(f"[{worker_id}] WorkerAgent initialized | model={self.llm_model}")

    # ----------------------------------------------------------
    # Receive global context → reset cache (context changes output)
    # ----------------------------------------------------------
    def set_global_context(self, context: Dict[str, Any]):
        self.global_context = context or {}
        self.cache.clear()           # invalidate old context-based cache
        self.cache_access.clear()
        logger.info(f"[{self.worker_id}] Global context received. Cache cleared.")
        return {"worker_id": self.worker_id, "status": "context_received"}

    # ----------------------------------------------------------
    def initialize(self):
        """Initialize LLM processor."""
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
            return {"worker_id": self.worker_id, "status": "error", "error": str(e)}

    # ----------------------------------------------------------
    # INTERNAL: create stable hash cache key
    # ----------------------------------------------------------
    def _make_cache_key(self, page_num, text, role, section_name):
        import hashlib
        raw = (
            f"{page_num}|{role}|{section_name}|"
            f"{text}|"
            f"{self.processing_requirements}|"
            f"{str(self.global_context)}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------------------------------
    # INTERNAL: perform LRU eviction
    # ----------------------------------------------------------
    def _evict_cache_if_needed(self):
        if len(self.cache) < self.cache_max_items:
            return

        # evict least recently accessed 20 percent
        import heapq
        evict_count = max(1, self.cache_max_items // 5)

        oldest = heapq.nsmallest(evict_count, self.cache_access.items(), key=lambda x: x[1])
        for key, _ in oldest:
            self.cache.pop(key, None)
            self.cache_access.pop(key, None)

        logger.info(f"[{self.worker_id}] Cache eviction complete, removed {evict_count} items")

    # ----------------------------------------------------------
    def process_page(self, page_num: int, text: str, role: str,
                     section_name: Optional[str] = None) -> Dict[str, Any]:

        start_time = time.time()

        # ------------------------------
        # 1) Build result base
        # ------------------------------
        result = {
            "page": page_num,
            "section": section_name or "Unknown",
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "char_count": len(text),
            "worker_id": self.worker_id,
            "global_context_used": bool(self.global_context)
        }

        # ------------------------------
        # 2) Cache Key
        # ------------------------------
        cache_key = self._make_cache_key(page_num, text, role, section_name)

        # ------------------------------
        # 3) Return from cache if available
        # ------------------------------
        if cache_key in self.cache:
            self.cache_access[cache_key] = time.time()
            cached = self.cache[cache_key]
            logger.info(f"[{self.worker_id}] Cache HIT for page {page_num}")
            cached["from_cache"] = True
            cached["processing_time"] = time.time() - start_time
            return cached

        logger.info(f"[{self.worker_id}] Cache MISS for page {page_num}")

        # ------------------------------
        # 4) If LLM not available
        # ------------------------------
        if not self.llm_processor:
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

        # ------------------------------
        # 5) Real LLM Analysis
        # ------------------------------
        try:
            analysis = analyze_page(
                llm_processor=self.llm_processor,
                role=role,
                text=text,
                page_num=page_num,
                section_name=section_name,
                processing_requirements=self.processing_requirements,
                global_context=self.global_context
            )

            result.update(analysis)
            result["status"] = "success"

        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM failed for page {page_num}: {e}")
            result.update({
                "summary": "[LLM failed]",
                "entities": [],
                "keywords": [],
                "llm_error": str(e),
                "status": "error"
            })

        # ------------------------------
        # 6) Save to cache
        # ------------------------------
        self.cache[cache_key] = result
        self.cache_access[cache_key] = time.time()
        self._evict_cache_if_needed()

        result["from_cache"] = False
        result["processing_time"] = time.time() - start_time
        return result
