# agents/sub_master.py

import time
import uuid
import ray
from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from agents.worker_agent import WorkerAgent
from pymongo import MongoClient
import os

logger = get_logger("SubMaster")


@ray.remote
class SubMaster:

    def __init__(self, plan_piece: Dict[str, Any], metadata: Dict[str, Any], residual_handle: Optional[Any] = None):
        self.sm_id = plan_piece.get("submaster_id", f"SM-{uuid.uuid4().hex[:6].upper()}")
        self.role = plan_piece.get("role", "generic")
        self.sections = plan_piece.get("assigned_sections", [])
        self.pages = plan_piece.get("page_range", [1, 1])
        self.meta = metadata
        self.status = "initialized"

        self.residual = residual_handle
        self.global_context: Dict[str, Any] = {}

        # -----------------------------------
        # Mongo initialization
        # -----------------------------------
        try:
            uri = os.getenv("MONGO_URI")
            dbname = os.getenv("MONGO_DB")
            sm_coll = os.getenv("MONGO_SUBMASTER_COLLECTION", "submaster_results")
            wk_coll = os.getenv("MONGO_WORKER_COLLECTION", "worker_results")

            if uri and dbname:
                client = MongoClient(uri)
                db = client[dbname]
                self.mongo_sm_coll = db[sm_coll]
                self.mongo_worker_coll = db[wk_coll]
                logger.info(f"[{self.sm_id}] Mongo connected")
            else:
                self.mongo_sm_coll = None
                self.mongo_worker_coll = None
        except Exception as e:
            logger.error(f"[{self.sm_id}] Mongo init failed {e}")
            self.mongo_sm_coll = None
            self.mongo_worker_coll = None

        # -----------------------------------
        # Runtime config
        # -----------------------------------
        self.pdf_path = metadata.get("file_path", "")
        self.processing_requirements = metadata.get("processing_requirements", [])
        self.llm_model = metadata.get("preferred_model", "mistral-small-latest")
        self.num_workers_per_submaster = metadata.get("num_workers_per_submaster", 3)

        self.pdf_extractor: Optional[PDFExtractor] = None
        self.workers: List[Any] = []

        logger.info(f"[{self.sm_id}] SubMaster initialized with {self.num_workers_per_submaster} workers")

    # ----------------------------------------------------------
    # allow orchestrator to fetch worker handles
    # ----------------------------------------------------------
    def get_worker_handles(self):
        return self.workers

    # ----------------------------------------------------------
    # Receive global context and forward to workers
    # ----------------------------------------------------------
    def set_global_context(self, context: Dict[str, Any]):
        try:
            self.global_context = context or {}
            logger.info(f"[{self.sm_id}] Received global context with {len(self.global_context.keys())} keys")

            # If workers spawned, forward immediately
            if self.workers:
                logger.info(f"[{self.sm_id}] Forwarding context to {len(self.workers)} workers")
                forward_futures = []
                for w in self.workers:
                    try:
                        fut = w.set_global_context.remote(self.global_context)
                        forward_futures.append(fut)
                    except Exception as e:
                        logger.exception(f"[{self.sm_id}] Failed forwarding global context to worker: {e}")
                
                # Wait for all workers to confirm receipt
                try:
                    ray.get(forward_futures)
                    logger.info(f"[{self.sm_id}] All workers confirmed context receipt")
                except Exception as e:
                    logger.error(f"[{self.sm_id}] Some workers failed to receive context: {e}")
            else:
                logger.warning(f"[{self.sm_id}] No workers available yet to forward context")

            return {"status": "ok", "workers_notified": len(self.workers)}

        except Exception as e:
            logger.exception(f"[{self.sm_id}] set_global_context error: {e}")
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------------
    # Initialize extractor + workers
    # ----------------------------------------------------------
    def initialize(self):
        self.status = "ready"

        # PDF extractor
        try:
            if self.pdf_path:
                self.pdf_extractor = PDFExtractor(self.pdf_path)
        except Exception as e:
            logger.error(f"[{self.sm_id}] PDF extractor init failed {e}")

        # spawn workers
        logger.info(f"[{self.sm_id}] Spawning {self.num_workers_per_submaster} workers...")
        for i in range(self.num_workers_per_submaster):
            wid = f"{self.sm_id}-W{i+1}"
            worker = WorkerAgent.remote(
                worker_id=wid,
                llm_model=self.llm_model,
                processing_requirements=self.processing_requirements
            )
            self.workers.append(worker)
            logger.info(f"[{self.sm_id}] Created worker: {wid}")

        # initialize workers
        try:
            init_futures = [w.initialize.remote() for w in self.workers]
            ray.get(init_futures)
            logger.info(f"[{self.sm_id}] All {len(self.workers)} workers initialized")
        except Exception as e:
            logger.exception(f"[{self.sm_id}] Worker initialization error: {e}")

        # if context was set before workers existed, send it now
        if self.global_context and self.workers:
            logger.info(f"[{self.sm_id}] Forwarding pre-existing global context to newly created workers")
            forward_futures = []
            for w in self.workers:
                try:
                    fut = w.set_global_context.remote(self.global_context)
                    forward_futures.append(fut)
                except Exception as e:
                    logger.exception(f"[{self.sm_id}] Failed sending context to worker during init: {e}")
            
            try:
                ray.get(forward_futures)
                logger.info(f"[{self.sm_id}] All workers received pre-existing context")
            except Exception as e:
                logger.error(f"[{self.sm_id}] Some workers failed to receive pre-existing context: {e}")

        # if context not set earlier, try to fetch snapshot from residual
        if not self.global_context and self.residual:
            try:
                logger.info(f"[{self.sm_id}] Attempting to pull global context from ResidualAgent")
                snapshot = ray.get(self.residual.get_snapshot.remote())
                if isinstance(snapshot, dict):
                    self.global_context = snapshot
                    logger.info(f"[{self.sm_id}] Pulled global context from ResidualAgent")
                    
                    # Forward to workers
                    forward_futures = []
                    for w in self.workers:
                        try:
                            fut = w.set_global_context.remote(self.global_context)
                            forward_futures.append(fut)
                        except Exception as e:
                            logger.exception(f"[{self.sm_id}] Failed sending pulled context to worker: {e}")
                    
                    try:
                        ray.get(forward_futures)
                        logger.info(f"[{self.sm_id}] Pulled context forwarded to all workers")
                    except Exception as e:
                        logger.error(f"[{self.sm_id}] Some workers failed to receive pulled context: {e}")
            except Exception as e:
                logger.exception(f"[{self.sm_id}] Could not pull global context from residual: {e}")

        return {"sm_id": self.sm_id, "status": "ready", "workers": len(self.workers)}

    # ----------------------------------------------------------
    # Report worker allocation to residual agent
    # ----------------------------------------------------------
    def _report_worker_allocation(self, pages: List[int]):
        if not self.residual:
            return

        work_map = []
        for idx, p in enumerate(pages):
            worker_id = f"{self.sm_id}-W{idx % len(self.workers) + 1}"
            work_map.append({
                "worker_id": worker_id,
                "task_type": "summary",
                "page_range": [p, p],
                "status": "assigned"
            })

        update = {
            "submaster_id": self.sm_id,
            "section_name": ",".join(self.sections),
            "work_distribution": work_map
        }

        try:
            self.residual.update_from_submaster.remote(update, author=self.sm_id)
            logger.info(f"[{self.sm_id}] Sent worker allocation ({len(work_map)} entries)")
        except Exception as e:
            logger.exception(f"[{self.sm_id}] Failed sending worker allocation: {e}")

    # ----------------------------------------------------------
    # Main page processing
    # ----------------------------------------------------------
    def process(self):
        self.status = "running"
        start = time.time()

        # build page list
        pages = []
        for i in range(0, len(self.pages), 2):
            s, e = self.pages[i], self.pages[i + 1]
            pages.extend(range(s, e + 1))

        if not pages:
            return {"sm_id": self.sm_id, "status": "completed", "output": {"results": []}, "elapsed": 0}

        self._report_worker_allocation(pages)

        extracted = {}
        if self.pdf_extractor:
            try:
                extracted = self.pdf_extractor.extract_page_range(pages[0], pages[-1])
            except Exception as e:
                logger.exception(f"[{self.sm_id}] PDF extraction failed: {e}")

        # CRITICAL: Ensure workers have context before processing
        if self.global_context:
            logger.info(f"[{self.sm_id}] Ensuring all workers have global context before processing")
            forward_futures = []
            for w in self.workers:
                try:
                    fut = w.set_global_context.remote(self.global_context)
                    forward_futures.append(fut)
                except Exception as e:
                    logger.exception(f"[{self.sm_id}] Failed re-sending global context: {e}")
            
            try:
                results = ray.get(forward_futures)
                logger.info(f"[{self.sm_id}] All workers confirmed context before processing: {results}")
            except Exception as e:
                logger.error(f"[{self.sm_id}] Some workers failed to confirm context: {e}")
        else:
            logger.warning(f"[{self.sm_id}] WARNING: No global_context available for workers!")

        # dispatch tasks
        futures = []
        for idx, page in enumerate(pages):
            w = self.workers[idx % len(self.workers)]
            text = extracted.get(page, "")
            section = self._get_section_for_page(page)

            fut = w.process_page.remote(
                page_num=page,
                text=text,
                role=self.role,
                section_name=section
            )
            futures.append(fut)

        logger.info(f"[{self.sm_id}] Dispatched {len(futures)} page processing tasks")

        # collect
        try:
            results = ray.get(futures)
            logger.info(f"[{self.sm_id}] Collected {len(results)} results")
        except Exception as e:
            logger.error(f"[{self.sm_id}] Worker error {e}")
            results = []

        # Check if any workers actually used global context
        context_usage = sum(1 for r in results if r.get("global_context_used", False))
        logger.info(f"[{self.sm_id}] Global context usage: {context_usage}/{len(results)} pages")

        # Save individual worker results to MongoDB
        if self.mongo_worker_coll is not None:
            for r in results:
                try:
                    self.mongo_worker_coll.insert_one(r)
                except Exception as e:
                    logger.error(f"[{self.sm_id}] Failed to insert worker result to MongoDB: {e}")

        # final update to residual agent
        if self.residual:
            try:
                self.residual.update_from_submaster.remote({
                    "submaster_id": self.sm_id,
                    "section_name": ",".join(self.sections),
                    "notes": f"Completed {len(results)} pages"
                }, author=self.sm_id)
            except Exception as e:
                logger.exception(f"[{self.sm_id}] Post-update failed: {e}")

        elapsed = time.time() - start
        self.status = "completed"

        # Build aggregated result
        aggregated_result = {
            "sm_id": self.sm_id,
            "status": "completed",
            "elapsed": elapsed,
            "timestamp": time.time(),
            "output": {
                "role": self.role,
                "assigned_sections": self.sections,
                "page_range": self.pages,
                "total_pages": len(pages),
                "context_usage": f"{context_usage}/{len(results)}",
                "results": results
            }
        }

        # **FIX: Save SubMaster aggregated results to MongoDB**
        if self.mongo_sm_coll is not None:
            try:
                insert_result = self.mongo_sm_coll.insert_one(aggregated_result.copy())
                logger.info(f"[{self.sm_id}] Saved aggregated results to MongoDB with ID: {insert_result.inserted_id}")
            except Exception as e:
                logger.error(f"[{self.sm_id}] Failed to save aggregated results to MongoDB: {e}")
        else:
            logger.warning(f"[{self.sm_id}] MongoDB SubMaster collection not available, skipping save")

        return aggregated_result

    # ----------------------------------------------------------
    # Determine section name for a page
    # ----------------------------------------------------------
    def _get_section_for_page(self, page: int) -> str:
        sections = self.meta.get("sections", {})
        for name, info in sections.items():
            try:
                if info["page_start"] <= page <= info["page_end"]:
                    return name
            except Exception:
                pass
        return "Unknown"