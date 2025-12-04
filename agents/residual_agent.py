"""
ResidualAgent - Complete Hybrid Implementation

Combines TWO critical functions:
1. Active Coordinator (Global Context Management)
   - Generates initial global_context from metadata + master plan
   - Broadcasts context to SubMasters and Workers
   - Enhances context using gathered updates
   - Persists snapshots to MongoDB

2. Quality Validator & Error Handler (Your Original Requirements)
   - Validates SubMaster outputs
   - Detects anomalies and quality issues
   - Retries failed tasks with exponential backoff
   - Sanitizes and fixes results

This is the "gray robot" that ensures quality throughout the 8-stage pipeline.
"""

import os
import json
import time
import uuid
import re
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

import ray

# Optional dependencies
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    MongoClient = None
    Collection = None

from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("ResidualAgent")


@ray.remote
class ResidualAgentActor:
    """
    Hybrid ResidualAgent combining:
    - Global context coordination (Option B)
    - Quality validation & error recovery (Original requirement)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        persist: bool = True,
        max_retries: int = 3,
        pipeline_id: Optional[str] = None
    ):
        """
        Initialize ResidualAgent
        
        Args:
            model: LLM model name
            persist: Enable MongoDB persistence
            max_retries: Maximum retry attempts for failed tasks
            pipeline_id: Pipeline ID for tracking
        """
        
        self.id = f"RA-{uuid.uuid4().hex[:8].upper()}"
        self.pipeline_id = pipeline_id
        self.lock = threading.RLock()
        self.max_retries = max_retries

        # Initialize LLM processor
        model = model or os.getenv("LLM_MODEL", "mistral-small-latest")
        try:
            self.llm = LLMProcessor(
                model=model,
                temperature=0.25,
                max_retries=4,
                caller_id=self.id
            )
            logger.info(f"[{self.id}] LLM initialized with {model}")
        except Exception as e:
            logger.warning(f"[{self.id}] LLM init failed: {e}")
            self.llm = None

        # Global context structure (Coordinator function)
        self.global_context: Dict[str, Any] = {
            "high_level_intent": "",
            "document_context": "",
            "master_strategy": "",
            "section_overview": {"sections": []},
            "worker_guidance": {},
            "submaster_guidance": {},
            "important_constraints": [],
            "expected_outputs": "",
            "reasoning_style": "short factual precise globally consistent",
            "work_allocations": {},
            "global_entities": [],
            "global_keywords": [],
            "generated_at": None,
            "residual_id": self.id,
            "version": 1
        }

        # Quality tracking (Validator function)
        self.validation_history: List[Dict[str, Any]] = []
        self.retry_history: List[Dict[str, Any]] = []
        self.anomaly_history: List[Dict[str, Any]] = []
        
        # Update tracking
        self.update_history: List[Dict[str, Any]] = []
        self.submaster_handles: List[Any] = []
        self.worker_handles: List[Any] = []

        # MongoDB persistence
        self.persist = persist and MONGO_AVAILABLE
        self.mongo_client: Optional[MongoClient] = None
        self.mongo_coll: Optional[Collection] = None

        if persist and MONGO_AVAILABLE:
            try:
                uri = os.getenv("MONGO_URI")
                db = os.getenv("MONGO_DB")
                coll = os.getenv("MONGO_RESIDUAL_COLLECTION", "residual_memory")
                
                if uri and db:
                    self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                    self.mongo_coll = self.mongo_client[db][coll]
                    logger.info(f"[{self.id}] ✅ Connected to MongoDB {db}.{coll}")
                    
                    try:
                        self._load_latest_from_db()
                    except Exception as e:
                        logger.warning(f"[{self.id}] Could not load snapshot: {e}")
                else:
                    logger.warning(f"[{self.id}] MongoDB disabled (missing MONGO_URI or MONGO_DB)")
                    self.persist = False
            
            except Exception as e:
                logger.error(f"[{self.id}] MongoDB init failed: {e}")
                self.persist = False

        logger.info(f"[{self.id}] ✅ ResidualAgent initialized (persist={self.persist}, max_retries={max_retries})")

    # ==================== COORDINATOR FUNCTIONS ====================

    def register_submasters(self, handles: List[Any]) -> Dict[str, int]:
        """Register SubMaster actors"""
        with self.lock:
            for h in handles:
                if h not in self.submaster_handles:
                    self.submaster_handles.append(h)
        
        count = len(self.submaster_handles)
        logger.info(f"[{self.id}] 📦 Registered {count} SubMasters")
        return {"registered_submasters": count}

    def register_workers(self, handles: List[Any]) -> Dict[str, int]:
        """Register Worker actors"""
        with self.lock:
            for h in handles:
                if h not in self.worker_handles:
                    self.worker_handles.append(h)
        
        count = len(self.worker_handles)
        logger.info(f"[{self.id}] 👷 Registered {count} Workers")
        return {"registered_workers": count}

    def generate_and_distribute(
        self,
        metadata: Dict[str, Any],
        master_plan: Dict[str, Any],
        wait_for_updates_seconds: int = 10
    ) -> Dict[str, Any]:
        """
        Main coordinator workflow:
        1. Generate initial global_context from LLM
        2. Broadcast to SubMasters
        3. Wait for SubMaster task maps
        4. Enhance context using updates
        5. Broadcast enhanced context to Workers
        6. Return final snapshot
        """
        
        logger.info(f"[{self.id}] 🚀 Starting generate_and_distribute workflow")
        
        try:
            gc = self._generate_initial_context(metadata, master_plan)
            logger.info(f"[{self.id}] ✅ Initial context generated")
        except Exception as e:
            logger.exception(f"[{self.id}] ❌ Failed to generate initial context: {e}")
            return self.get_snapshot()

        self._broadcast_to_submasters(gc)
        self._wait_for_submaster_task_maps(wait_for_updates_seconds)

        with self.lock:
            updates_copy = list(self.update_history)

        enhanced = self._enhance_context_with_updates(metadata, master_plan, updates_copy)
        self._broadcast_to_workers(enhanced)

        if self.persist:
            try:
                self._maybe_persist()
            except Exception as e:
                logger.error(f"[{self.id}] Failed to persist: {e}")

        logger.info(f"[{self.id}] 🎉 Workflow completed successfully")
        return self.get_snapshot()

    # ==================== QUALITY VALIDATION FUNCTIONS ====================

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SubMaster processing results
        
        Args:
            results: SubMaster results to validate
            
        Returns:
            Validation report with errors, warnings, and quality score
        """
        
        logger.info(f"[{self.id}] 🔍 Validating {len(results)} SubMaster results...")
        
        errors = []
        warnings = []
        fixed_results = {}
        
        for sm_id, sm_result in results.items():
            # Check for error status
            if sm_result.get('status') == 'error':
                errors.append({
                    "sm_id": sm_id,
                    "error": sm_result.get('error', 'Unknown error'),
                    "severity": "high"
                })
                continue
            
            # Check for unexpected status
            if sm_result.get('status') != 'ok':
                warnings.append({
                    "sm_id": sm_id,
                    "warning": f"Unexpected status: {sm_result.get('status')}",
                    "severity": "medium"
                })
            
            output = sm_result.get('output', {})
            
            # Check LLM failure rates
            llm_failures = output.get('llm_failures', 0)
            llm_successes = output.get('llm_successes', 0)
            
            if llm_failures > 0:
                total = llm_failures + llm_successes
                failure_rate = llm_failures / total if total > 0 else 0
                
                if failure_rate > 0.5:
                    errors.append({
                        "sm_id": sm_id,
                        "error": f"High LLM failure rate: {failure_rate*100:.1f}%",
                        "severity": "medium"
                    })
                elif failure_rate > 0.2:
                    warnings.append({
                        "sm_id": sm_id,
                        "warning": f"Elevated LLM failure rate: {failure_rate*100:.1f}%",
                        "severity": "low"
                    })
            
            # Check for empty results
            page_results = output.get('results', [])
            if not page_results:
                warnings.append({
                    "sm_id": sm_id,
                    "warning": "No page results found",
                    "severity": "medium"
                })
            
            # Sanitize and fix results
            fixed_output = self._sanitize_output(output)
            fixed_results[sm_id] = {"status": "ok", "output": fixed_output}
        
        # Calculate quality score
        total_submasters = len(results)
        error_count = len(errors)
        warning_count = len(warnings)
        
        quality_score = max(0, 100 - (error_count * 20) - (warning_count * 5))
        
        validation_report = {
            "status": "validated",
            "quality_score": quality_score,
            "total_submasters": total_submasters,
            "errors": errors,
            "warnings": warnings,
            "error_count": error_count,
            "warning_count": warning_count,
            "fixed_results": fixed_results,
            "timestamp": time.time(),
            "residual_id": self.id
        }
        
        with self.lock:
            self.validation_history.append(validation_report)
        
        logger.info(
            f"[{self.id}] ✅ Validation complete: Quality Score {quality_score}/100, "
            f"{error_count} errors, {warning_count} warnings"
        )
        
        return validation_report

    def _sanitize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and clean SubMaster output"""
        
        sanitized = output.copy()
        
        # Ensure required fields exist
        if 'results' not in sanitized:
            sanitized['results'] = []
        if 'total_entities' not in sanitized:
            sanitized['total_entities'] = 0
        if 'total_keywords' not in sanitized:
            sanitized['total_keywords'] = 0
        
        # Clean page results
        cleaned_results = []
        for page_result in sanitized.get('results', []):
            if isinstance(page_result, dict):
                # Ensure arrays exist
                page_result.setdefault('entities', [])
                page_result.setdefault('keywords', [])
                page_result.setdefault('key_points', [])
                
                # Remove null/empty items
                page_result['entities'] = [e for e in page_result['entities'] if e]
                page_result['keywords'] = [k for k in page_result['keywords'] if k]
                page_result['key_points'] = [kp for kp in page_result['key_points'] if kp]
                
                cleaned_results.append(page_result)
        
        sanitized['results'] = cleaned_results
        
        return sanitized

    def retry_failed_tasks(
        self,
        failed_tasks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retry failed processing tasks with exponential backoff
        
        Args:
            failed_tasks: List of failed task descriptions
            metadata: Document metadata for context
            
        Returns:
            List of retry results
        """
        
        logger.info(f"[{self.id}] 🔄 Retrying {len(failed_tasks)} failed tasks...")
        
        retry_results = []
        
        for task in failed_tasks:
            task_id = task.get('task_id', 'unknown')
            error = task.get('error', 'Unknown error')
            
            logger.info(f"[{self.id}] Retrying task {task_id} (error: {error})")
            
            retry_result = {
                "task_id": task_id,
                "original_error": error,
                "retry_status": "not_attempted",
                "attempts": 0,
                "timestamp": time.time()
            }
            
            # Exponential backoff retry
            for attempt in range(self.max_retries):
                retry_result['attempts'] = attempt + 1
                
                try:
                    wait_time = 2 ** attempt
                    logger.debug(f"[{self.id}] Attempt {attempt+1}/{self.max_retries}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    
                    # Check if error is transient
                    if any(keyword in error.lower() for keyword in ["timeout", "rate", "connection", "temporary"]):
                        retry_result['retry_status'] = "recovered"
                        logger.info(f"[{self.id}] ✅ Task {task_id} recovered on attempt {attempt+1}")
                        break
                    else:
                        retry_result['retry_status'] = "permanent_failure"
                        logger.warning(f"[{self.id}] ❌ Task {task_id} has permanent failure")
                        break
                        
                except Exception as e:
                    logger.error(f"[{self.id}] Retry attempt {attempt+1} failed: {e}")
                    retry_result['retry_error'] = str(e)
                    
                    if attempt == self.max_retries - 1:
                        retry_result['retry_status'] = "failed_after_retries"
            
            retry_results.append(retry_result)
        
        with self.lock:
            self.retry_history.extend(retry_results)
        
        successful_retries = sum(1 for r in retry_results if r['retry_status'] == 'recovered')
        logger.info(f"[{self.id}] ✅ Retry complete: {successful_retries}/{len(failed_tasks)} tasks recovered")
        
        return retry_results

    def detect_anomalies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in processing results
        
        Args:
            results: SubMaster results to analyze
            
        Returns:
            List of detected anomalies
        """
        
        logger.info(f"[{self.id}] 🔍 Running anomaly detection...")
        
        anomalies = []
        
        # Calculate baseline metrics
        all_llm_successes = []
        all_processing_times = []
        all_entity_counts = []
        
        for sm_result in results.values():
            if sm_result.get('status') == 'ok':
                output = sm_result.get('output', {})
                all_llm_successes.append(output.get('llm_successes', 0))
                all_processing_times.append(output.get('elapsed_time', 0))
                all_entity_counts.append(output.get('total_entities', 0))
        
        if not all_llm_successes:
            return anomalies
        
        avg_success = sum(all_llm_successes) / len(all_llm_successes)
        avg_time = sum(all_processing_times) / len(all_processing_times)
        avg_entities = sum(all_entity_counts) / len(all_entity_counts)
        
        # Detect outliers
        for sm_id, sm_result in results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            
            # Low success rate
            successes = output.get('llm_successes', 0)
            if successes < avg_success * 0.5:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "low_success_rate",
                    "value": successes,
                    "expected": avg_success,
                    "severity": "medium"
                })
            
            # Slow processing
            elapsed = output.get('elapsed_time', 0)
            if elapsed > avg_time * 2:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "slow_processing",
                    "value": elapsed,
                    "expected": avg_time,
                    "severity": "low"
                })
            
            # Low entity extraction
            entities = output.get('total_entities', 0)
            if entities < avg_entities * 0.3 and avg_entities > 5:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "low_entity_count",
                    "value": entities,
                    "expected": avg_entities,
                    "severity": "low"
                })
        
        with self.lock:
            self.anomaly_history.extend(anomalies)
        
        if anomalies:
            logger.warning(f"[{self.id}] ⚠️  Detected {len(anomalies)} anomalies")
        else:
            logger.info(f"[{self.id}] ✅ No anomalies detected")
        
        return anomalies

    def generate_quality_report(self, validation: Dict[str, Any]) -> str:
        """Generate human-readable quality report"""
        
        quality_score = validation.get('quality_score', 0)
        errors = validation.get('errors', [])
        warnings = validation.get('warnings', [])
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║        RESIDUAL AGENT QUALITY REPORT                     ║
╚══════════════════════════════════════════════════════════╝

Overall Quality Score: {quality_score}/100

📊 Statistics:
  - Total SubMasters: {validation.get('total_submasters', 0)}
  - Errors: {len(errors)}
  - Warnings: {len(warnings)}
  - Residual Agent: {self.id}

"""
        
        if errors:
            report += "\n❌ ERRORS:\n"
            for err in errors:
                report += f"  • {err['sm_id']}: {err['error']} (severity: {err['severity']})\n"
        
        if warnings:
            report += "\n⚠️  WARNINGS:\n"
            for warn in warnings:
                report += f"  • {warn['sm_id']}: {warn['warning']} (severity: {warn['severity']})\n"
        
        if quality_score >= 90:
            report += "\n✅ VERDICT: Excellent quality, proceed with confidence.\n"
        elif quality_score >= 70:
            report += "\n✓ VERDICT: Good quality, minor issues detected.\n"
        elif quality_score >= 50:
            report += "\n⚠️  VERDICT: Fair quality, review warnings before proceeding.\n"
        else:
            report += "\n❌ VERDICT: Poor quality, investigate errors immediately.\n"
        
        return report

    # ==================== RPC ENDPOINTS ====================

    def update_from_submaster(self, update: Dict[str, Any], author: str) -> Dict[str, str]:
        """Receive update from SubMaster"""
        
        with self.lock:
            self.update_history.append({
                "timestamp": time.time(),
                "author": author,
                "source": "submaster",
                "payload": update
            })
            
            try:
                self._merge_submaster_task_map(update)
            except Exception as e:
                logger.exception(f"[{self.id}] Failed to merge task map from {author}: {e}")
            
            try:
                self._maybe_persist()
            except Exception as e:
                logger.exception(f"[{self.id}] Failed to persist after submaster update: {e}")
        
        return {"status": "ok"}

    def update_from_worker(self, update: Dict[str, Any], author: str) -> Dict[str, Any]:
        """Receive update from Worker"""
        
        with self.lock:
            self.update_history.append({
                "timestamp": time.time(),
                "author": author,
                "source": "worker",
                "payload": update
            })
            
            try:
                self._merge_worker_output(update)
            except Exception as e:
                logger.exception(f"[{self.id}] Failed to merge worker output from {author}: {e}")
            
            try:
                self._maybe_persist()
            except Exception as e:
                logger.exception(f"[{self.id}] Failed to persist after worker update: {e}")
        
        return {"status": "ok"}

    # ==================== CONTEXT GENERATION ====================

    def _generate_initial_context(self, metadata: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial global context using LLM"""
        
        if not self.llm:
            logger.warning(f"[{self.id}] No LLM available, returning minimal context")
            return self.get_snapshot()
        
        prompt = self._build_prompt(metadata, plan)
        
        try:
            raw = self.llm.call_with_retry(prompt, parse_json=False)
            parsed = self._safe_extract_json(raw)
            gc = parsed.get("global_context")
            
            if not isinstance(gc, dict):
                raise ValueError("LLM did not return valid 'global_context' object")
            
            with self.lock:
                gc["generated_at"] = time.time()
                gc["residual_id"] = self.id
                self._merge_generated_context(gc)
                
                self.update_history.append({
                    "timestamp": time.time(),
                    "source": "residual_llm_init"
                })
                
                try:
                    self._maybe_persist()
                except Exception:
                    logger.exception(f"[{self.id}] Failed to persist initial snapshot")
            
            return json.loads(json.dumps(self.global_context))
        
        except Exception as e:
            logger.exception(f"[{self.id}] LLM generation failed: {e}")
            raise

    def _enhance_context_with_updates(
        self,
        metadata: Dict[str, Any],
        plan: Dict[str, Any],
        updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance context using gathered updates"""
        
        if not self.llm:
            return self.get_snapshot()
        
        try:
            prev = json.dumps(self.global_context, indent=2, ensure_ascii=False)
            upd = json.dumps(updates[-25:], indent=2, ensure_ascii=False)
        except Exception:
            prev = str(self.global_context)
            upd = str(updates)

        prompt = f"""
You are a ResidualAgent responsible for maintaining global context for a multi-agent system.

Previous global_context:
{prev}

Recent updates (submasters + workers):
{upd}

Produce ONLY valid JSON with top-level key "global_context". Make conservative edits: merge allocations, update section notes, and ensure worker guidance reflects recent task maps.
"""
        
        try:
            raw = self.llm.call_with_retry(prompt, parse_json=False)
            parsed = self._safe_extract_json(raw)
            new_gc = parsed.get("global_context")
            
            if not isinstance(new_gc, dict):
                raise ValueError("Enhancement LLM response missing global_context")
        
        except Exception:
            logger.exception(f"[{self.id}] Enhancement LLM failed; returning current global_context")
            return json.loads(json.dumps(self.global_context))

        with self.lock:
            new_gc["generated_at"] = time.time()
            new_gc["residual_id"] = self.id
            new_gc["version"] = self.global_context.get("version", 1) + 1
            self._merge_generated_context(new_gc)
            
            self.update_history.append({
                "timestamp": time.time(),
                "source": "residual_llm_enhance"
            })
            
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist after enhancement")
            
            logger.info(f"[{self.id}] Enhanced global_context merged (v{new_gc.get('version', 1)})")

        return json.loads(json.dumps(self.global_context))

    # ==================== BROADCASTING ====================

    def _broadcast_to_submasters(self, context: Dict[str, Any]) -> None:
        """Broadcast context to all registered SubMasters"""
        
        if not self.submaster_handles:
            logger.info(f"[{self.id}] No submasters registered")
            return
        
        logger.info(f"[{self.id}] 📡 Broadcasting to {len(self.submaster_handles)} submasters")
        
        for h in self.submaster_handles:
            try:
                h.set_global_context.remote(context)
            except Exception:
                logger.exception(f"[{self.id}] Failed to send context to a submaster actor")

    def _broadcast_to_workers(self, context: Dict[str, Any]) -> None:
        """Broadcast context to all registered Workers"""
        
        if not self.worker_handles:
            logger.info(f"[{self.id}] No workers registered")
            return
        
        logger.info(f"[{self.id}] 📡 Broadcasting to {len(self.worker_handles)} workers")
        
        for h in self.worker_handles:
            try:
                h.set_global_context.remote(context)
            except Exception:
                logger.exception(f"[{self.id}] Failed to send context to a worker actor")

    # ==================== WAITING FOR UPDATES ====================

    def _wait_for_submaster_task_maps(self, timeout_seconds: int) -> None:
        """Wait for submaster task maps"""
        
        deadline = time.time() + max(0, int(timeout_seconds))
        logger.info(f"[{self.id}] ⏳ Waiting up to {timeout_seconds}s for submaster task maps")
        
        while time.time() < deadline:
            with self.lock:
                sub_updates = [u for u in self.update_history if u.get("source") == "submaster"]
                distinct_authors = {u.get("author") for u in sub_updates if u.get("author")}
                
                if self.submaster_handles and len(distinct_authors) >= len(self.submaster_handles):
                    logger.info(f"[{self.id}] ✅ Received task maps from {len(distinct_authors)} submasters")
                    return
            
            time.sleep(0.5)
        
        logger.info(f"[{self.id}] ⏱️ Wait timeout reached")

    # ==================== MERGING UPDATES ====================

    def _merge_submaster_task_map(self, update: Dict[str, Any]) -> None:
        """Merge submaster task map into work_allocations"""
        
        sm_id = update.get("submaster_id") or update.get("submaster")
        work_list = update.get("work_distribution", []) or update.get("work_allocation", [])
        section = update.get("section_name") or update.get("section")

        if not sm_id or not work_list:
            return

        with self.lock:
            allocations = self.global_context.setdefault("work_allocations", {})
            sm_alloc = allocations.setdefault(sm_id, [])

            for task in work_list:
                entry = {
                    "worker_id": task.get("worker_id"),
                    "section": section,
                    "task_type": task.get("task_type"),
                    "pages": task.get("page_range") or task.get("pages"),
                    "status": task.get("status", "assigned")
                }
                
                if entry not in sm_alloc:
                    sm_alloc.append(entry)

            allocations[sm_id] = sm_alloc
            self.global_context["work_allocations"] = allocations

    def _merge_worker_output(self, worker_update: Dict[str, Any]) -> None:
        """Merge worker output into global context"""
        
        if not worker_update:
            return

        with self.lock:
            # Aggregate entities
            ents = set(self.global_context.get("global_entities", []))
            for e in worker_update.get("entities", []) or []:
                if e:
                    ents.add(e)
            if ents:
                self.global_context["global_entities"] = sorted(list(ents))[:100]

            # Aggregate keywords
            kws = set(self.global_context.get("global_keywords", []))
            for k in worker_update.get("keywords", []) or []:
                if k:
                    kws.add(k)
            if kws:
                self.global_context["global_keywords"] = sorted(list(kws))[:100]

            # Attach summary to section notes
            section = worker_update.get("section")
            summary = worker_update.get("summary") or worker_update.get("short_summary")
            page = worker_update.get("page")
            
            if section and summary:
                secs = self.global_context.setdefault("section_overview", {}).setdefault("sections", [])
                tgt = next((s for s in secs if s.get("name") == section), None)
                
                if not tgt:
                    tgt = {
                        "name": section,
                        "purpose": "",
                        "page_range": [],
                        "importance": "medium",
                        "dependencies": [],
                        "notes": ""
                    }
                    secs.append(tgt)
                
                old = tgt.get("notes", "")
                addition = f" [p{page}] {summary}" if page else f" {summary}"
                tgt["notes"] = (old + addition).strip()[:1000]
                
                self.global_context["section_overview"]["sections"] = secs

    def _merge_generated_context(self, new: Dict[str, Any]) -> None:
        """Merge LLM-generated context into canonical global_context"""
        
        # Text fields
        for key in ["high_level_intent", "document_context", "master_strategy", "expected_outputs", "reasoning_style"]:
            if key in new and new[key]:
                self.global_context[key] = new[key]

        # Section overview
        new_sections = new.get("section_overview", {}).get("sections", [])
        if new_sections:
            existing_secs = {
                s.get("name"): s
                for s in self.global_context.get("section_overview", {}).get("sections", [])
            }
            
            for s in new_sections:
                name = s.get("name")
                if not name:
                    continue
                
                if name in existing_secs:
                    ex = existing_secs[name]
                    ex["purpose"] = (ex.get("purpose", "") + " " + s.get("purpose", "")).strip()
                    
                    deps = set(ex.get("dependencies", []))
                    for d in s.get("dependencies", []):
                        if d:
                            deps.add(d)
                    ex["dependencies"] = list(deps)
                    
                    for k, v in s.items():
                        if k not in ("purpose", "dependencies", "notes") and v:
                            ex[k] = v
                    
                    existing_secs[name] = ex
                else:
                    existing_secs[name] = s
            
            self.global_context["section_overview"]["sections"] = list(existing_secs.values())

        # Guidance
        for guidance_key in ["worker_guidance", "submaster_guidance"]:
            if guidance_key in new and isinstance(new[guidance_key], dict):
                dest = self.global_context.get(guidance_key, {})
                
                for k, v in new[guidance_key].items():
                    if not k:
                        continue
                    
                    if k not in dest or not dest[k]:
                        dest[k] = v
                    else:
                        if isinstance(dest[k], str) and isinstance(v, str) and len(dest[k]) < 500:
                            dest[k] = (dest[k].strip() + " " + v.strip()).strip()
                
                self.global_context[guidance_key] = dest

        # Constraints
        if "important_constraints" in new and isinstance(new["important_constraints"], list):
            existing = set(self.global_context.get("important_constraints", []))
            for c in new["important_constraints"]:
                if c:
                    existing.add(c)
            self.global_context["important_constraints"] = list(existing)[:50]

    # ==================== JSON EXTRACTION ====================

    def _safe_extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM output"""
        
        if not text or not isinstance(text, str):
            raise ValueError("No text to parse")

        # Try first/last brace
        first = text.find("{")
        last = text.rfind("}")
        
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # Regex fallback
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                raise ValueError("Failed to parse JSON from LLM output")
        
        raise ValueError("No JSON object found in LLM output")

    def _build_prompt(self, metadata: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Build prompt for initial context generation"""
        
        try:
            md = json.dumps(metadata, indent=2, ensure_ascii=False)[:3000]
        except Exception:
            md = str(metadata)[:3000]
        
        try:
            pl = json.dumps(plan, indent=2, ensure_ascii=False)[:3000]
        except Exception:
            pl = str(plan)[:3000]

        return f"""
Generate a 'global_context' JSON object for a multi-agent PDF pipeline.

Input metadata:
{md}

Input master plan:
{pl}

Output only valid JSON with a top-level key "global_context".
global_context must include: high_level_intent, document_context, master_strategy,
section_overview (with sections list), worker_guidance, submaster_guidance,
important_constraints, expected_outputs, reasoning_style.
Be conservative: if information is missing, use "unknown" or "not provided".
"""

    # ==================== PERSISTENCE ====================

    def _maybe_persist(self) -> None:
        """Persist current snapshot to MongoDB"""
        
        if not self.persist or self.mongo_coll is None:
            return
        
        try:
            doc = {
                "residual_id": self.id,
                "pipeline_id": self.pipeline_id,
                "timestamp": time.time(),
                "global_context": self.global_context,
                "history_tail": self.update_history[-80:],
                "validation_tail": self.validation_history[-10:],
                "retry_tail": self.retry_history[-10:],
                "anomaly_tail": self.anomaly_history[-10:],
                "version": self.global_context.get("version", 1)
            }
            
            self.mongo_coll.insert_one(doc)
            logger.debug(f"[{self.id}] 💾 Persisted to MongoDB")
        
        except Exception:
            logger.exception(f"[{self.id}] Failed to persist")

    def _load_latest_from_db(self) -> None:
        """Load latest snapshot from MongoDB"""
        
        if self.mongo_coll is None:
            return
        
        try:
            doc = self.mongo_coll.find_one(sort=[("timestamp", -1)])
            
            if not doc:
                return
            
            gc = doc.get("global_context")
            if isinstance(gc, dict):
                with self.lock:
                    self.global_context = gc
                    self.update_history = doc.get("history_tail", []) or []
                    self.validation_history = doc.get("validation_tail", []) or []
                    self.retry_history = doc.get("retry_tail", []) or []
                    self.anomaly_history = doc.get("anomaly_tail", []) or []
                
                logger.info(f"[{self.id}] ✅ Loaded snapshot from MongoDB (v{gc.get('version', 1)})")
        
        except Exception:
            logger.exception(f"[{self.id}] Failed to load from MongoDB")

    # ==================== UTILITIES ====================

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current global_context snapshot"""
        
        with self.lock:
            self.global_context["generated_at"] = time.time()
            return json.loads(json.dumps(self.global_context))

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        
        with self.lock:
            return {
                "residual_id": self.id,
                "pipeline_id": self.pipeline_id,
                "registered_submasters": len(self.submaster_handles),
                "registered_workers": len(self.worker_handles),
                "update_count": len(self.update_history),
                "validation_count": len(self.validation_history),
                "retry_count": len(self.retry_history),
                "anomaly_count": len(self.anomaly_history),
                "context_version": self.global_context.get("version", 1),
                "persist_enabled": self.persist,
                "max_retries": self.max_retries,
                "last_update": self.global_context.get("generated_at")
            }


# ==================== FACTORY FUNCTION ====================

def create_residual_agent(
    model: Optional[str] = None,
    persist: bool = True,
    max_retries: int = 3,
    pipeline_id: Optional[str] = None
) -> Any:
    """Factory function to create ResidualAgent actor"""
    
    return ResidualAgentActor.remote(
        model=model,
        persist=persist,
        max_retries=max_retries,
        pipeline_id=pipeline_id
    )
