"""
SubMaster coordinates Worker Agents to process document sections.
Now supports ResidualAgent for global context distribution.
"""

import time
import uuid
import ray
from typing import Dict, Any, List, Optional, Tuple
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from agents.worker_agent import WorkerAgent

# Import event emission (optional - graceful fallback if API not available)
try:
    from api.events import EventType, event_bus
    EVENTS_ENABLED = True
except ImportError:
    EVENTS_ENABLED = False

logger = get_logger("SubMaster")


def emit_event(event_type, pipeline_id, data=None, agent_id=None, agent_type=None):
    """Emit event if API layer is available."""
    if EVENTS_ENABLED and pipeline_id:
        try:
            event_bus.emit_simple(
                event_type, pipeline_id, data or {}, agent_id=agent_id, agent_type=agent_type
            )
        except Exception as e:
            logger.debug(f"Event emission failed: {e}")


@ray.remote
class SubMaster:
    """
    SubMaster coordinates Worker Agents to process document sections.
    Supports global context from ResidualAgent.
    """
    
    def __init__(
        self,
        plan_piece: Dict[str, Any],
        metadata: Dict[str, Any],
        pipeline_id: Optional[str] = None,
        residual_agent: Optional[Any] = None
    ):
        """
        Initialize SubMaster.
        
        Args:
            plan_piece: SubMaster configuration from MasterAgent
            metadata: Document metadata from Mapper
            pipeline_id: Pipeline ID for event tracking
            residual_agent: ResidualAgent Ray actor handle (optional)
        """
        self.sm_id = plan_piece.get("submaster_id", f"SM-{uuid.uuid4().hex[:6].upper()}")
        self.role = plan_piece.get("role", "generic")
        self.sections = plan_piece.get("assigned_sections", [])
        self.pages = plan_piece.get("page_range", [1, 1])
        self.meta = metadata
        self.status = "initialized"
        self.pipeline_id = pipeline_id
        self.residual_agent = residual_agent
        
        # Global context from ResidualAgent
        self.global_context: Dict[str, Any] = {}
        
        # Get PDF path from metadata
        self.pdf_path = metadata.get("file_path", "")
        if not self.pdf_path:
            logger.error(f"[{self.sm_id}] No file_path in metadata!")
        
        # Get processing requirements
        self.processing_requirements = metadata.get("processing_requirements", [])
        
        # Get LLM config
        self.llm_model = metadata.get("preferred_model", "mistral-small-latest")
        
        # Worker configuration
        self.num_workers_per_submaster = metadata.get("num_workers_per_submaster", 3)
        self.pdf_extractor = None
        self.workers: List[Any] = []
        
        logger.info(
            f"[{self.sm_id}] Initialized: role={self.role}, pages={self.pages}, "
            f"workers={self.num_workers_per_submaster}, model={self.llm_model}"
        )
    
    def set_global_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Receive global context from ResidualAgent.
        
        Args:
            context: Global context dictionary
            
        Returns:
            Status dict
        """
        self.global_context = context
        
        logger.info(
            f"[{self.sm_id}] ✅ Received global context "
            f"(v{context.get('version', 1)}, "
            f"{len(context.get('section_overview', {}).get('sections', []))} sections)"
        )
        
        # Broadcast to workers if already spawned
        if self.workers:
            for worker in self.workers:
                try:
                    worker.set_global_context.remote(context)
                except Exception as e:
                    logger.warning(f"[{self.sm_id}] Failed to send context to worker: {e}")
        
        return {"status": "ok", "sm_id": self.sm_id}
    
    def get_global_context(self) -> Dict[str, Any]:
        """Get current global context."""
        return self.global_context
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize PDF extractor and spawn Worker Agents."""
        self.status = "ready"
        
        # Try to get global context from ResidualAgent
        if self.residual_agent and not self.global_context:
            try:
                context = ray.get(self.residual_agent.get_snapshot.remote(), timeout=5)
                self.global_context = context
                logger.info(f"[{self.sm_id}] Retrieved global context from ResidualAgent")
            except Exception as e:
                logger.warning(f"[{self.sm_id}] Could not get context from ResidualAgent: {e}")
        
        # Initialize PDF extractor
        try:
            if self.pdf_path:
                self.pdf_extractor = PDFExtractor(self.pdf_path)
                logger.info(f"[{self.sm_id}] PDF extractor initialized: {self.pdf_extractor.num_pages} pages")
            else:
                logger.warning(f"[{self.sm_id}] No PDF path provided")
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to initialize PDF extractor: {e}")
            self.pdf_extractor = None
        
        # Spawn Worker Agents
        try:
            logger.info(f"[{self.sm_id}] Spawning {self.num_workers_per_submaster} workers...")
            
            for i in range(self.num_workers_per_submaster):
                worker_id = f"{self.sm_id}-W{i+1}"
                
                worker = WorkerAgent.remote(
                    worker_id=worker_id,
                    llm_model=self.llm_model,
                    processing_requirements=self.processing_requirements,
                    pipeline_id=self.pipeline_id,
                    submaster_id=self.sm_id,
                    residual_agent=self.residual_agent  # Pass ResidualAgent handle
                )
                
                self.workers.append(worker)
                logger.debug(f"[{self.sm_id}] Spawned worker: {worker_id}")
                
                # Emit worker spawn event
                emit_event(
                    EventType.WORKER_SPAWNED,
                    self.pipeline_id,
                    {"worker_id": worker_id, "submaster_id": self.sm_id},
                    agent_id=worker_id,
                    agent_type="worker",
                )
            
            # Initialize all workers
            init_results = ray.get([w.initialize.remote() for w in self.workers])
            
            success_count = sum(1 for r in init_results if r.get("status") == "ready")
            logger.info(
                f"[{self.sm_id}] Initialized {success_count}/{self.num_workers_per_submaster} workers"
            )
            
            # Send global context to workers if available
            if self.global_context:
                logger.info(f"[{self.sm_id}] Broadcasting global context to {len(self.workers)} workers")
                for worker in self.workers:
                    try:
                        worker.set_global_context.remote(self.global_context)
                    except Exception as e:
                        logger.warning(f"[{self.sm_id}] Failed to send context to worker: {e}")
            
            # Send task map to ResidualAgent
            if self.residual_agent:
                self._send_task_map_to_residual()
            
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to initialize workers: {e}")
            self.workers = []
        
        return {"sm_id": self.sm_id, "status": "ready", "num_workers": len(self.workers)}
    
    def _send_task_map_to_residual(self) -> None:
        """Send worker task allocation to ResidualAgent."""
        if not self.residual_agent:
            return
        
        try:
            # Handle multiple page ranges
            all_pages = []
            for i in range(0, len(self.pages), 2):
                start, end = self.pages[i], self.pages[i + 1]
                all_pages.extend(range(start, end + 1))
            
            # Create work distribution
            work_distribution = []
            for idx, page_num in enumerate(all_pages):
                worker_idx = idx % len(self.workers)
                worker_id = f"{self.sm_id}-W{worker_idx+1}"
                section_name = self._get_section_for_page(page_num)
                
                work_distribution.append({
                    "worker_id": worker_id,
                    "task_type": "page_analysis",
                    "page_range": [page_num, page_num],
                    "status": "assigned"
                })
            
            update = {
                "submaster_id": self.sm_id,
                "section_name": ", ".join(self.sections) if self.sections else "Multiple",
                "work_distribution": work_distribution
            }
            
            # Fire and forget
            self.residual_agent.update_from_submaster.remote(update, self.sm_id)
            
            logger.info(f"[{self.sm_id}] Sent task map to ResidualAgent ({len(work_distribution)} tasks)")
        
        except Exception as e:
            logger.warning(f"[{self.sm_id}] Failed to send task map to ResidualAgent: {e}")
    
    def process(self) -> Dict[str, Any]:
        """Process assigned pages by delegating to Worker Agents."""
        start_time = time.time()
        self.status = "running"
        output = []
        
        # Track statistics
        total_chars_extracted = 0
        total_entities = 0
        total_keywords = 0
        llm_successes = 0
        llm_failures = 0
        
        # Handle multiple page ranges
        if len(self.pages) % 2 != 0:
            logger.error(f"[{self.sm_id}] Invalid page_range: {self.pages}")
            return self._create_error_result([], "Invalid page_range format")
        
        # Collect all pages to process
        all_pages = []
        for i in range(0, len(self.pages), 2):
            start, end = self.pages[i], self.pages[i + 1]
            all_pages.extend(range(start, end + 1))
        
        logger.info(
            f"[{self.sm_id}] Processing {len(all_pages)} pages "
            f"({all_pages[0]}-{all_pages[-1]}) with {len(self.workers)} workers"
        )
        
        if not self.pdf_extractor:
            logger.error(f"[{self.sm_id}] No PDF extractor available")
            return self._create_error_result(all_pages, "No PDF extractor")
        
        if not self.workers:
            logger.error(f"[{self.sm_id}] No workers available")
            return self._create_error_result(all_pages, "No workers")
        
        # Extract text for all pages first
        try:
            logger.info(f"[{self.sm_id}] Extracting text from pages...")
            extracted_pages = self.pdf_extractor.extract_page_range(all_pages[0], all_pages[-1])
            logger.info(f"[{self.sm_id}] ✅ Extracted {len(extracted_pages)} pages")
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to extract pages: {e}")
            return self._create_error_result(all_pages, f"Extraction failed: {e}")
        
        # Distribute pages to workers using Ray futures
        page_futures: List[Tuple[int, Any]] = []
        
        for idx, page_num in enumerate(all_pages):
            text = extracted_pages.get(page_num, "")
            total_chars_extracted += len(text)
            
            # Round-robin worker assignment
            worker_idx = idx % len(self.workers)
            worker = self.workers[worker_idx]
            
            # Get section name for this page
            section_name = self._get_section_for_page(page_num)
            
            # Submit page processing task to worker
            future = worker.process_page.remote(
                page_num=page_num,
                text=text,
                role=self.role,
                section_name=section_name
            )
            
            page_futures.append((page_num, future))
        
        logger.info(f"[{self.sm_id}] Submitted {len(page_futures)} page tasks to workers")
        
        # Collect results as they complete
        completed_pages = 0
        for page_num, future in page_futures:
            try:
                page_result = ray.get(future)
                output.append(page_result)
                completed_pages += 1
                
                # Update statistics
                if page_result.get("status") in ["success", "skipped"]:
                    llm_successes += 1
                    total_entities += len(page_result.get("entities", []))
                    total_keywords += len(page_result.get("keywords", []))
                else:
                    llm_failures += 1
                
                # Emit progress event
                emit_event(
                    EventType.SUBMASTER_PROGRESS,
                    self.pipeline_id,
                    {
                        "current_page": completed_pages,
                        "total_pages": len(page_futures),
                        "progress_percent": round((completed_pages / len(page_futures)) * 100, 1),
                        "page_num": page_num,
                        "worker_id": page_result.get("worker_id"),
                    },
                    agent_id=self.sm_id,
                    agent_type="submaster",
                )
                
                logger.debug(f"[{self.sm_id}] ✅ Completed page {page_num}")
                
            except Exception as e:
                logger.error(f"[{self.sm_id}] Worker failed on page {page_num}: {e}")
                output.append({
                    "page": page_num,
                    "error": str(e),
                    "summary": f"[ERROR: Worker failed]",
                    "status": "error",
                    "entities": [],
                    "keywords": []
                })
                llm_failures += 1
        
        # Sort results by page number
        output.sort(key=lambda x: x.get("page", 0))
        
        self.status = "completed"
        elapsed = time.time() - start_time
        
        # Generate aggregate summary
        aggregate_summary = self._generate_aggregate_summary(output)
        
        logger.info(
            f"[{self.sm_id}] ✅ Completed in {elapsed:.1f}s: "
            f"{len(output)} pages, {total_chars_extracted:,} chars, "
            f"{llm_successes} successes, {llm_failures} failures"
        )
        
        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "assigned_sections": self.sections,
            "page_range": self.pages,
            "num_workers": len(self.workers),
            "results": output,
            "total_pages": len(output),
            "total_chars": total_chars_extracted,
            "total_entities": total_entities,
            "total_keywords": total_keywords,
            "llm_successes": llm_successes,
            "llm_failures": llm_failures,
            "aggregate_summary": aggregate_summary,
            "elapsed_time": elapsed,
            "used_global_context": bool(self.global_context)
        }
    
    def _get_section_for_page(self, page_num: int) -> str:
        """Determine which section a page belongs to."""
        sections = self.meta.get("sections", {})
        
        for section_name, page_info in sections.items():
            start = page_info.get("page_start", 0)
            end = page_info.get("page_end", 0)
            
            if start <= page_num <= end:
                return section_name
        
        return "Unknown"
    
    def _generate_aggregate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate an overall summary from all page results."""
        summaries = [
            r.get("summary", "") 
            for r in results 
            if r.get("summary") and not r["summary"].startswith("[")
        ]
        
        if not summaries:
            return "No analysis results available."
        
        # Combine summaries intelligently
        if len(summaries) <= 3:
            combined = " ".join(summaries)
        else:
            # Use first, middle, and last summaries
            combined = f"{summaries[0]} ... {summaries[len(summaries)//2]} ... {summaries[-1]}"
        
        if len(combined) > 600:
            combined = combined[:600] + "..."
        
        return combined
    
    def _create_error_result(self, pages: List[int], error_msg: str) -> Dict[str, Any]:
        """Create error result when processing cannot proceed."""
        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "assigned_sections": self.sections,
            "page_range": self.pages,
            "num_workers": 0,
            "results": [
                {
                    "page": p,
                    "error": error_msg,
                    "summary": f"[ERROR: {error_msg}]",
                    "status": "error",
                    "entities": [],
                    "keywords": []
                }
                for p in pages
            ],
            "total_pages": len(pages),
            "total_chars": 0,
            "total_entities": 0,
            "total_keywords": 0,
            "llm_successes": 0,
            "llm_failures": len(pages),
            "aggregate_summary": f"Processing failed: {error_msg}",
            "elapsed_time": 0,
            "used_global_context": bool(self.global_context)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get SubMaster status."""
        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "status": self.status,
            "num_workers": len(self.workers),
            "has_global_context": bool(self.global_context),
            "context_version": self.global_context.get("version", 0),
            "pipeline_id": self.pipeline_id,
            "has_residual_agent": self.residual_agent is not None
        }
