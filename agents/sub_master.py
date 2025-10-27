# agents/sub_master.py
import time
import uuid
import ray
from typing import Dict, Any, List
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from agents.worker_agent import WorkerAgent

logger = get_logger("SubMaster")


@ray.remote
class SubMaster:
    """
    SubMaster coordinates Worker Agents to process document sections.
    """
    
    def __init__(self, plan_piece, metadata):
        self.sm_id = plan_piece.get("submaster_id", f"SM-{uuid.uuid4().hex[:6].upper()}")
        self.role = plan_piece.get("role", "generic")
        self.sections = plan_piece.get("assigned_sections", [])
        self.pages = plan_piece.get("page_range", [1, 1])
        self.meta = metadata
        self.status = "initialized"
        
        # Get PDF path from metadata
        self.pdf_path = metadata.get("file_path", "")
        if not self.pdf_path:
            logger.error(f"[{self.sm_id}] No file_path in metadata!")
        
        # Get processing requirements
        self.processing_requirements = metadata.get("processing_requirements", [])
        
        # Get LLM config from metadata
        self.llm_model = metadata.get("preferred_model", "gemini-2.0-flash-exp")
        
        # Worker configuration
        self.num_workers_per_submaster = metadata.get("num_workers_per_submaster", 3)
        self.pdf_extractor = None
        self.workers = []
        
        logger.info(
            f"[{self.sm_id}] Init: role={self.role}, pages={self.pages}, "
            f"workers={self.num_workers_per_submaster}, model={self.llm_model}"
        )

    def initialize(self):
        """Initialize PDF extractor and spawn Worker Agents."""
        self.status = "ready"
        
        # Initialize PDF extractor
        try:
            if self.pdf_path:
                self.pdf_extractor = PDFExtractor(self.pdf_path)
                logger.info(f"[{self.sm_id}] PDF extractor initialized")
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
                    processing_requirements=self.processing_requirements
                )
                
                self.workers.append(worker)
                logger.debug(f"[{self.sm_id}] Spawned worker: {worker_id}")
            
            # Initialize all workers
            init_results = ray.get([w.initialize.remote() for w in self.workers])
            
            success_count = sum(1 for r in init_results if r.get("status") == "ready")
            logger.info(
                f"[{self.sm_id}] Initialized {success_count}/{self.num_workers_per_submaster} workers"
            )
            
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to initialize workers: {e}")
            self.workers = []
        
        return {"sm_id": self.sm_id, "status": "ready", "num_workers": len(self.workers)}

    def process(self):
        """
        Process assigned pages by delegating to Worker Agents.
        Uses round-robin distribution of pages to workers.
        """
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
            raise ValueError(f"Invalid page_range for {self.sm_id}: {self.pages}")

        # Collect all pages to process
        all_pages = []
        for i in range(0, len(self.pages), 2):
            start, end = self.pages[i], self.pages[i + 1]
            all_pages.extend(range(start, end + 1))
        
        logger.info(f"[{self.sm_id}] Processing {len(all_pages)} pages with {len(self.workers)} workers")
        
        if not self.pdf_extractor:
            logger.error(f"[{self.sm_id}] No PDF extractor available")
            return self._create_error_result(all_pages)
        
        if not self.workers:
            logger.error(f"[{self.sm_id}] No workers available")
            return self._create_error_result(all_pages)
        
        # Extract text for all pages first
        try:
            logger.info(f"[{self.sm_id}] Extracting text from pages {all_pages[0]}-{all_pages[-1]}...")
            extracted_pages = self.pdf_extractor.extract_page_range(all_pages[0], all_pages[-1])
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to extract pages: {e}")
            return self._create_error_result(all_pages)
        
        # Distribute pages to workers using Ray futures
        page_futures = []
        
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
        for page_num, future in page_futures:
            try:
                page_result = ray.get(future)
                output.append(page_result)
                
                # Update statistics
                if page_result.get("status") == "success":
                    llm_successes += 1
                    total_entities += len(page_result.get("entities", []))
                    total_keywords += len(page_result.get("keywords", []))
                else:
                    llm_failures += 1
                
                logger.debug(f"[{self.sm_id}] Completed page {page_num}")
                
            except Exception as e:
                logger.error(f"[{self.sm_id}] Worker failed on page {page_num}: {e}")
                output.append({
                    "page": page_num,
                    "error": str(e),
                    "summary": f"[ERROR: Worker failed on page {page_num}]",
                    "status": "error"
                })
                llm_failures += 1
        
        # Sort results by page number
        output.sort(key=lambda x: x.get("page", 0))
        
        self.status = "completed"
        
        # Generate aggregate summary
        aggregate_summary = self._generate_aggregate_summary(output)
        
        logger.info(
            f"[{self.sm_id}] Completed: {len(output)} pages, "
            f"{total_chars_extracted:,} chars, "
            f"{llm_successes} LLM successes, {llm_failures} failures"
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
            "aggregate_summary": aggregate_summary
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
    
    def _generate_aggregate_summary(self, results: list) -> str:
        """Generate an overall summary from all page results."""
        summaries = [
            r.get("summary", "") 
            for r in results 
            if r.get("summary") and not r["summary"].startswith("[")
        ]
        
        if not summaries:
            return "No analysis results available."
        
        # Combine first few summaries
        combined = " ".join(summaries[:3])
        
        if len(combined) > 500:
            combined = combined[:500] + "..."
        
        return combined
    
    def _create_error_result(self, pages: List[int]) -> Dict[str, Any]:
        """Create error result when processing cannot proceed."""
        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "assigned_sections": self.sections,
            "page_range": self.pages,
            "num_workers": 0,
            "results": [
                {"page": p, "error": "Processing failed", "summary": "[ERROR]", "status": "error"}
                for p in pages
            ],
            "total_pages": len(pages),
            "total_chars": 0,
            "total_entities": 0,
            "total_keywords": 0,
            "llm_successes": 0,
            "llm_failures": len(pages),
            "aggregate_summary": "Processing failed"
        }
