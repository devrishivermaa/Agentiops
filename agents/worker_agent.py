# agents/worker_agent.py
"""
WorkerAgent: Handles individual page processing tasks.
Called by SubMasters for fine-grained parallelization.
"""

import time
import ray
from typing import Dict, Any, Optional
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor, analyze_page

logger = get_logger("WorkerAgent")

@ray.remote
class WorkerAgent:
    """WorkerAgent processes individual pages or small chunks."""
    
    def __init__(
        self, 
        worker_id: str,
        llm_model: str = "gemini-2.0-flash-exp",
        processing_requirements: list = None
    ):
        """Initialize Worker Agent."""
        self.worker_id = worker_id
        self.llm_model = llm_model
        self.processing_requirements = processing_requirements or []
        self.llm_processor = None
        
        logger.info(f"[{worker_id}] WorkerAgent initialized")
    
    def initialize(self):
        """Initialize LLM processor for this worker."""
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
            logger.error(f"[{self.worker_id}] Failed to initialize: {e}")
            return {"worker_id": self.worker_id, "status": "error", "error": str(e)}
    
    def process_page(
        self,
        page_num: int,
        text: str,
        role: str,
        section_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single page with LLM analysis."""
        start_time = time.time()
        logger.info(f"[{self.worker_id}] Processing page {page_num}")
        
        page_result = {
            "page": page_num,
            "section": section_name or "Unknown",
            "char_count": len(text),
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "worker_id": self.worker_id
        }
        
        # Use LLM to analyze if available and text is not empty
        if self.llm_processor and len(text.strip()) > 50:
            try:
                logger.debug(f"[{self.worker_id}] Analyzing page {page_num} with LLM...")
                
                analysis = analyze_page(
                    llm_processor=self.llm_processor,
                    role=role,
                    text=text,
                    page_num=page_num,
                    section_name=section_name,
                    processing_requirements=self.processing_requirements
                )
                
                # Merge analysis results
                page_result.update(analysis)
                
                logger.info(
                    f"[{self.worker_id}] ✅ Page {page_num} analyzed: "
                    f"{len(analysis.get('entities', []))} entities, "
                    f"{len(analysis.get('keywords', []))} keywords"
                )
                
            except Exception as e:
                logger.error(f"[{self.worker_id}] LLM analysis failed for page {page_num}: {e}")
                page_result["llm_error"] = str(e)
                page_result["summary"] = "[LLM analysis failed - text extracted only]"
                page_result["status"] = "error"
                page_result["entities"] = []
                page_result["keywords"] = []
        else:
            # No LLM processing
            page_result["summary"] = "[Text too short for analysis]"
            page_result["status"] = "skipped"
            page_result["entities"] = []
            page_result["keywords"] = []
            
            if not self.llm_processor:
                logger.warning(f"[{self.worker_id}] No LLM processor available")
        
        page_result["processing_time"] = time.time() - start_time
        
        return page_result
