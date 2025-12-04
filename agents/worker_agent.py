# agents/worker_agent.py
"""
WorkerAgent: Handles individual page processing tasks.
Called by SubMasters for fine-grained parallelization.
"""

import os
import time
import ray
from typing import Dict, Any, Optional
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor, analyze_page

# Import event emission (optional - graceful fallback if API not available)
try:
    from api.events import EventType, event_bus
    EVENTS_ENABLED = True
except ImportError:
    EVENTS_ENABLED = False

logger = get_logger("WorkerAgent")


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
class WorkerAgent:
    """WorkerAgent processes individual pages or small chunks."""
    
    def __init__(
        self, 
        worker_id: str,
        llm_model: str = None,
        processing_requirements: list = None,
        pipeline_id: Optional[str] = None,
        submaster_id: Optional[str] = None
    ):
        """Initialize Worker Agent."""
        self.worker_id = worker_id
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mistral-small-latest")
        self.processing_requirements = processing_requirements or []
        self.pipeline_id = pipeline_id
        self.submaster_id = submaster_id
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
        
        # Emit processing started event
        emit_event(
            EventType.WORKER_PROCESSING,
            self.pipeline_id,
            {
                "worker_id": self.worker_id,
                "submaster_id": self.submaster_id,
                "page_num": page_num,
                "section": section_name,
            },
            agent_id=self.worker_id,
            agent_type="worker",
        )
        
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
        
        # Emit completion event
        status = page_result.get("status", "success")
        event_type = EventType.WORKER_COMPLETED if status != "error" else EventType.WORKER_FAILED
        emit_event(
            event_type,
            self.pipeline_id,
            {
                "worker_id": self.worker_id,
                "submaster_id": self.submaster_id,
                "page_num": page_num,
                "status": status,
                "processing_time": page_result["processing_time"],
                "entities_count": len(page_result.get("entities", [])),
                "keywords_count": len(page_result.get("keywords", [])),
            },
            agent_id=self.worker_id,
            agent_type="worker",
        )
        
        return page_result
