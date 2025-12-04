"""
WorkerAgent: Handles individual page processing tasks.
Called by SubMasters for fine-grained parallelization.
Now supports global context from ResidualAgent.
"""

import os
import time
import ray
from typing import Dict, Any, Optional, List
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
    """
    WorkerAgent processes individual pages or small chunks.
    Supports global context from ResidualAgent.
    """
    
    def __init__(
        self, 
        worker_id: str,
        llm_model: str = None,
        processing_requirements: List[str] = None,
        pipeline_id: Optional[str] = None,
        submaster_id: Optional[str] = None,
        residual_agent: Optional[Any] = None
    ):
        """
        Initialize Worker Agent.
        
        Args:
            worker_id: Unique worker identifier
            llm_model: LLM model to use
            processing_requirements: List of processing requirements
            pipeline_id: Pipeline ID for event tracking
            submaster_id: Parent SubMaster ID
            residual_agent: ResidualAgent Ray actor handle (optional)
        """
        self.worker_id = worker_id
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mistral-small-latest")
        self.processing_requirements = processing_requirements or []
        self.pipeline_id = pipeline_id
        self.submaster_id = submaster_id
        self.residual_agent = residual_agent
        self.llm_processor = None
        
        # Global context from ResidualAgent
        self.global_context: Dict[str, Any] = {}
        
        logger.info(f"[{worker_id}] WorkerAgent initialized")
    
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
            f"[{self.worker_id}] ✅ Received global context "
            f"(v{context.get('version', 1)}, "
            f"{len(context.get('section_overview', {}).get('sections', []))} sections)"
        )
        
        return {"status": "ok", "worker_id": self.worker_id}
    
    def get_global_context(self) -> Dict[str, Any]:
        """Get current global context."""
        return self.global_context
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize LLM processor for this worker."""
        try:
            self.llm_processor = LLMProcessor(
                model=self.llm_model,
                temperature=0.3,
                max_retries=5,
                caller_id=self.worker_id
            )
            logger.info(f"[{self.worker_id}] LLM processor initialized")
            
            # Try to get global context from ResidualAgent if available
            if self.residual_agent:
                try:
                    context = ray.get(self.residual_agent.get_snapshot.remote(), timeout=5)
                    self.global_context = context
                    logger.info(f"[{self.worker_id}] Retrieved global context from ResidualAgent")
                except Exception as e:
                    logger.warning(f"[{self.worker_id}] Could not get context from ResidualAgent: {e}")
            
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
        """
        Process a single page with LLM analysis using global context.
        
        Args:
            page_num: Page number
            text: Page text content
            role: Worker role/purpose
            section_name: Section name for this page
            
        Returns:
            Page analysis result dict
        """
        start_time = time.time()
        logger.info(f"[{self.worker_id}] Processing page {page_num} (section: {section_name})")
        
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
            "worker_id": self.worker_id,
            "used_global_context": bool(self.global_context)
        }
        
        # Use LLM to analyze if available and text is not empty
        if self.llm_processor and len(text.strip()) > 50:
            try:
                logger.debug(f"[{self.worker_id}] Analyzing page {page_num} with LLM...")
                
                # Get context-aware guidance
                worker_guidance = self._get_worker_guidance(section_name)
                enhanced_requirements = self._enhance_requirements_with_context(section_name)
                
                # Build enhanced prompt with context
                analysis = self._analyze_page_with_context(
                    text=text,
                    page_num=page_num,
                    role=role,
                    section_name=section_name,
                    worker_guidance=worker_guidance,
                    enhanced_requirements=enhanced_requirements
                )
                
                # Merge analysis results
                page_result.update(analysis)
                
                logger.info(
                    f"[{self.worker_id}] ✅ Page {page_num} analyzed: "
                    f"{len(analysis.get('entities', []))} entities, "
                    f"{len(analysis.get('keywords', []))} keywords"
                )
                
                # Send update to ResidualAgent if available
                if self.residual_agent and analysis.get('status') == 'success':
                    self._send_update_to_residual(page_num, section_name, analysis)
                
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
    
    def _analyze_page_with_context(
        self,
        text: str,
        page_num: int,
        role: str,
        section_name: Optional[str],
        worker_guidance: str,
        enhanced_requirements: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze page with LLM using global context.
        
        Args:
            text: Page text
            page_num: Page number
            role: Worker role
            section_name: Section name
            worker_guidance: Guidance from global context
            enhanced_requirements: Enhanced requirements
            
        Returns:
            Analysis results dict
        """
        
        # Build enhanced prompt
        prompt_parts = [
            f"You are a {role} analyzing page {page_num}"
        ]
        
        if section_name:
            prompt_parts.append(f" from the '{section_name}' section")
        
        prompt_parts.append(" of a document.\n")
        
        # Add global context information
        if self.global_context:
            doc_context = self.global_context.get("document_context", "")
            if doc_context:
                prompt_parts.append(f"\n**Document Context:** {doc_context}\n")
            
            strategy = self.global_context.get("master_strategy", "")
            if strategy:
                prompt_parts.append(f"\n**Analysis Strategy:** {strategy}\n")
            
            reasoning = self.global_context.get("reasoning_style", "")
            if reasoning:
                prompt_parts.append(f"\n**Reasoning Style:** {reasoning}\n")
        
        # Add worker guidance
        if worker_guidance:
            prompt_parts.append(f"\n**Guidance:** {worker_guidance}\n")
        
        # Add requirements
        if enhanced_requirements:
            prompt_parts.append("\n**Requirements:**\n")
            for req in enhanced_requirements[:5]:  # Limit to 5
                prompt_parts.append(f"- {req}\n")
        
        # Add text to analyze
        text_limit = 3000
        if len(text) > text_limit:
            prompt_parts.append(f"\n**Text to analyze (excerpt):**\n{text[:text_limit]}...\n")
        else:
            prompt_parts.append(f"\n**Text to analyze:**\n{text}\n")
        
        # Add extraction instructions
        prompt_parts.append("""
Extract the following in JSON format:
{
  "summary": "Brief 2-3 sentence summary of the page content",
  "entities": ["entity1", "entity2", "entity3"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "key_points": ["point1", "point2"]
}

Focus on the most important information. Be concise and accurate.
""")
        
        prompt = "".join(prompt_parts)
        
        try:
            result = self.llm_processor.call_with_retry(prompt, parse_json=True)
            
            # Ensure all required fields exist
            result.setdefault("summary", "No summary available")
            result.setdefault("entities", [])
            result.setdefault("keywords", [])
            result.setdefault("key_points", [])
            result["status"] = "success"
            
            # Clean up results
            result["entities"] = [e for e in result["entities"] if e][:10]
            result["keywords"] = [k for k in result["keywords"] if k][:10]
            result["key_points"] = [kp for kp in result["key_points"] if kp][:5]
            
            return result
        
        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM call failed: {e}")
            return {
                "summary": f"[Analysis error: {str(e)}]",
                "entities": [],
                "keywords": [],
                "key_points": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_worker_guidance(self, section_name: Optional[str]) -> str:
        """
        Extract worker guidance from global context.
        
        Args:
            section_name: Section name to get guidance for
            
        Returns:
            Guidance string
        """
        if not self.global_context:
            return ""
        
        worker_guidance = self.global_context.get("worker_guidance", {})
        
        # Try section-specific guidance first
        if section_name and section_name in worker_guidance:
            return worker_guidance[section_name]
        
        # Try generic guidance
        if "general" in worker_guidance:
            return worker_guidance["general"]
        
        # Fallback to reasoning style
        reasoning_style = self.global_context.get("reasoning_style", "")
        if reasoning_style:
            return f"Apply {reasoning_style} reasoning"
        
        return ""
    
    def _enhance_requirements_with_context(self, section_name: Optional[str]) -> List[str]:
        """
        Enhance processing requirements using global context.
        
        Args:
            section_name: Section name
            
        Returns:
            Enhanced requirements list
        """
        enhanced = list(self.processing_requirements)
        
        if not self.global_context:
            return enhanced
        
        # Add constraints from global context
        constraints = self.global_context.get("important_constraints", [])
        for constraint in constraints[:3]:  # Limit to 3
            if constraint and constraint not in enhanced:
                enhanced.append(constraint)
        
        # Add section-specific requirements
        sections = self.global_context.get("section_overview", {}).get("sections", [])
        section_info = next((s for s in sections if s.get("name") == section_name), None)
        
        if section_info:
            # Add dependencies as context
            deps = section_info.get("dependencies", [])
            if deps:
                enhanced.append(f"Related to: {', '.join(deps[:3])}")
            
            # Add importance level
            importance = section_info.get("importance", "medium")
            if importance == "high":
                enhanced.append("High-priority section: extract detailed information")
        
        return enhanced
    
    def _send_update_to_residual(
        self,
        page_num: int,
        section_name: Optional[str],
        analysis: Dict[str, Any]
    ) -> None:
        """
        Send worker update to ResidualAgent.
        
        Args:
            page_num: Page number
            section_name: Section name
            analysis: Analysis results
        """
        if not self.residual_agent:
            return
        
        try:
            update = {
                "worker_id": self.worker_id,
                "page": page_num,
                "section": section_name,
                "status": "done",
                "entities": analysis.get("entities", [])[:5],  # Top 5
                "keywords": analysis.get("keywords", [])[:5],  # Top 5
                "summary": analysis.get("summary", "")[:150]  # Limit length
            }
            
            # Fire and forget - don't wait for response
            self.residual_agent.update_from_worker.remote(update, self.worker_id)
            
            logger.debug(f"[{self.worker_id}] Sent update to ResidualAgent for page {page_num}")
        
        except Exception as e:
            logger.debug(f"[{self.worker_id}] Failed to send update to ResidualAgent: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "worker_id": self.worker_id,
            "llm_model": self.llm_model,
            "llm_ready": self.llm_processor is not None,
            "has_global_context": bool(self.global_context),
            "context_version": self.global_context.get("version", 0),
            "submaster_id": self.submaster_id,
            "pipeline_id": self.pipeline_id,
            "has_residual_agent": self.residual_agent is not None
        }
