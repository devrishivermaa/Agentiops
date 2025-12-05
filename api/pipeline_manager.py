"""
Pipeline Manager: Manages pipeline lifecycle and integrates with the orchestrator
"""

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .events import (
    event_bus,
    EventType,
    emit_pipeline_started,
    emit_pipeline_step,
    emit_pipeline_completed,
    emit_pipeline_failed,
)
from .models import (
    PipelineStatus,
    PipelineRunCreate,
    PipelineRunFromMetadata,
    PipelineRunResponse,
    SubMasterPlan,
)

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages pipeline execution lifecycle.
    
    Provides async wrappers around the sync pipeline execution,
    tracks state, and coordinates with the EventBus.
    """
    
    def __init__(self):
        # Active pipeline runs: pipeline_id -> PipelineState
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Pending approvals: pipeline_id -> asyncio.Event
        self._approval_events: Dict[str, asyncio.Event] = {}
        
        # Thread pool for running sync pipeline code
        self._executor_threads: Dict[str, threading.Thread] = {}
        
        # Base paths
        self._output_dir = Path("output")
        self._data_dir = Path("data")
    
    def generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID"""
        return f"pipeline-{uuid.uuid4().hex[:8]}"
    
    async def start_pipeline(self, request: PipelineRunCreate) -> PipelineRunResponse:
        """
        Start a new pipeline run.
        
        This runs the pipeline in a background thread and returns immediately.
        """
        pipeline_id = self.generate_pipeline_id()
        
        # Initialize pipeline state
        self._pipelines[pipeline_id] = {
            "id": pipeline_id,
            "status": PipelineStatus.PENDING,
            "file_path": request.file_path,
            "file_name": os.path.basename(request.file_path),
            "started_at": datetime.utcnow(),
            "completed_at": None,
            "current_step": None,
            "progress_percent": 0.0,
            "num_submasters": 0,
            "num_workers": 0,
            "submaster_plan": None,
            "result": None,
            "error": None,
            "auto_approve": request.auto_approve,
            "config": {
                "user_notes": request.user_notes,
                "max_parallel_submasters": request.max_parallel_submasters,
                "num_workers_per_submaster": request.num_workers_per_submaster,
            },
        }
        
        # Create approval event for non-auto-approve pipelines
        if not request.auto_approve:
            self._approval_events[pipeline_id] = asyncio.Event()
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(pipeline_id, request),
            daemon=True,
        )
        self._executor_threads[pipeline_id] = thread
        thread.start()
        
        logger.info(f"Started pipeline {pipeline_id} for {request.file_path}")
        
        return self._get_pipeline_response(pipeline_id)
    
    async def start_pipeline_from_metadata(
        self, request: PipelineRunFromMetadata
    ) -> PipelineRunResponse:
        """Start pipeline from existing metadata file"""
        pipeline_id = self.generate_pipeline_id()
        
        # Load metadata to get file info
        try:
            with open(request.metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load metadata: {e}")
        
        self._pipelines[pipeline_id] = {
            "id": pipeline_id,
            "status": PipelineStatus.PENDING,
            "file_path": metadata.get("file_path", request.metadata_path),
            "file_name": metadata.get("file_name", os.path.basename(request.metadata_path)),
            "started_at": datetime.utcnow(),
            "completed_at": None,
            "current_step": None,
            "progress_percent": 0.0,
            "num_submasters": 0,
            "num_workers": 0,
            "submaster_plan": None,
            "result": None,
            "error": None,
            "auto_approve": request.auto_approve,
            "metadata": metadata,
        }
        
        if not request.auto_approve:
            self._approval_events[pipeline_id] = asyncio.Event()
        
        thread = threading.Thread(
            target=self._run_pipeline_from_metadata_thread,
            args=(pipeline_id, request.metadata_path, metadata, request),
            daemon=True,
        )
        self._executor_threads[pipeline_id] = thread
        thread.start()
        
        return self._get_pipeline_response(pipeline_id)
    
    def _run_pipeline_thread(self, pipeline_id: str, request: PipelineRunCreate):
        """Run the full pipeline in a background thread"""
        try:
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="mapper")
            emit_pipeline_started(pipeline_id, request.file_path, {"config": self._pipelines[pipeline_id]["config"]})
            
            # Import here to avoid circular imports
            from workflows.mapper import Mapper
            from agents.master_agent import MasterAgent
            from orchestrator import spawn_submasters_and_run
            from utils.report_generator import generate_analysis_report
            
            # Default user config for pipeline
            user_config = {
                "document_type": "research_paper",
                "processing_requirements": [
                    "summary_generation",
                    "entity_extraction",
                    "keyword_indexing"
                ],
                "user_notes": request.user_notes or "",
                "high_level_intent": getattr(request, 'high_level_intent', None) or "Analyze and summarize",
                "document_context": getattr(request, 'document_context', None) or "",
                "complexity_level": "high",
                "preferred_model": "mistral-small-latest",
                "max_parallel_submasters": request.max_parallel_submasters or 3,
                "num_workers_per_submaster": request.num_workers_per_submaster or 3,
                "feedback_required": not request.auto_approve
            }
            
            # Step 1: Mapper
            emit_pipeline_step(pipeline_id, "mapper", "started")
            mapper = Mapper(output_dir="./output")
            mapper_result = mapper.execute(request.file_path, user_config)
            
            if mapper_result.get("status") != "success":
                raise RuntimeError(f"Mapper failed: {mapper_result.get('error', mapper_result.get('errors', 'Unknown error'))}")
            
            # Load metadata from saved file
            metadata_path = mapper_result["metadata_path"]
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            emit_pipeline_step(pipeline_id, "mapper", "completed", {"pages": metadata.get("num_pages")})
            self._pipelines[pipeline_id]["metadata"] = metadata
            self._update_progress(pipeline_id, 20)
            
            # Step 2: MasterAgent generates plan using API-friendly method
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="master_planning")
            emit_pipeline_step(pipeline_id, "master_planning", "started")
            
            master = MasterAgent()
            
            # Use execute_api with intent and context from request
            high_level_intent = getattr(request, 'high_level_intent', None) or metadata.get("user_notes", "Analyze and summarize the document")
            document_context = getattr(request, 'document_context', None) or ""
            
            try:
                plan = master.execute_api(
                    metadata_path=metadata_path,
                    high_level_intent=high_level_intent,
                    user_document_context=document_context
                )
            except Exception as e:
                raise RuntimeError(f"MasterAgent failed to generate plan: {e}")
            
            if plan is None:
                raise RuntimeError("MasterAgent failed to generate plan")
            
            self._pipelines[pipeline_id]["submaster_plan"] = plan
            self._pipelines[pipeline_id]["num_submasters"] = plan.get("num_submasters", 0)
            emit_pipeline_step(pipeline_id, "master_planning", "completed", {"plan": plan})
            self._update_progress(pipeline_id, 30)
            
            # Step 3: Wait for approval if needed
            if not request.auto_approve:
                self._update_status(pipeline_id, PipelineStatus.AWAITING_APPROVAL, step="awaiting_approval")
                event_bus.emit_simple(
                    EventType.MASTER_AWAITING_FEEDBACK,
                    pipeline_id,
                    {"plan": plan},
                    agent_id="master",
                    agent_type="master",
                )
                
                # Wait for approval (blocking in thread)
                approval_event = self._approval_events.get(pipeline_id)
                if approval_event:
                    # Use a loop to check periodically (since we're in a thread)
                    while not self._check_approval(pipeline_id):
                        import time
                        time.sleep(0.5)
            
            # Step 4: Run SubMasters via orchestrator
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="orchestration")
            emit_pipeline_step(pipeline_id, "orchestration", "started")
            
            # Initialize ResidualAgent for context coordination
            import ray
            from agents.residual_agent import ResidualAgentActor
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=4)
            residual = ResidualAgentActor.remote()
            
            results = spawn_submasters_and_run(plan, metadata, residual_handle=residual)
            
            emit_pipeline_step(pipeline_id, "orchestration", "completed", {"num_results": len(results)})
            self._update_progress(pipeline_id, 80)
            
            # Step 5: Generate report
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="report_generation")
            emit_pipeline_step(pipeline_id, "report_generation", "started")
            
            report = generate_analysis_report(results, metadata)
            
            emit_pipeline_step(pipeline_id, "report_generation", "completed")
            self._update_progress(pipeline_id, 100)
            
            # Complete
            self._update_status(pipeline_id, PipelineStatus.COMPLETED)
            self._pipelines[pipeline_id]["result"] = report
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_completed(pipeline_id, report)
            
        except Exception as e:
            logger.exception(f"Pipeline {pipeline_id} failed")
            self._update_status(pipeline_id, PipelineStatus.FAILED)
            self._pipelines[pipeline_id]["error"] = str(e)
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_failed(pipeline_id, str(e), self._pipelines[pipeline_id].get("current_step"))
    
    def _run_pipeline_from_metadata_thread(
        self, pipeline_id: str, metadata_path: str, metadata: Dict, request = None
    ):
        """Run pipeline from pre-generated metadata"""
        try:
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="master_planning")
            emit_pipeline_started(pipeline_id, metadata.get("file_path", metadata_path), metadata)
            
            from agents.master_agent import MasterAgent
            from orchestrator import spawn_submasters_and_run
            from utils.report_generator import generate_analysis_report
            
            # Step 1: MasterAgent generates plan using API method
            emit_pipeline_step(pipeline_id, "master_planning", "started")
            master = MasterAgent(pipeline_id=pipeline_id)
            
            # Get intent and context from request or metadata
            high_level_intent = None
            document_context = None
            if request:
                high_level_intent = getattr(request, 'high_level_intent', None)
                document_context = getattr(request, 'document_context', None)
            
            if not high_level_intent:
                high_level_intent = metadata.get("user_notes", "Analyze and summarize the document")
            
            try:
                plan = master.execute_api(
                    metadata_path=metadata_path,
                    high_level_intent=high_level_intent,
                    user_document_context=document_context
                )
            except Exception as e:
                raise RuntimeError(f"MasterAgent failed to generate plan: {e}")
            
            if plan is None:
                raise RuntimeError("MasterAgent failed to generate plan")
            
            self._pipelines[pipeline_id]["submaster_plan"] = plan
            self._pipelines[pipeline_id]["num_submasters"] = plan.get("num_submasters", 0)
            emit_pipeline_step(pipeline_id, "master_planning", "completed", {"plan": plan})
            self._update_progress(pipeline_id, 30)
            
            # Step 2: Wait for approval if needed
            auto_approve = self._pipelines[pipeline_id].get("auto_approve", True)
            if not auto_approve:
                self._update_status(pipeline_id, PipelineStatus.AWAITING_APPROVAL, step="awaiting_approval")
                event_bus.emit_simple(
                    EventType.MASTER_AWAITING_FEEDBACK,
                    pipeline_id,
                    {"plan": plan},
                    agent_id="master",
                    agent_type="master",
                )
                
                while not self._check_approval(pipeline_id):
                    import time
                    time.sleep(0.5)
            
            # Step 3: Run SubMasters
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="orchestration")
            emit_pipeline_step(pipeline_id, "orchestration", "started")
            
            # Initialize ResidualAgent for context coordination
            import ray
            from agents.residual_agent import ResidualAgentActor
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=4)
            
            # Create ResidualAgent with pipeline_id for event tracking
            residual = ResidualAgentActor.remote(pipeline_id=pipeline_id)
            
            # Add pipeline_id to metadata for agents
            metadata["pipeline_id"] = pipeline_id
            
            results = spawn_submasters_and_run(plan, metadata, residual_handle=residual, pipeline_id=pipeline_id)
            
            emit_pipeline_step(pipeline_id, "orchestration", "completed", {"num_results": len(results)})
            self._update_progress(pipeline_id, 80)
            
            # Step 4: Generate report using Reducer
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="report_generation")
            emit_pipeline_step(pipeline_id, "report_generation", "started")
            
            # Use Reducer for aggregation
            from workflows.reducer import Reducer
            reducer = Reducer(pipeline_id=pipeline_id)
            reduced_results = reducer.reduce(list(results.values()))
            
            report = generate_analysis_report(results, metadata)
            
            emit_pipeline_step(pipeline_id, "report_generation", "completed")
            self._update_progress(pipeline_id, 100)
            
            # Complete
            self._update_status(pipeline_id, PipelineStatus.COMPLETED)
            self._pipelines[pipeline_id]["result"] = report
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_completed(pipeline_id, report)
            
        except Exception as e:
            logger.exception(f"Pipeline {pipeline_id} failed")
            self._update_status(pipeline_id, PipelineStatus.FAILED)
            self._pipelines[pipeline_id]["error"] = str(e)
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_failed(pipeline_id, str(e), self._pipelines[pipeline_id].get("current_step"))
    
    def _check_approval(self, pipeline_id: str) -> bool:
        """Check if pipeline has been approved"""
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return True
        return pipeline.get("approved", False) or pipeline.get("auto_approve", False)
    
    def approve_pipeline(self, pipeline_id: str, approved: bool, feedback: Optional[str] = None):
        """Approve or reject a pipeline's SubMaster plan"""
        if pipeline_id not in self._pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self._pipelines[pipeline_id]
        
        if pipeline["status"] != PipelineStatus.AWAITING_APPROVAL:
            raise ValueError(f"Pipeline {pipeline_id} is not awaiting approval")
        
        if approved:
            pipeline["approved"] = True
            event_bus.emit_simple(
                EventType.MASTER_PLAN_APPROVED,
                pipeline_id,
                {"feedback": feedback},
                agent_id="master",
                agent_type="master",
            )
        else:
            # Handle rejection - for now, fail the pipeline
            pipeline["status"] = PipelineStatus.FAILED
            pipeline["error"] = f"Plan rejected: {feedback or 'No feedback provided'}"
            emit_pipeline_failed(pipeline_id, pipeline["error"], "approval")
    
    def _update_status(
        self, pipeline_id: str, status: PipelineStatus, step: Optional[str] = None
    ):
        """Update pipeline status"""
        if pipeline_id in self._pipelines:
            self._pipelines[pipeline_id]["status"] = status
            if step:
                self._pipelines[pipeline_id]["current_step"] = step
    
    def _update_progress(self, pipeline_id: str, percent: float):
        """Update pipeline progress percentage"""
        if pipeline_id in self._pipelines:
            self._pipelines[pipeline_id]["progress_percent"] = percent
    
    def update_agent_counts(self, pipeline_id: str, submasters: int = 0, workers: int = 0):
        """Update agent counts for a pipeline"""
        if pipeline_id in self._pipelines:
            if submasters:
                self._pipelines[pipeline_id]["num_submasters"] = submasters
            if workers:
                self._pipelines[pipeline_id]["num_workers"] = workers
    
    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineRunResponse]:
        """Get pipeline status"""
        if pipeline_id not in self._pipelines:
            return None
        return self._get_pipeline_response(pipeline_id)
    
    def _get_pipeline_response(self, pipeline_id: str) -> PipelineRunResponse:
        """Convert internal state to response model"""
        pipeline = self._pipelines[pipeline_id]
        
        return PipelineRunResponse(
            pipeline_id=pipeline["id"],
            status=pipeline["status"],
            file_path=pipeline.get("file_path"),
            file_name=pipeline.get("file_name"),
            started_at=pipeline["started_at"],
            completed_at=pipeline.get("completed_at"),
            current_step=pipeline.get("current_step"),
            progress_percent=pipeline.get("progress_percent", 0.0),
            num_submasters=pipeline.get("num_submasters", 0),
            num_workers=pipeline.get("num_workers", 0),
            submaster_plan=pipeline.get("submaster_plan"),
            result=pipeline.get("result"),
            error=pipeline.get("error"),
        )
    
    def list_pipelines(self, status: Optional[PipelineStatus] = None) -> List[PipelineRunResponse]:
        """List all pipelines, optionally filtered by status"""
        pipelines = []
        for pipeline_id in self._pipelines:
            response = self._get_pipeline_response(pipeline_id)
            if status is None or response.status == status:
                pipelines.append(response)
        return pipelines
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline"""
        if pipeline_id not in self._pipelines:
            return False
        
        pipeline = self._pipelines[pipeline_id]
        if pipeline["status"] in [PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
            return False
        
        pipeline["status"] = PipelineStatus.CANCELLED
        pipeline["completed_at"] = datetime.utcnow()
        
        event_bus.emit_simple(
            EventType.PIPELINE_FAILED,
            pipeline_id,
            {"reason": "cancelled", "cancelled_by": "user"},
        )
        
        return True


# Global singleton instance
pipeline_manager = PipelineManager()
