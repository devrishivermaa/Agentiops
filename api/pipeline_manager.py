"""
Pipeline Manager: Manages pipeline lifecycle and integrates with the orchestrator
Implements the FULL unified pipeline: Mapper -> Reducer SubMasters -> Residual Agent -> Master Merger
"""

import asyncio
import json
import os
import threading
import uuid
import time
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
        """
        Run the FULL unified pipeline in a background thread.
        
        This mirrors run_unified_pipeline.py exactly:
        
        PHASE 1 - MAPPER PIPELINE:
        1. Mapper - Extract PDF content and generate metadata
        2. MasterAgent - Generate SubMaster plan
        3. ResidualAgent - Context coordinator
        4. Orchestrator - Run SubMasters + Workers
        5. Report Generator - Initial report
        
        PHASE 2 - REDUCER PIPELINE:
        6. Reducer SubMasters - Aggregate and enhance mapper results
        7. Reducer Residual Agent - Build global context from reducer results
        8. Master Merger - Create comprehensive final synthesis
        9. PDF Generation - Generate final summary PDF
        """
        try:
            # ================================================================
            # PHASE 1: MAPPER PIPELINE
            # ================================================================
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
            logger.info(f"[{pipeline_id}] MAPPER STEP 1: Starting Mapper")
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
            self._pipelines[pipeline_id]["metadata_path"] = metadata_path
            self._update_progress(pipeline_id, 10)
            
            # Step 2: MasterAgent generates plan
            logger.info(f"[{pipeline_id}] MAPPER STEP 2: MasterAgent planning")
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
            self._update_progress(pipeline_id, 15)
            
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
                while not self._check_approval(pipeline_id):
                    time.sleep(0.5)
            
            # Step 4: Initialize ResidualAgent and run SubMasters via orchestrator
            logger.info(f"[{pipeline_id}] MAPPER STEP 3-4: ResidualAgent + Orchestrator")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="orchestration")
            emit_pipeline_step(pipeline_id, "orchestration", "started")
            
            import ray
            from agents.residual_agent import ResidualAgentActor
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True, 
                    num_cpus=4
                    
                )
            residual = ResidualAgentActor.remote()
            
            results = spawn_submasters_and_run(plan, metadata, residual_handle=residual)
            
            emit_pipeline_step(pipeline_id, "orchestration", "completed", {"num_results": len(results)})
            self._update_progress(pipeline_id, 40)
            
            # Step 5: Generate initial report
            logger.info(f"[{pipeline_id}] MAPPER STEP 5: Report Generation")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="report_generation")
            emit_pipeline_step(pipeline_id, "report_generation", "started")
            
            report = generate_analysis_report(results, metadata)
            
            emit_pipeline_step(pipeline_id, "report_generation", "completed")
            self._update_progress(pipeline_id, 45)
            
            # Small delay to ensure MongoDB writes are complete
            time.sleep(2)
            
            # ================================================================
            # PHASE 2: REDUCER PIPELINE
            # ================================================================
            logger.info(f"[{pipeline_id}] Starting REDUCER PIPELINE")
            
            reducer_results = self._run_reducer_pipeline_phase(pipeline_id)
            
            if reducer_results.get("status") != "success":
                logger.warning(f"[{pipeline_id}] Reducer pipeline had issues: {reducer_results.get('error', 'Unknown')}")
            
            # Combine all results
            final_result = {
                "mapper_report": report,
                "reducer_results": reducer_results,
                "final_summary": reducer_results.get("phases", {}).get("merger", {}).get("data", {}),
                "pdf_path": reducer_results.get("phases", {}).get("merger", {}).get("pdf_path")
            }
            
            # Save final synthesis as JSON (first pipeline method)
            json_path = self._save_final_results_json(pipeline_id, final_result, metadata)
            if json_path:
                final_result["json_path"] = json_path
            
            # Complete
            self._update_status(pipeline_id, PipelineStatus.COMPLETED)
            self._pipelines[pipeline_id]["result"] = final_result
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_completed(pipeline_id, final_result)
            
            logger.info(f"[{pipeline_id}] UNIFIED PIPELINE COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            logger.exception(f"Pipeline {pipeline_id} failed")
            self._update_status(pipeline_id, PipelineStatus.FAILED)
            self._pipelines[pipeline_id]["error"] = str(e)
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_failed(pipeline_id, str(e), self._pipelines[pipeline_id].get("current_step"))
    
    def _run_reducer_pipeline_phase(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Run the complete reducer pipeline phase.
        
        Phase 1: Reducer SubMasters - Load mapper results and process
        Phase 2: Reducer Residual Agent - Build global context
        Phase 3: Master Merger - Create comprehensive synthesis
        Phase 4: PDF Generation
        """
        import ray
        
        reducer_results = {"phases": {}}
        
        try:
            # ============================================
            # REDUCER PHASE 1: REDUCER SUBMASTERS
            # ============================================
            logger.info(f"[{pipeline_id}] REDUCER PHASE 1: Reducer SubMasters")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="reducer_submasters")
            
            event_bus.emit_simple(
                EventType.REDUCER_SUBMASTER_STARTED,
                pipeline_id,
                {"phase": "reducer_submasters"},
                agent_id="reducer",
                agent_type="reducer_submaster"
            )
            
            from reducers.reducer_submaster import run_reducer_global
            
            phase_start = time.time()
            aggregated = run_reducer_global(metadata=None)
            
            if not aggregated:
                logger.error(f"[{pipeline_id}] Reducer phase returned no results")
                reducer_results["phases"]["reducer"] = {
                    "status": "failed",
                    "error": "No results from reducer submasters"
                }
                return {"status": "failed", "error": "Reducer returned no results", "phases": reducer_results["phases"]}
            
            elapsed = time.time() - phase_start
            logger.info(f"[{pipeline_id}] Reducer SubMasters completed in {elapsed:.2f}s")
            
            reducer_results["phases"]["reducer"] = {
                "status": "success",
                "elapsed_time": elapsed,
                "data": aggregated
            }
            
            event_bus.emit_simple(
                EventType.REDUCER_SUBMASTER_COMPLETED,
                pipeline_id,
                {"elapsed_time": elapsed, "num_results": len(aggregated) if isinstance(aggregated, dict) else 1},
                agent_id="reducer",
                agent_type="reducer_submaster"
            )
            
            self._update_progress(pipeline_id, 60)
            
            # ============================================
            # REDUCER PHASE 2: RESIDUAL AGENT
            # ============================================
            logger.info(f"[{pipeline_id}] REDUCER PHASE 2: Reducer Residual Agent")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="reducer_residual")
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_STARTED,
                pipeline_id,
                {"phase": "residual_agent"},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            residual_result = self._run_reducer_residual_phase(pipeline_id, reducer_results["phases"]["reducer"])
            reducer_results["phases"]["residual"] = residual_result
            
            if residual_result.get("status") != "success":
                logger.warning(f"[{pipeline_id}] Residual phase had issues: {residual_result.get('error')}")
            
            self._update_progress(pipeline_id, 75)
            
            # ============================================
            # REDUCER PHASE 3: MASTER MERGER
            # ============================================
            logger.info(f"[{pipeline_id}] REDUCER PHASE 3: Master Merger")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="master_merger")
            
            event_bus.emit_simple(
                EventType.MASTER_MERGER_STARTED,
                pipeline_id,
                {"phase": "master_merger"},
                agent_id="master_merger",
                agent_type="master_merger"
            )
            
            merger_result = self._run_master_merger_phase(
                pipeline_id,
                reducer_results["phases"]["reducer"],
                residual_result.get("global_context", {}),
                residual_result.get("processing_plan", {})
            )
            reducer_results["phases"]["merger"] = merger_result
            
            self._update_progress(pipeline_id, 90)
            
            # ============================================
            # REDUCER PHASE 4: PDF GENERATION
            # ============================================
            if merger_result.get("status") == "success" and merger_result.get("data"):
                logger.info(f"[{pipeline_id}] REDUCER PHASE 4: PDF Generation")
                self._update_status(pipeline_id, PipelineStatus.RUNNING, step="pdf_generation")
                
                event_bus.emit_simple(
                    EventType.PDF_GENERATION_STARTED,
                    pipeline_id,
                    {"phase": "pdf_generation"},
                    agent_id="pdf_generator",
                    agent_type="pdf"
                )
                
                pdf_path = self._generate_final_summary_pdf(pipeline_id, merger_result.get("data", {}))
                if pdf_path:
                    logger.info(f"[{pipeline_id}] Final summary PDF saved: {pdf_path}")
                    reducer_results["phases"]["merger"]["pdf_path"] = pdf_path
                    
                    event_bus.emit_simple(
                        EventType.PDF_GENERATION_COMPLETED,
                        pipeline_id,
                        {"pdf_path": pdf_path},
                        agent_id="pdf_generator",
                        agent_type="pdf"
                    )
            
            self._update_progress(pipeline_id, 100)
            
            return {
                "status": "success",
                "phases": reducer_results["phases"]
            }
            
        except Exception as e:
            logger.exception(f"[{pipeline_id}] Reducer pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "phases": reducer_results.get("phases", {})
            }
    
    def _run_reducer_residual_phase(self, pipeline_id: str, reducer_results: dict) -> dict:
        """Run residual agent phase to build global context."""
        import ray
        start = time.time()
        
        try:
            from reducers.reducer_residual_agent import ResidualAgent as ReducerResidualAgent
        except ImportError:
            try:
                from reducers.reducer_residual_agent import ResidualAgentActor as ReducerResidualAgent
            except ImportError:
                logger.error(f"[{pipeline_id}] ReducerResidualAgent not available")
                return {
                    "status": "failed",
                    "error": "ReducerResidualAgent class not found",
                    "elapsed_time": time.time() - start,
                    "global_context": {},
                    "processing_plan": {}
                }
        
        try:
            reducer_data = reducer_results.get('data', reducer_results)
            
            logger.info(f"[{pipeline_id}] Creating ReducerResidualAgent actor...")
            agent = ReducerResidualAgent.remote()
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_CONTEXT_UPDATING,
                pipeline_id,
                {"status": "updating_context"},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            logger.info(f"[{pipeline_id}] Updating context from reducer results...")
            context_future = agent.update_context_from_reducer_results.remote(reducer_data)
            global_context = ray.get(context_future)
            logger.info(f"[{pipeline_id}] Global context updated: {len(str(global_context))} chars")
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_CONTEXT_UPDATED,
                pipeline_id,
                {"context_size": len(str(global_context))},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_PLAN_CREATING,
                pipeline_id,
                {"status": "creating_plan"},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            logger.info(f"[{pipeline_id}] Creating processing plan...")
            plan_future = agent.create_processing_plan.remote()
            processing_plan = ray.get(plan_future)
            logger.info(f"[{pipeline_id}] Processing plan created: {len(str(processing_plan))} chars")
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_PLAN_CREATED,
                pipeline_id,
                {"plan_size": len(str(processing_plan))},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            elapsed = time.time() - start
            logger.info(f"[{pipeline_id}] Residual agent phase completed in {elapsed:.2f}s")
            
            event_bus.emit_simple(
                EventType.REDUCER_RESIDUAL_COMPLETED,
                pipeline_id,
                {"elapsed_time": elapsed},
                agent_id="reducer_residual",
                agent_type="reducer_residual"
            )
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "global_context": global_context,
                "processing_plan": processing_plan
            }
            
        except Exception as e:
            logger.exception(f"[{pipeline_id}] Residual agent phase failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start,
                "global_context": {},
                "processing_plan": {}
            }
    
    def _run_master_merger_phase(
        self,
        pipeline_id: str,
        reducer_results: dict,
        global_context: dict,
        processing_plan: dict
    ) -> dict:
        """Run master merger phase for final synthesis."""
        import ray
        start = time.time()
        
        try:
            from reducers.master_merger import MasterMergerAgent
        except ImportError:
            try:
                from reducers.master_merger import MasterMergerActor as MasterMergerAgent
            except ImportError:
                logger.error(f"[{pipeline_id}] MasterMergerAgent not available")
                return {
                    "status": "failed",
                    "error": "MasterMergerAgent class not found",
                    "elapsed_time": time.time() - start
                }
        
        try:
            reducer_data = reducer_results.get('data', reducer_results)
            
            logger.info(f"[{pipeline_id}] Creating MasterMergerAgent actor...")
            merger = MasterMergerAgent.remote()
            
            event_bus.emit_simple(
                EventType.MASTER_MERGER_SYNTHESIZING,
                pipeline_id,
                {"status": "synthesizing"},
                agent_id="master_merger",
                agent_type="master_merger"
            )
            
            logger.info(f"[{pipeline_id}] Synthesizing final document...")
            result_future = merger.synthesize_final_document.remote(
                reducer_data,
                global_context,
                processing_plan
            )
            final_result = ray.get(result_future)
            logger.info(f"[{pipeline_id}] Final synthesis complete: {len(str(final_result))} chars")
            
            elapsed = time.time() - start
            logger.info(f"[{pipeline_id}] Master merger phase completed in {elapsed:.2f}s")
            
            event_bus.emit_simple(
                EventType.MASTER_MERGER_COMPLETED,
                pipeline_id,
                {"elapsed_time": elapsed, "result_size": len(str(final_result))},
                agent_id="master_merger",
                agent_type="master_merger"
            )
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "data": final_result
            }
            
        except Exception as e:
            logger.exception(f"[{pipeline_id}] Master merger phase failed: {e}")
            
            event_bus.emit_simple(
                EventType.MASTER_MERGER_FAILED,
                pipeline_id,
                {"error": str(e)},
                agent_id="master_merger",
                agent_type="master_merger"
            )
            
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start
            }
    
    def _generate_final_summary_pdf(self, pipeline_id: str, merger_data: dict) -> Optional[str]:
        """
        Generate a comprehensive PDF report from Master Merger results.
        Mirrors the FULL PDF generation from run_unified_pipeline.py
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        except ImportError:
            logger.warning(f"[{pipeline_id}] reportlab not installed. Skipping PDF generation.")
            return None
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"final_summary_{pipeline_id}_{timestamp}.pdf"
            pdf_path = str(self._output_dir / pdf_filename)
            
            logger.info(f"[{pipeline_id}] Generating final summary PDF: {pdf_path}")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Container for PDF elements
            story = []
            
            # Styles
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor='#1a1a1a',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor='#2c3e50',
                spaceAfter=12,
                spaceBefore=12
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubHeading',
                parent=styles['Heading3'],
                fontSize=14,
                textColor='#34495e',
                spaceAfter=10,
                spaceBefore=10
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['BodyText'],
                fontSize=11,
                alignment=TA_JUSTIFY,
                spaceAfter=12
            )
            
            # Title Page
            story.append(Paragraph("Document Analysis Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Pipeline info
            agent_id = merger_data.get('agent_id', 'Unknown')
            timestamp_val = merger_data.get('timestamp', time.time())
            date_str = datetime.fromtimestamp(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
            
            story.append(Paragraph(f"<b>Pipeline ID:</b> {pipeline_id}", body_style))
            story.append(Paragraph(f"<b>Agent ID:</b> {agent_id}", body_style))
            story.append(Paragraph(f"<b>Generated:</b> {date_str}", body_style))
            story.append(Paragraph(f"<b>Processing Time:</b> {merger_data.get('processing_time', 0):.2f} seconds", body_style))
            
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            exec_summary = merger_data.get('executive_summary', 'No executive summary available.')
            story.append(Paragraph(str(exec_summary), body_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Source Statistics
            stats = merger_data.get('source_statistics', {})
            if stats:
                story.append(Paragraph("Document Statistics", heading_style))
                story.append(Paragraph(f"<b>Number of Sections:</b> {stats.get('num_reducer_submasters', 0)}", body_style))
                story.append(Paragraph(f"<b>Total Entities:</b> {stats.get('total_entities', 0)}", body_style))
                story.append(Paragraph(f"<b>Total Keywords:</b> {stats.get('total_keywords', 0)}", body_style))
                story.append(Paragraph(f"<b>Key Points:</b> {stats.get('total_key_points', 0)}", body_style))
                story.append(Paragraph(f"<b>Insights:</b> {stats.get('total_insights', 0)}", body_style))
                story.append(Spacer(1, 0.2*inch))
            
            story.append(PageBreak())
            
            # Detailed Synthesis - FULL VERSION
            detailed = merger_data.get('detailed_synthesis', {})
            if detailed:
                story.append(Paragraph("Detailed Analysis", heading_style))
                
                # Section-by-section analysis
                sections = detailed.get('sections', [])
                if sections:
                    story.append(Paragraph("Section Analysis", subheading_style))
                    for section in sections:
                        section_id = section.get('section_id', 'Unknown')
                        synthesis = section.get('synthesis', '')
                        
                        story.append(Paragraph(f"<b>{section_id}</b>", body_style))
                        story.append(Paragraph(str(synthesis), body_style))
                        
                        # Key entities
                        entities = section.get('key_entities', [])
                        if entities:
                            entities_str = ', '.join(str(e) for e in entities[:10])
                            story.append(Paragraph(f"<i>Key Entities: {entities_str}</i>", body_style))
                        
                        story.append(Spacer(1, 0.15*inch))
                
                # Cross-section analysis
                cross_section = detailed.get('cross_section_analysis', '')
                if cross_section:
                    story.append(PageBreak())
                    story.append(Paragraph("Cross-Section Analysis", subheading_style))
                    story.append(Paragraph(str(cross_section), body_style))
                    story.append(Spacer(1, 0.2*inch))
                
                # Technical deep dive
                technical = detailed.get('technical_deep_dive', '')
                if technical:
                    story.append(PageBreak())
                    story.append(Paragraph("Technical Deep Dive", subheading_style))
                    story.append(Paragraph(str(technical), body_style))
                    story.append(Spacer(1, 0.2*inch))
            
            # Metadata - FULL VERSION
            metadata = merger_data.get('metadata', {})
            if metadata:
                story.append(PageBreak())
                story.append(Paragraph("Key Metadata", heading_style))
                
                # Top entities
                top_entities = metadata.get('top_entities', {})
                if top_entities:
                    story.append(Paragraph("Top Entities", subheading_style))
                    entities_list = sorted(top_entities.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:20]
                    entities_text = ', '.join([f"{k} ({v})" for k, v in entities_list])
                    story.append(Paragraph(entities_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Top keywords
                top_keywords = metadata.get('top_keywords', {})
                if top_keywords:
                    story.append(Paragraph("Top Keywords", subheading_style))
                    keywords_list = sorted(top_keywords.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:20]
                    keywords_text = ', '.join([f"{k} ({v})" for k, v in keywords_list])
                    story.append(Paragraph(keywords_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Document themes
                themes = metadata.get('document_themes', [])
                if themes:
                    story.append(Paragraph("Document Themes", subheading_style))
                    themes_text = ', '.join(str(t) for t in themes[:15])
                    story.append(Paragraph(themes_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
            
            # Insights and Conclusions - FULL VERSION
            insights = merger_data.get('insights_and_conclusions', {})
            if insights:
                story.append(PageBreak())
                story.append(Paragraph("Insights and Conclusions", heading_style))
                
                # Key findings
                findings = insights.get('key_findings', [])
                if findings:
                    story.append(Paragraph("Key Findings", subheading_style))
                    for i, finding in enumerate(findings[:10], 1):
                        story.append(Paragraph(f"{i}. {finding}", body_style))
                    story.append(Spacer(1, 0.15*inch))
                
                # Conclusions
                conclusions = insights.get('conclusions', '')
                if conclusions:
                    story.append(Paragraph("Conclusions", subheading_style))
                    story.append(Paragraph(str(conclusions), body_style))
                    story.append(Spacer(1, 0.15*inch))
                
                # Implications
                implications = insights.get('implications', [])
                if implications:
                    story.append(Paragraph("Implications", subheading_style))
                    for i, impl in enumerate(implications[:10], 1):
                        story.append(Paragraph(f"{i}. {impl}", body_style))
                    story.append(Spacer(1, 0.15*inch))
                
                # Recommendations
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    story.append(Paragraph("Recommendations", subheading_style))
                    for i, rec in enumerate(recommendations[:10], 1):
                        story.append(Paragraph(f"{i}. {rec}", body_style))
                    story.append(Spacer(1, 0.15*inch))
                
                # Future directions
                future = insights.get('future_directions', [])
                if future:
                    story.append(Paragraph("Future Directions", subheading_style))
                    for i, direction in enumerate(future[:10], 1):
                        story.append(Paragraph(f"{i}. {direction}", body_style))
            
            # Build PDF
            doc.build(story)
            logger.info(f"[{pipeline_id}] PDF successfully generated: {pdf_path}")
            
            return pdf_path
            
        except Exception as e:
            logger.exception(f"[{pipeline_id}] Failed to generate PDF: {e}")
            return None
    
    def _save_final_results_json(self, pipeline_id: str, final_result: dict, metadata: dict) -> Optional[str]:
        """
        Save the final pipeline results as a JSON file.
        Matches the output format from run_unified_pipeline.py
        """
        import json
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            doc_name = os.path.splitext(metadata.get("file_name", "document"))[0]
            json_filename = f"{doc_name}_results_{timestamp}.json"
            json_path = str(self._output_dir / json_filename)
            
            logger.info(f"[{pipeline_id}] Saving final results JSON: {json_path}")
            
            # Build a comprehensive results object
            results_to_save = {
                "pipeline_id": pipeline_id,
                "pipeline_type": "unified",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "file_name": metadata.get("file_name", ""),
                    "file_path": metadata.get("file_path", ""),
                    "total_pages": metadata.get("total_pages", 0),
                    "user_notes": metadata.get("user_notes", ""),
                },
                "mapper_report": final_result.get("mapper_report", {}),
                "reducer_results": {
                    "status": final_result.get("reducer_results", {}).get("status", "unknown"),
                    "phases": final_result.get("reducer_results", {}).get("phases", {}),
                },
                "final_summary": final_result.get("final_summary", {}),
                "pdf_path": final_result.get("pdf_path", ""),
            }
            
            # Write to JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"[{pipeline_id}] Final results JSON saved: {json_path}")
            return json_path
            
        except Exception as e:
            logger.exception(f"[{pipeline_id}] Failed to save results JSON: {e}")
            return None
    
    def _run_pipeline_from_metadata_thread(
        self, pipeline_id: str, metadata_path: str, metadata: Dict, request = None
    ):
        """
        Run FULL unified pipeline from pre-generated metadata.
        
        Skips Mapper step but runs everything else:
        1. MasterAgent - Generate SubMaster plan
        2. ResidualAgent - Context coordinator  
        3. Orchestrator - Run SubMasters + Workers
        4. Report Generator - Initial report
        5. Reducer SubMasters - Process mapper results
        6. Reducer Residual Agent - Build global context
        7. Master Merger - Final synthesis
        8. PDF Generation
        """
        try:
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="master_planning")
            emit_pipeline_started(pipeline_id, metadata.get("file_path", metadata_path), metadata)
            
            from agents.master_agent import MasterAgent
            from orchestrator import spawn_submasters_and_run
            from utils.report_generator import generate_analysis_report
            
            # Step 1: MasterAgent generates plan using API method
            logger.info(f"[{pipeline_id}] STEP 1: MasterAgent planning")
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
            self._pipelines[pipeline_id]["metadata_path"] = metadata_path
            emit_pipeline_step(pipeline_id, "master_planning", "completed", {"plan": plan})
            self._update_progress(pipeline_id, 15)
            
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
                    time.sleep(0.5)
            
            # Step 3: Run SubMasters
            logger.info(f"[{pipeline_id}] STEP 2-3: ResidualAgent + Orchestrator")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="orchestration")
            emit_pipeline_step(pipeline_id, "orchestration", "started")
            
            import ray
            from agents.residual_agent import ResidualAgentActor
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True, 
                    num_cpus=4
                    
                )
            
            # Create ResidualAgent with pipeline_id for event tracking
            residual = ResidualAgentActor.remote(pipeline_id=pipeline_id)
            
            # Add pipeline_id to metadata for agents
            metadata["pipeline_id"] = pipeline_id
            
            results = spawn_submasters_and_run(plan, metadata, residual_handle=residual, pipeline_id=pipeline_id)
            
            emit_pipeline_step(pipeline_id, "orchestration", "completed", {"num_results": len(results)})
            self._update_progress(pipeline_id, 40)
            
            # Step 4: Generate initial report
            logger.info(f"[{pipeline_id}] STEP 4: Initial Report Generation")
            self._update_status(pipeline_id, PipelineStatus.RUNNING, step="report_generation")
            emit_pipeline_step(pipeline_id, "report_generation", "started")
            
            report = generate_analysis_report(results, metadata)
            
            emit_pipeline_step(pipeline_id, "report_generation", "completed")
            self._update_progress(pipeline_id, 45)
            
            # Small delay to ensure MongoDB writes are complete
            time.sleep(2)
            
            # ================================================================
            # PHASE 2: REDUCER PIPELINE
            # ================================================================
            logger.info(f"[{pipeline_id}] Starting REDUCER PIPELINE")
            
            reducer_results = self._run_reducer_pipeline_phase(pipeline_id)
            
            if reducer_results.get("status") != "success":
                logger.warning(f"[{pipeline_id}] Reducer pipeline had issues: {reducer_results.get('error', 'Unknown')}")
            
            # Combine all results
            final_result = {
                "mapper_report": report,
                "reducer_results": reducer_results,
                "final_summary": reducer_results.get("phases", {}).get("merger", {}).get("data", {}),
                "pdf_path": reducer_results.get("phases", {}).get("merger", {}).get("pdf_path")
            }
            
            # Save final synthesis as JSON (metadata pipeline method)
            json_path = self._save_final_results_json(pipeline_id, final_result, metadata)
            if json_path:
                final_result["json_path"] = json_path
            
            # Complete
            self._update_status(pipeline_id, PipelineStatus.COMPLETED)
            self._pipelines[pipeline_id]["result"] = final_result
            self._pipelines[pipeline_id]["completed_at"] = datetime.utcnow()
            emit_pipeline_completed(pipeline_id, final_result)
            
            logger.info(f"[{pipeline_id}] UNIFIED PIPELINE COMPLETED SUCCESSFULLY")
            
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
