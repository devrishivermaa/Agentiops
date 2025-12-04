"""
Unified Complete Pipeline: Mapper -> Reducer -> Residual -> Master Merger

This script combines both pipelines:
1. MAPPER PIPELINE: PDF Processing -> Mapper -> MasterAgent -> ResidualAgent -> SubMasters
2. REDUCER PIPELINE: Reducer SubMasters -> Residual Agent -> Master Merger

Run from project root:
    python run_unified_pipeline.py <pdf_path>
    
Or with custom settings:
    python run_unified_pipeline.py <pdf_path> --config <config.json>
    python run_unified_pipeline.py <pdf_path> --skip-mapper --skip-report
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Ensure project root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ray
from pymongo import MongoClient

# Mapper pipeline imports
from workflows.mapper import Mapper
from agents.master_agent import MasterAgent
from agents.residual_agent import ResidualAgentActor
from orchestrator import spawn_submasters_and_run
from utils.report_generator import generate_analysis_report

# Reducer pipeline imports
from reducers.reducer_submaster import run_reducer_global
try:
    from reducers.reducer_residual_agent import ResidualAgent as ReducerResidualAgent
except ImportError:
    try:
        from reducers.reducer_residual_agent import ResidualAgentActor as ReducerResidualAgent
    except ImportError:
        ReducerResidualAgent = None

try:
    from reducers.master_merger import MasterMergerAgent
except ImportError:
    try:
        from reducers.master_merger import MasterMergerActor as MasterMergerAgent
    except ImportError:
        MasterMergerAgent = None

from utils.logger import get_logger

logger = get_logger("UnifiedPipeline")

# Import for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed. PDF generation will be skipped. Install with: pip install reportlab")


class UnifiedPipeline:
    """
    Unified pipeline that runs mapper and reducer phases sequentially.
    """
    
    def __init__(self):
        self.pipeline_id = f"UNIFIED-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.start_time = None
        self.results = {
            "mapper": {},
            "reducer": {}
        }
        self.output_dir = "output"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # MongoDB setup
        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_db = os.getenv("MONGO_DB")
        
        if not self.mongo_uri or not self.mongo_db:
            raise RuntimeError("MONGO_URI and MONGO_DB environment variables must be set")
        
        logger.info(f"[{self.pipeline_id}] Unified pipeline initialized")
    
    def run(
        self,
        pdf_path: str,
        config: dict = None,
        skip_mapper: bool = False,
        skip_report: bool = False,
        skip_reducer: bool = False,
        skip_residual: bool = False,
        skip_merger: bool = False
    ) -> dict:
        """
        Run the complete unified pipeline.
        
        Args:
            pdf_path: Path to PDF file
            config: Configuration dictionary
            skip_mapper: Skip entire mapper pipeline
            skip_report: Skip report generation
            skip_reducer: Skip reducer phase
            skip_residual: Skip residual agent phase
            skip_merger: Skip master merger phase
            
        Returns:
            Complete pipeline results
        """
        self.start_time = time.time()
        
        print("\n" + "="*80)
        print(f"UNIFIED AGENTOPS PIPELINE [{self.pipeline_id}]")
        print("="*80)
        
        # Initialize Ray once for entire pipeline
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized")
        
        try:
            # ========================================
            # PART 1: MAPPER PIPELINE
            # ========================================
            
            if not skip_mapper:
                mapper_results = self._run_mapper_pipeline(pdf_path, config, skip_report)
                self.results["mapper"] = mapper_results
                
                if mapper_results.get("status") != "success":
                    logger.error("Mapper pipeline failed, aborting")
                    return self._finalize(success=False, error="Mapper pipeline failed")
            else:
                logger.info("Skipping mapper pipeline")
                self.results["mapper"] = {"status": "skipped"}
            
            # Small delay to ensure MongoDB writes are complete
            time.sleep(2)
            
            # ========================================
            # PART 2: REDUCER PIPELINE
            # ========================================
            
            logger.info("\n" + "="*80)
            logger.info("STARTING REDUCER PIPELINE")
            logger.info("="*80)
            
            reducer_results = self._run_reducer_pipeline(
                skip_reducer=skip_reducer,
                skip_residual=skip_residual,
                skip_merger=skip_merger
            )
            self.results["reducer"] = reducer_results
            
            return self._finalize(success=True)
            
        except Exception as e:
            logger.exception(f"[{self.pipeline_id}] Pipeline failed: {e}")
            return self._finalize(success=False, error=str(e))
        
        finally:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")
    
    def _run_mapper_pipeline(
        self,
        pdf_path: str,
        config: dict = None,
        skip_report: bool = False
    ) -> dict:
        """Run the complete mapper pipeline."""
        
        logger.info("\n" + "="*80)
        logger.info("PART 1: MAPPER PIPELINE")
        logger.info("="*80)
        
        # Default config
        if config is None:
            config = {
                "document_type": "research_paper",
                "processing_requirements": [
                    "summary_generation",
                    "entity_extraction",
                    "keyword_indexing"
                ],
                "user_notes": "Extract key findings, methods, and results.",
                "brief_info": "Research paper analysis",
                "complexity_level": "high",
                "priority": "high",
                "preferred_model": "mistral-small-latest",
                "max_parallel_submasters": 3,
                "num_workers_per_submaster": 4,
                "feedback_required": True
            }
        
        mapper_start = time.time()
        
        try:
            # STEP 1: MAPPER
            logger.info("\n" + "-"*80)
            logger.info("[MAPPER STEP 1] MAPPER")
            logger.info("-"*80)
            
            mapper = Mapper(output_dir="./output")
            mapper_result = mapper.execute(pdf_path, config)
            
            if mapper_result["status"] != "success":
                logger.error(f"Mapper failed: {mapper_result}")
                return {
                    "status": "failed",
                    "error": "Mapper execution failed",
                    "elapsed_time": time.time() - mapper_start
                }
            
            metadata_path = mapper_result["metadata_path"]
            logger.info(f"Metadata generated at: {metadata_path}")
            
            # STEP 2: MASTER AGENT
            logger.info("\n" + "-"*80)
            logger.info("[MAPPER STEP 2] MASTER AGENT")
            logger.info("-"*80)
            
            master = MasterAgent()
            plan = master.execute(metadata_path)
            
            if plan is None or plan.get("status") != "approved":
                logger.error("Plan not approved or not generated")
                return {
                    "status": "failed",
                    "error": "Master agent plan not approved",
                    "elapsed_time": time.time() - mapper_start
                }
            
            logger.info(f"Plan approved with {plan.get('num_submasters')} submasters")
            
            # STEP 3: RESIDUAL AGENT
            logger.info("\n" + "-"*80)
            logger.info("[MAPPER STEP 3] RESIDUAL AGENT INITIALIZATION")
            logger.info("-"*80)
            
            with open(metadata_path, "r", encoding="utf8") as f:
                metadata = json.load(f)
            
            residual = ResidualAgentActor.remote()
            logger.info("ResidualAgent actor created")
            
            # STEP 4: ORCHESTRATOR
            logger.info("\n" + "-"*80)
            logger.info("[MAPPER STEP 4] ORCHESTRATOR + SUBMASTERS")
            logger.info("-"*80)
            
            results = spawn_submasters_and_run(plan, metadata, residual_handle=residual)
            logger.info(f"Completed: {len(results)} submasters")
            
            # Print context usage summary
            for sm_id, result in results.items():
                if result.get("status") == "ok":
                    context_usage = result.get("output", {}).get("output", {}).get("context_usage", "N/A")
                    logger.info(f"  {sm_id}: {context_usage}")
            
            # STEP 5: REPORT GENERATOR
            if not skip_report:
                logger.info("\n" + "-"*80)
                logger.info("[MAPPER STEP 5] REPORT GENERATOR")
                logger.info("-"*80)
                
                files = generate_analysis_report(results, metadata, output_dir="output")
                
                logger.info("Reports generated:")
                for k, v in files.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info("Skipping report generation")
                files = {}
            
            elapsed = time.time() - mapper_start
            logger.info(f"\nMapper pipeline completed in {elapsed:.2f}s")
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "metadata_path": metadata_path,
                "num_submasters": len(results),
                "report_files": files
            }
            
        except Exception as e:
            logger.exception(f"Mapper pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - mapper_start
            }
    
    def _run_reducer_pipeline(
        self,
        skip_reducer: bool = False,
        skip_residual: bool = False,
        skip_merger: bool = False
    ) -> dict:
        """Run the complete reducer pipeline."""
        
        reducer_start = time.time()
        reducer_results = {}
        
        try:
            # PHASE 1: REDUCER SUBMASTERS
            if not skip_reducer:
                logger.info("\n" + "-"*80)
                logger.info("[REDUCER PHASE 1] REDUCER SUBMASTERS")
                logger.info("-"*80)
                
                phase_start = time.time()
                aggregated = run_reducer_global(metadata=None)
                
                if not aggregated:
                    logger.error("Reducer phase returned no results")
                    reducer_results["reducer"] = {
                        "status": "failed",
                        "error": "No results from reducer submasters"
                    }
                else:
                    elapsed = time.time() - phase_start
                    logger.info(f"Reducer phase completed in {elapsed:.2f}s")
                    reducer_results["reducer"] = {
                        "status": "success",
                        "elapsed_time": elapsed,
                        "data": aggregated
                    }
            else:
                logger.info("Skipping reducer phase, loading from MongoDB...")
                reducer_results["reducer"] = self._load_latest_reducer_results()
            
            # Check if we have reducer data to continue
            if not reducer_results.get("reducer") or reducer_results["reducer"].get("status") != "success":
                logger.error("Cannot continue without reducer results")
                return {
                    "status": "failed",
                    "error": "Reducer results not available",
                    "elapsed_time": time.time() - reducer_start,
                    "phases": reducer_results
                }
            
            # PHASE 2: RESIDUAL AGENT
            if not skip_residual:
                logger.info("\n" + "-"*80)
                logger.info("[REDUCER PHASE 2] RESIDUAL AGENT")
                logger.info("-"*80)
                
                residual_result = self._run_residual_phase(reducer_results["reducer"])
                reducer_results["residual"] = residual_result
            else:
                logger.info("Skipping residual phase, loading from MongoDB...")
                reducer_results["residual"] = self._load_latest_residual_context()
            
            # PHASE 3: MASTER MERGER
            if not skip_merger:
                logger.info("\n" + "-"*80)
                logger.info("[REDUCER PHASE 3] MASTER MERGER")
                logger.info("-"*80)
                
                merger_result = self._run_merger_phase(
                    reducer_results["reducer"],
                    reducer_results["residual"].get("global_context", {}),
                    reducer_results["residual"].get("processing_plan", {})
                )
                reducer_results["merger"] = merger_result
                
                # Generate PDF from merger results
                if merger_result.get("status") == "success":
                    pdf_path = self._generate_final_summary_pdf(merger_result.get("data", {}))
                    if pdf_path:
                        logger.info(f"Final summary PDF saved: {pdf_path}")
                        reducer_results["merger"]["pdf_path"] = pdf_path
            else:
                logger.info("Skipping merger phase")
            
            elapsed = time.time() - reducer_start
            logger.info(f"\nReducer pipeline completed in {elapsed:.2f}s")
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "phases": reducer_results
            }
            
        except Exception as e:
            logger.exception(f"Reducer pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - reducer_start,
                "phases": reducer_results
            }
    
    def _run_residual_phase(self, reducer_results: dict) -> dict:
        """Run residual agent phase."""
        start = time.time()
        
        if ReducerResidualAgent is None:
            logger.error("ReducerResidualAgent not available")
            return {
                "status": "failed",
                "error": "ReducerResidualAgent class not found",
                "elapsed_time": time.time() - start,
                "global_context": {},
                "processing_plan": {}
            }
        
        try:
            reducer_data = reducer_results.get('data', reducer_results)
            
            logger.info("Creating ReducerResidualAgent actor...")
            agent = ReducerResidualAgent.remote()
            
            logger.info("Updating context from reducer results...")
            context_future = agent.update_context_from_reducer_results.remote(reducer_data)
            global_context = ray.get(context_future)
            logger.info(f"Global context updated: {len(str(global_context))} chars")
            
            logger.info("Creating processing plan...")
            plan_future = agent.create_processing_plan.remote()
            processing_plan = ray.get(plan_future)
            logger.info(f"Processing plan created: {len(str(processing_plan))} chars")
            
            elapsed = time.time() - start
            logger.info(f"Residual agent phase completed in {elapsed:.2f}s")
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "global_context": global_context,
                "processing_plan": processing_plan
            }
            
        except Exception as e:
            logger.exception(f"Residual agent phase failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start,
                "global_context": {},
                "processing_plan": {}
            }
    
    def _run_merger_phase(
        self,
        reducer_results: dict,
        global_context: dict,
        processing_plan: dict
    ) -> dict:
        """Run master merger phase."""
        start = time.time()
        
        if MasterMergerAgent is None:
            logger.error("MasterMergerAgent not available")
            return {
                "status": "failed",
                "error": "MasterMergerAgent class not found",
                "elapsed_time": time.time() - start
            }
        
        try:
            reducer_data = reducer_results.get('data', reducer_results)
            
            logger.info("Creating MasterMergerAgent actor...")
            merger = MasterMergerAgent.remote()
            
            logger.info("Synthesizing final document...")
            result_future = merger.synthesize_final_document.remote(
                reducer_data,
                global_context,
                processing_plan
            )
            final_result = ray.get(result_future)
            logger.info(f"Final synthesis complete: {len(str(final_result))} chars")
            
            elapsed = time.time() - start
            logger.info(f"Master merger phase completed in {elapsed:.2f}s")
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "data": final_result
            }
            
        except Exception as e:
            logger.exception(f"Master merger phase failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start
            }
    
    def _load_latest_reducer_results(self) -> dict:
        """Load latest reducer results from MongoDB."""
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            coll = db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")]
            
            doc = coll.find_one(sort=[("timestamp", -1)], projection={"_id": 0})
            
            if doc:
                logger.info(f"Loaded latest reducer results from {coll.name}")
                return {"status": "success", "data": doc}
            else:
                logger.error(f"No reducer results found in {coll.name}")
                return {"status": "failed", "error": "No results found"}
                
        except Exception as e:
            logger.error(f"Failed to load reducer results: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _load_latest_residual_context(self) -> dict:
        """Load latest residual context from MongoDB."""
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            coll = db[os.getenv("MONGO_RESIDUAL_COLLECTION", "residual_memory")]
            
            doc = coll.find_one(sort=[("timestamp", -1)], projection={"_id": 0})
            
            if doc:
                logger.info(f"Loaded latest residual context from {coll.name}")
                return {
                    "status": "success",
                    "global_context": doc.get("global_context", {}),
                    "processing_plan": doc.get("processing_plan", {})
                }
            else:
                logger.warning(f"No residual context found in {coll.name}")
                return {
                    "status": "not_found",
                    "global_context": {},
                    "processing_plan": {}
                }
                
        except Exception as e:
            logger.error(f"Failed to load residual context: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "global_context": {},
                "processing_plan": {}
            }
    
    def _finalize(self, success: bool, error: str = None) -> dict:
        """Finalize and return complete pipeline results."""
        total_time = time.time() - self.start_time
        
        final_results = {
            "pipeline_id": self.pipeline_id,
            "pipeline_type": "unified",
            "success": success,
            "total_elapsed_time": total_time,
            "timestamp": time.time(),
            "results": self.results
        }
        
        if error:
            final_results["error"] = error
        
        # Log summary
        print("\n" + "="*80)
        print("UNIFIED PIPELINE SUMMARY")
        print("="*80)
        print(f"Pipeline ID: {self.pipeline_id}")
        print(f"Success: {success}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"\nMapper Pipeline: {self.results.get('mapper', {}).get('status', 'unknown')}")
        if 'mapper' in self.results and self.results['mapper'].get('status') == 'success':
            print(f"  Time: {self.results['mapper'].get('elapsed_time', 0):.2f}s")
            print(f"  SubMasters: {self.results['mapper'].get('num_submasters', 0)}")
        
        print(f"\nReducer Pipeline: {self.results.get('reducer', {}).get('status', 'unknown')}")
        if 'reducer' in self.results and self.results['reducer'].get('status') == 'success':
            print(f"  Time: {self.results['reducer'].get('elapsed_time', 0):.2f}s")
            phases = self.results['reducer'].get('phases', {})
            for phase_name, phase_data in phases.items():
                if isinstance(phase_data, dict):
                    status = phase_data.get('status', 'unknown')
                    elapsed = phase_data.get('elapsed_time', 0)
                    print(f"    {phase_name}: {status} ({elapsed:.2f}s)")
                    
                    # Print PDF path if available
                    if phase_name == 'merger' and 'pdf_path' in phase_data:
                        print(f"    PDF Report: {phase_data['pdf_path']}")
        
        print("="*80)
        
        # Save to MongoDB
        self._save_summary(final_results)
        
        return final_results
    
    def _save_summary(self, results: dict):
        """Save pipeline summary to MongoDB."""
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            coll = db.get_collection("unified_pipeline_summaries")
            
            coll.insert_one(results)
            logger.info("Unified pipeline summary saved to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {e}")
    
    def _generate_final_summary_pdf(self, merger_data: dict) -> str:
        """
        Generate a comprehensive PDF report from Master Merger results.
        
        Args:
            merger_data: Complete output from MasterMergerAgent
            
        Returns:
            Path to generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("reportlab not available. Install with: pip install reportlab")
            return None
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"final_summary_{timestamp}.pdf"
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            
            logger.info(f"Generating final summary PDF: {pdf_path}")
            
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
            
            # Custom styles
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
            pipeline_id = merger_data.get('agent_id', 'Unknown')
            timestamp_val = merger_data.get('timestamp', time.time())
            date_str = datetime.fromtimestamp(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
            
            story.append(Paragraph(f"<b>Pipeline ID:</b> {self.pipeline_id}", body_style))
            story.append(Paragraph(f"<b>Agent ID:</b> {pipeline_id}", body_style))
            story.append(Paragraph(f"<b>Generated:</b> {date_str}", body_style))
            story.append(Paragraph(f"<b>Processing Time:</b> {merger_data.get('processing_time', 0):.2f} seconds", body_style))
            
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            exec_summary = merger_data.get('executive_summary', 'No executive summary available.')
            story.append(Paragraph(exec_summary, body_style))
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
            
            # Detailed Synthesis
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
                        story.append(Paragraph(synthesis, body_style))
                        
                        # Key entities
                        entities = section.get('key_entities', [])
                        if entities:
                            entities_str = ', '.join(entities[:10])
                            story.append(Paragraph(f"<i>Key Entities: {entities_str}</i>", body_style))
                        
                        story.append(Spacer(1, 0.15*inch))
                
                # Cross-section analysis
                cross_section = detailed.get('cross_section_analysis', '')
                if cross_section:
                    story.append(PageBreak())
                    story.append(Paragraph("Cross-Section Analysis", subheading_style))
                    story.append(Paragraph(cross_section, body_style))
                    story.append(Spacer(1, 0.2*inch))
                
                # Technical deep dive
                technical = detailed.get('technical_deep_dive', '')
                if technical:
                    story.append(PageBreak())
                    story.append(Paragraph("Technical Deep Dive", subheading_style))
                    story.append(Paragraph(technical, body_style))
                    story.append(Spacer(1, 0.2*inch))
            
            # Metadata
            metadata = merger_data.get('metadata', {})
            if metadata:
                story.append(PageBreak())
                story.append(Paragraph("Key Metadata", heading_style))
                
                # Top entities
                top_entities = metadata.get('top_entities', {})
                if top_entities:
                    story.append(Paragraph("Top Entities", subheading_style))
                    entities_list = sorted(top_entities.items(), key=lambda x: x[1], reverse=True)[:20]
                    entities_text = ', '.join([f"{k} ({v})" for k, v in entities_list])
                    story.append(Paragraph(entities_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Top keywords
                top_keywords = metadata.get('top_keywords', {})
                if top_keywords:
                    story.append(Paragraph("Top Keywords", subheading_style))
                    keywords_list = sorted(top_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
                    keywords_text = ', '.join([f"{k} ({v})" for k, v in keywords_list])
                    story.append(Paragraph(keywords_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Document themes
                themes = metadata.get('document_themes', [])
                if themes:
                    story.append(Paragraph("Document Themes", subheading_style))
                    themes_text = ', '.join(themes[:15])
                    story.append(Paragraph(themes_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
            
            # Insights and Conclusions
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
                    story.append(Paragraph(conclusions, body_style))
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
            logger.info(f"PDF successfully generated: {pdf_path}")
            
            return pdf_path
            
        except Exception as e:
            logger.exception(f"Failed to generate PDF: {e}")
            return None


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="Run unified mapper + reducer pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--skip-mapper", action="store_true", help="Skip entire mapper pipeline")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    parser.add_argument("--skip-reducer", action="store_true", help="Skip reducer phase")
    parser.add_argument("--skip-residual", action="store_true", help="Skip residual agent phase")
    parser.add_argument("--skip-merger", action="store_true", help="Skip master merger phase")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check PDF exists
    if not args.skip_mapper and not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Create and run unified pipeline
    pipeline = UnifiedPipeline()
    
    results = pipeline.run(
        pdf_path=args.pdf_path,
        config=config,
        skip_mapper=args.skip_mapper,
        skip_report=args.skip_report,
        skip_reducer=args.skip_reducer,
        skip_residual=args.skip_residual,
        skip_merger=args.skip_merger
    )
    
    # Print results
    if args.verbose:
        print("\n=== COMPLETE UNIFIED PIPELINE RESULTS ===\n")
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()