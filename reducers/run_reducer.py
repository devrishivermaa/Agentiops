"""
Complete Pipeline Orchestrator - FIXED for correct file paths

Runs the entire document processing pipeline in sequence:
1. Reducer Phase (reducer_submaster.py)
2. Residual Agent Phase (reducer_residual_agent.py)
3. Master Merger Phase (master_merger.py)

Run from project root:
    python -m reducers.run_reducer
    
Or with custom settings:
    python -m reducers.run_reducer --skip-reducer --verbose
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ray
from pymongo import MongoClient

from utils.logger import get_logger
from reducers.reducer_submaster import run_reducer_global

# Import from correct locations - files are in reducers/ folder
try:
    from reducers.reducer_residual_agent import ResidualAgent
except ImportError:
    try:
        from reducers.reducer_residual_agent import ResidualAgentActor as ResidualAgent
    except ImportError:
        ResidualAgent = None
        print("WARNING: Could not import ResidualAgent from reducers.reducer_residual_agent")

try:
    from reducers.master_merger import MasterMergerAgent
except ImportError:
    try:
        from reducers.master_merger import MasterMergerActor as MasterMergerAgent
    except ImportError:
        MasterMergerAgent = None
        print("WARNING: Could not import MasterMergerAgent from reducers.master_merger")

logger = get_logger("PipelineOrchestrator")


class PipelineOrchestrator:
    """
    Orchestrates the complete document processing pipeline.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.results = {}
        self.start_time = None
        self.pipeline_id = f"PIPELINE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # MongoDB setup with corrected environment variables
        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_db = os.getenv("MONGO_DB")
        
        if not self.mongo_uri or not self.mongo_db:
            raise RuntimeError("MONGO_URI and MONGO_DB environment variables must be set")
        
        logger.info(f"[{self.pipeline_id}] Pipeline orchestrator initialized")
        logger.info(f"[{self.pipeline_id}] MongoDB: {self.mongo_db}")
    
    def run_complete_pipeline(
        self,
        skip_reducer: bool = False,
        skip_residual: bool = False,
        skip_merger: bool = False,
        metadata: dict = None
    ) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            skip_reducer: Skip reducer phase (use existing results)
            skip_residual: Skip residual agent phase
            skip_merger: Skip master merger phase
            metadata: Optional metadata for processing
            
        Returns:
            Complete pipeline results
        """
        self.start_time = time.time()
        logger.info(f"[{self.pipeline_id}] Starting complete pipeline")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized")
        
        try:
            # Phase 1: Reducer
            if not skip_reducer:
                logger.info("\n" + "="*60)
                logger.info("PHASE 1: REDUCER SUBMASTERS")
                logger.info("="*60)
                reducer_results = self._run_reducer_phase(metadata)
                self.results['reducer'] = reducer_results
            else:
                logger.info("Skipping reducer phase, loading existing results...")
                reducer_results = self._load_latest_reducer_results()
                self.results['reducer'] = reducer_results
            
            if not reducer_results:
                logger.error("No reducer results available, aborting pipeline")
                return self._finalize_results(success=False)
            
            # Phase 2: Residual Agent
            if not skip_residual:
                logger.info("\n" + "="*60)
                logger.info("PHASE 2: RESIDUAL AGENT (GLOBAL CONTEXT)")
                logger.info("="*60)
                residual_results = self._run_residual_phase(reducer_results)
                self.results['residual'] = residual_results
            else:
                logger.info("Skipping residual phase, loading existing context...")
                residual_results = self._load_latest_residual_context()
                self.results['residual'] = residual_results
            
            # Phase 3: Master Merger
            if not skip_merger:
                logger.info("\n" + "="*60)
                logger.info("PHASE 3: MASTER MERGER (FINAL SYNTHESIS)")
                logger.info("="*60)
                merger_results = self._run_merger_phase(
                    reducer_results,
                    residual_results.get('global_context', {}),
                    residual_results.get('processing_plan', {})
                )
                self.results['merger'] = merger_results
            else:
                logger.info("Skipping merger phase")
            
            return self._finalize_results(success=True)
            
        except Exception as e:
            logger.exception(f"[{self.pipeline_id}] Pipeline failed: {e}")
            return self._finalize_results(success=False, error=str(e))
        
        finally:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")
    
    def _run_reducer_phase(self, metadata: dict = None) -> dict:
        """Run reducer submaster phase."""
        logger.info("Starting reducer phase...")
        start = time.time()
        
        try:
            reducer_results = run_reducer_global(metadata=metadata)
            
            if not reducer_results:
                logger.error("Reducer phase returned no results")
                return None
            
            elapsed = time.time() - start
            logger.info(f"Reducer phase completed in {elapsed:.2f}s")
            
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "data": reducer_results
            }
            
        except Exception as e:
            logger.exception(f"Reducer phase failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start
            }
    
    def _run_residual_phase(self, reducer_results: dict) -> dict:
        """Run residual agent phase."""
        logger.info("Starting residual agent phase...")
        start = time.time()
        
        if ResidualAgent is None:
            logger.error("ResidualAgent not available - check if reducers/reducer_residual_agent.py exists")
            return {
                "status": "failed",
                "error": "ResidualAgent class not found in reducers.reducer_residual_agent",
                "elapsed_time": time.time() - start,
                "global_context": {},
                "processing_plan": {}
            }
        
        try:
            # Get reducer data
            reducer_data = reducer_results.get('data') if isinstance(reducer_results, dict) and 'data' in reducer_results else reducer_results
            
            logger.info(f"Creating ResidualAgent actor...")
            # Create residual agent (Ray actor)
            agent = ResidualAgent.remote()
            
            # Call the methods as defined in reducer_residual_agent.py
            try:
                logger.info("Updating context from reducer results...")
                context_future = agent.update_context_from_reducer_results.remote(reducer_data)
                global_context = ray.get(context_future)
                logger.info(f"Global context updated: {len(str(global_context))} chars")
                
                # Create processing plan
                logger.info("Creating processing plan...")
                plan_future = agent.create_processing_plan.remote()
                processing_plan = ray.get(plan_future)
                logger.info(f"Processing plan created: {len(str(processing_plan))} chars")
                
            except AttributeError as ae:
                logger.error(f"Method not found on ResidualAgent: {ae}")
                logger.error("Available methods should include: update_context_from_reducer_results, create_processing_plan")
                # Fallback: try to get context directly if the actor exists
                try:
                    context_future = agent.get_context.remote()
                    global_context = ray.get(context_future)
                    processing_plan = {}
                except Exception as e2:
                    logger.error(f"Fallback get_context also failed: {e2}")
                    global_context = {}
                    processing_plan = {}
            
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
        logger.info("Starting master merger phase...")
        start = time.time()
        
        if MasterMergerAgent is None:
            logger.error("MasterMergerAgent not available - check if reducers/master_merger.py exists")
            return {
                "status": "failed",
                "error": "MasterMergerAgent class not found in reducers.master_merger",
                "elapsed_time": time.time() - start
            }
        
        try:
            # Get reducer data
            reducer_data = reducer_results.get('data') if isinstance(reducer_results, dict) and 'data' in reducer_results else reducer_results
            
            logger.info("Creating MasterMergerAgent actor...")
            # Create master merger (Ray actor)
            merger = MasterMergerAgent.remote()
            
            # Synthesize final document
            try:
                logger.info("Synthesizing final document...")
                result_future = merger.synthesize_final_document.remote(
                    reducer_data,
                    global_context,
                    processing_plan
                )
                final_result = ray.get(result_future)
                logger.info(f"Final synthesis complete: {len(str(final_result))} chars")
                
            except AttributeError as ae:
                logger.error(f"Method not found on MasterMergerAgent: {ae}")
                logger.error("Available method should be: synthesize_final_document")
                final_result = {
                    "status": "failed",
                    "error": f"Method not found: {ae}"
                }
            
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
            # Use correct collection name from .env
            coll = db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")]
            
            doc = coll.find_one(sort=[("timestamp", -1)], projection={"_id": 0})
            
            if doc:
                logger.info(f"Loaded latest reducer results from {coll.name}")
                return {"status": "success", "data": doc}
            else:
                logger.error(f"No reducer results found in {coll.name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load reducer results: {e}")
            return None
    
    def _load_latest_residual_context(self) -> dict:
        """Load latest residual context from MongoDB."""
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            # Use the LAST defined value: residual_memory (not reducer_residual_memory)
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
    
    def _finalize_results(self, success: bool, error: str = None) -> dict:
        """Finalize and return pipeline results."""
        total_time = time.time() - self.start_time
        
        final_results = {
            "pipeline_id": self.pipeline_id,
            "success": success,
            "total_elapsed_time": total_time,
            "timestamp": time.time(),
            "phases": self.results
        }
        
        if error:
            final_results["error"] = error
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Pipeline ID: {self.pipeline_id}")
        logger.info(f"Success: {success}")
        logger.info(f"Total Time: {total_time:.2f}s")
        
        for phase, result in self.results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                elapsed = result.get('elapsed_time', 0)
                error_msg = result.get('error', '')
                if error_msg:
                    logger.info(f"  {phase.upper()}: {status} ({elapsed:.2f}s) - {error_msg}")
                else:
                    logger.info(f"  {phase.upper()}: {status} ({elapsed:.2f}s)")
        
        logger.info("="*60)
        
        # Save pipeline summary to MongoDB
        self._save_pipeline_summary(final_results)
        
        return final_results
    
    def _save_pipeline_summary(self, results: dict):
        """Save pipeline summary to MongoDB."""
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            coll = db.get_collection("pipeline_summaries")
            
            coll.insert_one(results)
            logger.info("Pipeline summary saved to MongoDB (pipeline_summaries collection)")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {e}")


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="Run complete document processing pipeline")
    parser.add_argument("--skip-reducer", action="store_true", help="Skip reducer phase")
    parser.add_argument("--skip-residual", action="store_true", help="Skip residual agent phase")
    parser.add_argument("--skip-merger", action="store_true", help="Skip master merger phase")
    parser.add_argument("--metadata", type=str, help="Path to metadata JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load metadata if provided
    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        try:
            with open(args.metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {args.metadata}")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Run pipeline
    results = orchestrator.run_complete_pipeline(
        skip_reducer=args.skip_reducer,
        skip_residual=args.skip_residual,
        skip_merger=args.skip_merger,
        metadata=metadata
    )
    
    # Print final results
    if args.verbose:
        print("\n=== COMPLETE PIPELINE RESULTS ===\n")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\nPipeline completed: {results['success']}")
        print(f"Total time: {results['total_elapsed_time']:.2f}s")
        print(f"Pipeline ID: {results['pipeline_id']}")


if __name__ == "__main__":
    main()