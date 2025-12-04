"""
COMPLETE PIPELINE: Mapper → MasterAgent → SubMasters → Reducer → Merger → Reports
Fully implements Architecture 1 with all stages.
"""

import os
import sys
import json
import time
from datetime import datetime
from config import Config
from workflows.mapper import Mapper
from workflows.reducer import Reducer
from agents.master_agent import MasterAgent
from agents.residual_agent import ResidualAgent
from orchestrator import spawn_submasters_and_run
from merger.merge_supervisor import MergeSupervisor
from utils.report_generator import generate_analysis_report
from utils.logger import get_logger

# Optional: Vector DB integration
if Config.ENABLE_VECTOR_DB:
    try:
        from services.vector_store_chroma import VectorStoreChroma, create_chunks_from_pages
        VECTOR_DB_AVAILABLE = True
    except ImportError:
        VECTOR_DB_AVAILABLE = False
        print("⚠️  Vector DB disabled (ChromaDB not installed)")
else:
    VECTOR_DB_AVAILABLE = False

logger = get_logger("CompletePipeline")


def run_complete_pipeline(pdf_path: str, config: dict = None, pipeline_id: str = None):
    """
    Run the COMPLETE document processing pipeline with all stages.
    
    Pipeline Stages:
    1. Mapper: Extract metadata and validate PDF
    2. MasterAgent: Generate SubMaster execution plan
    3. Orchestrator: Execute SubMasters in parallel (Mapper Stage)
    4. [Optional] Vector DB: Store embeddings for semantic search
    5. Reducer: Aggregate and consolidate results
    6. [Optional] Residual Agent: Validate and fix errors
    7. Merger: Combine Mapper + Reducer outputs
    8. Report Generator: Create final PDF + JSON reports
    
    Args:
        pdf_path: Path to PDF file
        config: Optional user configuration
        pipeline_id: Optional pipeline ID for tracking
    """
    pipeline_start = time.time()
    
    print("\n" + "=" * 80)
    print("🚀 AGENTOPS COMPLETE PIPELINE (Architecture 1)")
    print("=" * 80)
    print(f"📄 Input: {pdf_path}")
    print(f"🆔 Pipeline ID: {pipeline_id or 'N/A'}")
    print("=" * 80)
    
    # Use config or defaults
    if config is None:
        config = Config.get_user_config()
    
    # ========================================================================
    # STAGE 1: MAPPER - Validate PDF and Extract Metadata
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 [STAGE 1/8] MAPPER: Validating PDF and extracting metadata...")
    print("=" * 80)
    
    mapper = Mapper(output_dir=Config.OUTPUT_DIR)
    
    try:
        mapper_result = mapper.execute(pdf_path, config)
        
        if mapper_result["status"] != "success":
            print(f"\n❌ Mapper failed: {mapper_result}")
            return 1
        
        metadata_path = mapper_result["metadata_path"]
        print(f"\n✅ Metadata generated: {metadata_path}")
        print(f"   📊 Pages: {mapper_result['num_pages']}, Sections: {mapper_result['num_sections']}")
        
    except Exception as e:
        print(f"\n❌ Mapper failed: {e}")
        logger.exception("Mapper stage failed")
        return 1
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # ========================================================================
    # STAGE 2: MASTER AGENT - Generate SubMaster Execution Plan
    # ========================================================================
    print("\n" + "=" * 80)
    print("🤖 [STAGE 2/8] MASTER AGENT: Generating SubMaster execution plan...")
    print("=" * 80)
    
    try:
        master_agent = MasterAgent()
        plan = master_agent.execute(metadata_path)
        
        if plan is None or plan.get("status") != "approved":
            print("\n❌ Plan generation failed or not approved")
            return 1
        
        print(f"\n✅ Plan approved: {plan.get('num_submasters')} SubMasters")
        
    except Exception as e:
        print(f"\n❌ Master Agent failed: {e}")
        logger.exception("MasterAgent stage failed")
        return 1
    
    # ========================================================================
    # STAGE 3: ORCHESTRATOR - Execute SubMasters (Mapper Stage)
    # ========================================================================
    print("\n" + "=" * 80)
    print("⚙️  [STAGE 3/8] ORCHESTRATOR: Executing SubMasters in parallel...")
    print("=" * 80)
    
    try:
        mapper_results = spawn_submasters_and_run(plan, metadata, pipeline_id)
        
        success_count = sum(1 for r in mapper_results.values() if r.get('status') == 'ok')
        print(f"\n✅ SubMaster execution completed: {success_count}/{len(mapper_results)} succeeded")
        
    except Exception as e:
        print(f"\n❌ SubMaster execution failed: {e}")
        logger.exception("Orchestrator stage failed")
        return 1
    
    # ========================================================================
    # STAGE 4: VECTOR DB (Optional) - Store Embeddings
    # ========================================================================
    if VECTOR_DB_AVAILABLE and Config.ENABLE_VECTOR_DB:
        print("\n" + "=" * 80)
        print("🔍 [STAGE 4/8] VECTOR DB: Storing document embeddings...")
        print("=" * 80)
        
        try:
            vector_store = VectorStoreChroma(
                collection_name=Config.CHROMA_COLLECTION_NAME,
                persist_directory=Config.CHROMA_PERSIST_DIR
            )
            
            doc_id = os.path.splitext(metadata.get('file_name', 'document'))[0]
            
            # Collect all page results
            all_page_results = []
            for sm_result in mapper_results.values():
                if sm_result.get('status') == 'ok':
                    all_page_results.extend(sm_result['output'].get('results', []))
            
            # Create chunks and add to vector store
            chunks = create_chunks_from_pages(all_page_results)
            vector_store.add_document_chunks(
                doc_id=doc_id,
                chunks=chunks,
                metadata={
                    "file_name": metadata.get('file_name'),
                    "document_type": metadata.get('document_type')
                }
            )
            
            print(f"\n✅ Vector DB: Stored {len(chunks)} chunks for semantic search")
            
        except Exception as e:
            print(f"\n⚠️  Vector DB storage failed (non-critical): {e}")
            logger.warning(f"Vector DB stage failed: {e}")
    else:
        print("\n⏭️  [STAGE 4/8] VECTOR DB: Skipped (disabled or not available)")
    
    # ========================================================================
    # STAGE 5: REDUCER - Aggregate and Consolidate Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 [STAGE 5/8] REDUCER: Aggregating SubMaster results...")
    print("=" * 80)
    
    try:
        reducer = Reducer(output_dir=Config.OUTPUT_DIR)
        reduced_results = reducer.execute(mapper_results, metadata)
        
        consolidated = reduced_results.get('consolidated_analysis', {})
        print(f"\n✅ Reducer completed")
        print(f"   📈 Unique Entities: {consolidated.get('total_unique_entities', 0)}")
        print(f"   🔑 Unique Keywords: {consolidated.get('total_unique_keywords', 0)}")
        print(f"   💾 Output: {reduced_results.get('output_path')}")
        
    except Exception as e:
        print(f"\n❌ Reducer failed: {e}")
        logger.exception("Reducer stage failed")
        return 1
    
    # ========================================================================
    # STAGE 6: RESIDUAL AGENT (Optional) - Quality Validation
    # ========================================================================
    if Config.ENABLE_RESIDUAL_AGENT:
        print("\n" + "=" * 80)
        print("🔧 [STAGE 6/8] RESIDUAL AGENT: Validating and fixing results...")
        print("=" * 80)
        
        try:
            residual_agent = ResidualAgent(max_retries=3)
            
            # Validate results
            validation = residual_agent.validate_results(mapper_results)
            print(f"\n✅ Validation complete")
            print(f"   📊 Quality Score: {validation['quality_score']}/100")
            print(f"   ❌ Errors: {validation['error_count']}")
            print(f"   ⚠️  Warnings: {validation['warning_count']}")
            
            # Use fixed results if available
            if validation['fixed_results']:
                mapper_results = validation['fixed_results']
                print(f"   🔧 Applied fixes to {len(validation['fixed_results'])} SubMaster results")
            
            # Detect anomalies
            anomalies = residual_agent.detect_anomalies(mapper_results)
            if anomalies:
                print(f"   ⚠️  Detected {len(anomalies)} anomalies")
            
        except Exception as e:
            print(f"\n⚠️  Residual Agent failed (non-critical): {e}")
            logger.warning(f"Residual Agent stage failed: {e}")
    else:
        print("\n⏭️  [STAGE 6/8] RESIDUAL AGENT: Skipped (disabled)")
    
    # ========================================================================
    # STAGE 7: MERGE SUPERVISOR - Combine Mapper + Reducer
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔀 [STAGE 7/8] MERGE SUPERVISOR: Combining Mapper + Reducer outputs...")
    print("=" * 80)
    
    try:
        merge_supervisor = MergeSupervisor(use_llm=Config.ENABLE_LLM_MERGE)
        final_report = merge_supervisor.merge(mapper_results, reduced_results, metadata)
        
        # Save merged report
        merged_path = merge_supervisor.save_merged_report(final_report, Config.OUTPUT_DIR)
        
        quality = final_report.get('quality_metrics', {})
        print(f"\n✅ Merge completed")
        print(f"   📊 Overall Quality: {quality.get('overall_quality_score', 0)}/100 ({quality.get('quality_rating', 'N/A')})")
        print(f"   💾 Final Report: {merged_path}")
        
    except Exception as e:
        print(f"\n❌ Merge Supervisor failed: {e}")
        logger.exception("Merge Supervisor stage failed")
        return 1
    
    # ========================================================================
    # STAGE 8: REPORT GENERATOR - Create PDF + JSON Reports
    # ========================================================================
    print("\n" + "=" * 80)
    print("📄 [STAGE 8/8] REPORT GENERATOR: Creating final reports...")
    print("=" * 80)
    
    try:
        report_files = generate_analysis_report(mapper_results, metadata, Config.OUTPUT_DIR)
        
        print("\n✅ Reports generated:")
        if 'json' in report_files:
            print(f"   📊 JSON: {report_files['json']}")
        if 'pdf' in report_files:
            print(f"   📄 PDF: {report_files['pdf']}")
        elif 'pdf_error' in report_files:
            print(f"   ⚠️  PDF generation failed: {report_files['pdf_error']}")
        
    except Exception as e:
        print(f"\n⚠️  Report generation failed (non-critical): {e}")
        logger.warning(f"Report generation failed: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_elapsed = time.time() - pipeline_start
    
    print("\n" + "=" * 80)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    # Processing stats
    processing_stats = reduced_results.get('processing_stats', {})
    print(f"\n📈 PROCESSING STATISTICS:")
    print(f"   Total Pages: {metadata.get('num_pages', 0)}")
    print(f"   Pages Processed: {processing_stats.get('total_submasters', 0) * metadata.get('num_pages', 0) // len(mapper_results)}")
    print(f"   SubMasters: {processing_stats.get('total_submasters', 0)}")
    print(f"   LLM Success Rate: {processing_stats.get('success_rate', 0):.1f}%")
    
    # Entity/Keyword stats
    consolidated = reduced_results.get('consolidated_analysis', {})
    print(f"\n🔍 EXTRACTION STATISTICS:")
    print(f"   Unique Entities: {consolidated.get('total_unique_entities', 0)}")
    print(f"   Unique Keywords: {consolidated.get('total_unique_keywords', 0)}")
    print(f"   Top Entity: {consolidated.get('top_entities', [{}])[0].get('entity', 'N/A') if consolidated.get('top_entities') else 'N/A'}")
    print(f"   Top Keyword: {consolidated.get('top_keywords', [{}])[0].get('keyword', 'N/A') if consolidated.get('top_keywords') else 'N/A'}")
    
    # Quality metrics
    quality = final_report.get('quality_metrics', {})
    print(f"\n✅ QUALITY METRICS:")
    print(f"   Overall Score: {quality.get('overall_quality_score', 0):.1f}/100")
    print(f"   Rating: {quality.get('quality_rating', 'N/A')}")
    print(f"   Coverage: {quality.get('coverage_score', 0):.1f}%")
    
    # Timing
    print(f"\n⏱️  TIMING:")
    print(f"   Total Pipeline Time: {total_elapsed:.2f}s")
    print(f"   Mapper Time: ~{mapper_result.get('elapsed_time', 0):.2f}s")
    print(f"   Orchestrator Time: ~{sum(r['output'].get('elapsed_time', 0) for r in mapper_results.values() if r.get('status') == 'ok'):.2f}s")
    print(f"   Reducer Time: {processing_stats.get('elapsed_time', 0):.2f}s")
    print(f"   Merge Time: {final_report['processing_statistics'].get('merge_time', 0):.2f}s")
    
    # Output files
    print(f"\n📁 OUTPUT FILES:")
    print(f"   Metadata: {metadata_path}")
    print(f"   Reduced Results: {reduced_results.get('output_path')}")
    print(f"   Final Report: {merged_path}")
    if 'json' in report_files:
        print(f"   Analysis JSON: {report_files['json']}")
    if 'pdf' in report_files:
        print(f"   Analysis PDF: {report_files['pdf']}")
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Usage: python run_complete_pipeline.py <pdf_path> [config.json]")
        print("\nExample:")
        print("  python run_complete_pipeline.py data/paper.pdf")
        print("  python run_complete_pipeline.py data/paper.pdf config.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Optional: Load custom config
    config = None
    if len(sys.argv) > 2:
        config_path = sys.argv[2]
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded custom config from: {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to load config (using defaults): {e}")
    
    # Generate pipeline ID
    pipeline_id = f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    exit_code = run_complete_pipeline(pdf_path, config, pipeline_id)
    sys.exit(exit_code)
