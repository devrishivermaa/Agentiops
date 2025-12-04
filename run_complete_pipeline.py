"""
COMPLETE PIPELINE: Mapper → MasterAgent → ResidualAgent → SubMasters → Reducer → Merger → Reports
Fully implements Architecture 1 with all 8 stages including ResidualAgent coordination.

FIXED:
- Safe None handling for global_context
- Proper string slicing with type validation
- Type checking before operations
- Graceful degradation when components fail
- MongoDB SSL errors handled
"""

import os
import sys
import json
import time
from datetime import datetime
from config import Config
from workflows.mapper import Mapper
from workflows.reducer import Reducer
from workflows.merge_supervisor import MergeSupervisor
from agents.master_agent import MasterAgent
from agents.residual_agent import ResidualAgentActor
from orchestrator import spawn_submasters_and_run
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


def safe_truncate(value, max_len=60, suffix='...'):
    """
    Safely truncate any value to string with max length.
    
    Args:
        value: Any value (str, dict, list, None, etc.)
        max_len: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string representation
    """
    if value is None:
        return 'N/A'
    
    # Convert to string if not already
    if not isinstance(value, str):
        try:
            value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        except:
            value_str = str(value)
    else:
        value_str = value
    
    # Truncate if needed
    if len(value_str) > max_len:
        return value_str[:max_len] + suffix
    
    return value_str


def run_complete_pipeline(pdf_path: str, config: dict = None, pipeline_id: str = None):
    """
    Run the COMPLETE document processing pipeline with all stages.
    
    Pipeline Stages (Architecture 1):
    1. Mapper: Extract metadata and validate PDF
    2. MasterAgent: Generate SubMaster execution plan
    3. ResidualAgent: Generate global context and coordinate agents
    4. Orchestrator: Execute SubMasters in parallel with global context
    5. [Optional] Vector DB: Store embeddings for semantic search
    6. Reducer: Aggregate and consolidate results
    7. ResidualAgent Quality Check: Validate and fix errors
    8. Merger: Combine Mapper + Reducer outputs
    9. Report Generator: Create final PDF + JSON reports
    
    Args:
        pdf_path: Path to PDF file
        config: Optional user configuration
        pipeline_id: Optional pipeline ID for tracking
    
    Returns:
        0 on success, 1 on failure
    """
    pipeline_start = time.time()
    
    print("\n" + "=" * 80)
    print("🚀 AGENTOPS COMPLETE PIPELINE (Architecture 1 + ResidualAgent)")
    print("=" * 80)
    print(f"📄 Input: {pdf_path}")
    print(f"🆔 Pipeline ID: {pipeline_id or 'N/A'}")
    print(f"🤖 ResidualAgent: {'ENABLED' if Config.ENABLE_RESIDUAL_AGENT else 'DISABLED'}")
    print("=" * 80)
    
    # Use config or defaults
    if config is None:
        config = Config.get_user_config()
    
    # ========================================================================
    # STAGE 1: MAPPER - Validate PDF and Extract Metadata
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 [STAGE 1/9] MAPPER: Validating PDF and extracting metadata...")
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
    print("🤖 [STAGE 2/9] MASTER AGENT: Generating SubMaster execution plan...")
    print("=" * 80)
    
    try:
        master_agent = MasterAgent()
        plan = master_agent.execute(metadata_path)
        
        if plan is None or plan.get("status") != "approved":
            print("\n❌ Plan generation failed or not approved")
            return 1
        
        num_submasters = plan.get('num_submasters', len(plan.get('submasters', [])))
        print(f"\n✅ Plan approved: {num_submasters} SubMasters")
        
        # Show SubMaster breakdown
        for sm in plan.get('submasters', [])[:3]:  # Show first 3
            print(f"   • {sm['submaster_id']}: {sm.get('role', 'N/A')[:50]} (pages {sm.get('page_range', [])})")
        if num_submasters > 3:
            print(f"   ... and {num_submasters - 3} more")
        
    except Exception as e:
        print(f"\n❌ Master Agent failed: {e}")
        logger.exception("MasterAgent stage failed")
        return 1
    
    # ========================================================================
    # STAGE 3: ORCHESTRATOR - Execute SubMasters with ResidualAgent
    # ========================================================================
    print("\n" + "=" * 80)
    print("⚙️  [STAGE 3/9] ORCHESTRATOR: Executing SubMasters with ResidualAgent...")
    print("=" * 80)
    
    try:
        # Orchestrator handles ResidualAgent creation internally
        orchestration_result = spawn_submasters_and_run(
            plan=plan,
            metadata=metadata,
            pipeline_id=pipeline_id,
            use_residual_agent=Config.ENABLE_RESIDUAL_AGENT
        )
        
        mapper_results = orchestration_result['results']
        summary = orchestration_result['summary']
        global_context = orchestration_result.get('residual_context', {})
        residual_used = orchestration_result.get('residual_agent_used', False)
        
        print(f"\n✅ Orchestration completed")
        print(f"   📊 SubMasters: {summary['successful_submasters']}/{summary['total_submasters']} succeeded")
        print(f"   📄 Pages: {summary['total_pages_processed']}")
        print(f"   🎯 LLM Success: {summary['llm_success_rate']:.1f}%")
        print(f"   ⏱️  Time: {summary['elapsed_time']:.2f}s")
        
        # FIXED: Safe context version extraction
        context_version = global_context.get('version', 1) if isinstance(global_context, dict) else 0
        print(f"   🤖 ResidualAgent: {'Used (v{})'.format(context_version) if residual_used else 'Not used'}")
        
        # Show global context summary if available
        if residual_used and isinstance(global_context, dict):
            print(f"\n🧠 Global Context Summary:")
            
            # FIXED: Safe display with helper function
            def show_field(label, value, max_len=60):
                """Display field safely with truncation"""
                if not value:
                    return
                val_str = str(value) if not isinstance(value, str) else value
                if len(val_str) > max_len:
                    val_str = val_str[:max_len] + '...'
                print(f"   • {label}: {val_str}")
            
            show_field("Intent", global_context.get('high_level_intent'))
            show_field("Strategy", global_context.get('master_strategy'))
            show_field("Context", global_context.get('document_context'))
            
            # FIXED: Safe list/dict extraction
            entities = global_context.get('global_entities', [])
            keywords = global_context.get('global_keywords', [])
            section_overview = global_context.get('section_overview', {})
            sections = section_overview.get('sections', []) if isinstance(section_overview, dict) else []
            
            print(f"   • Sections: {len(sections) if isinstance(sections, list) else 0}")
            print(f"   • Global Entities: {len(entities) if isinstance(entities, list) else 0}")
            print(f"   • Global Keywords: {len(keywords) if isinstance(keywords, list) else 0}")
        
    except Exception as e:
        print(f"\n❌ Orchestrator failed: {e}")
        logger.exception("Orchestrator stage failed")
        return 1
    
    # ========================================================================
    # STAGE 4: VECTOR DB (Optional) - Store Embeddings
    # ========================================================================
    if VECTOR_DB_AVAILABLE and Config.ENABLE_VECTOR_DB:
        print("\n" + "=" * 80)
        print("🔍 [STAGE 4/9] VECTOR DB: Storing document embeddings...")
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
                    "document_type": metadata.get('document_type'),
                    "pipeline_id": pipeline_id
                }
            )
            
            print(f"\n✅ Vector DB: Stored {len(chunks)} chunks for semantic search")
            
        except Exception as e:
            print(f"\n⚠️  Vector DB storage failed (non-critical): {e}")
            logger.warning(f"Vector DB stage failed: {e}")
    else:
        print("\n⏭️  [STAGE 4/9] VECTOR DB: Skipped (disabled or not available)")
    
    # ========================================================================
    # STAGE 5: REDUCER - Aggregate and Consolidate Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 [STAGE 5/9] REDUCER: Aggregating SubMaster results...")
    print("=" * 80)
    
    try:
        reducer = Reducer(output_dir=Config.OUTPUT_DIR)
        reduced_results = reducer.execute(mapper_results, metadata)
        
        consolidated = reduced_results.get('consolidated_analysis', {})
        print(f"\n✅ Reducer completed")
        print(f"   📈 Unique Entities: {consolidated.get('total_unique_entities', 0)}")
        print(f"   🔑 Unique Keywords: {consolidated.get('total_unique_keywords', 0)}")
        
        # FIXED: Safe access to processing_stats
        processing_stats = reduced_results.get('processing_stats', {})
        document_info = reduced_results.get('document', {})
        total_pages = document_info.get('pages_processed', 0)
        print(f"   📄 Pages Analyzed: {total_pages}")
        
        output_path = reduced_results.get('output_path', 'N/A')
        print(f"   💾 Output: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Reducer failed: {e}")
        logger.exception("Reducer stage failed")
        return 1
    
    # ========================================================================
    # STAGE 6: RESIDUAL AGENT QUALITY CHECK - Validate Results
    # ========================================================================
    if Config.ENABLE_RESIDUAL_AGENT:
        print("\n" + "=" * 80)
        print("🔧 [STAGE 6/9] RESIDUAL AGENT: Quality validation and error recovery...")
        print("=" * 80)
        
        try:
            from agents.residual_agent_validator import ResidualAgentValidator
            
            validator = ResidualAgentValidator(max_retries=3)
            
            # Validate results
            validation = validator.validate_results(mapper_results)
            print(f"\n✅ Validation complete")
            print(f"   📊 Quality Score: {validation['quality_score']}/100")
            print(f"   ❌ Errors: {validation['error_count']}")
            print(f"   ⚠️  Warnings: {validation['warning_count']}")
            
            # Use fixed results if available
            if validation.get('fixed_results'):
                mapper_results = validation['fixed_results']
                print(f"   🔧 Applied fixes to {len(validation['fixed_results'])} SubMaster results")
            
            # Detect anomalies
            anomalies = validator.detect_anomalies(mapper_results)
            if anomalies:
                print(f"   ⚠️  Detected {len(anomalies)} anomalies")
                for anomaly in anomalies[:3]:  # Show first 3
                    print(f"      • {anomaly['sm_id']}: {anomaly['type']} (severity: {anomaly['severity']})")
            
            # Analyze failed tasks if any
            failed_tasks = [
                {"task_id": sm_id, "error": r.get('error', 'Unknown')}
                for sm_id, r in mapper_results.items()
                if r.get('status') == 'error'
            ]
            
            if failed_tasks:
                print(f"\n📋 Analyzing {len(failed_tasks)} failed tasks...")
                retry_analysis = validator.retry_failed_tasks(failed_tasks, metadata)
                recommended = sum(1 for r in retry_analysis if r.get('retry_recommended'))
                print(f"   ✅ {recommended}/{len(failed_tasks)} tasks recommended for retry")
            
        except ImportError:
            print(f"\n⚠️  ResidualAgent validator not available, using basic validation")
            logger.warning("ResidualAgent validator not found - create agents/residual_agent_validator.py")
            
            # Basic validation
            validation_score = 100
            error_count = sum(1 for r in mapper_results.values() if r.get('status') == 'error')
            warning_count = 0
            
            if error_count > 0:
                validation_score -= (error_count * 20)
            
            print(f"\n✅ Basic validation complete")
            print(f"   📊 Quality Score: {validation_score}/100")
            print(f"   ❌ Errors: {error_count}")
            print(f"   ⚠️  Warnings: {warning_count}")
            
        except Exception as e:
            print(f"\n⚠️  Residual Agent validation failed (non-critical): {e}")
            logger.warning(f"Residual Agent validation failed: {e}")
    else:
        print("\n⏭️  [STAGE 6/9] RESIDUAL AGENT: Skipped (disabled)")
    
    # ========================================================================
    # STAGE 7: MERGE SUPERVISOR - Combine Mapper + Reducer
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔀 [STAGE 7/9] MERGE SUPERVISOR: Combining outputs...")
    print("=" * 80)
    
    try:
        merge_supervisor = MergeSupervisor(use_llm=Config.ENABLE_LLM_MERGE)
        
        # FIXED: Add global context to metadata instead of passing as kwarg
        if residual_used and global_context and isinstance(global_context, dict):
            metadata['residual_context'] = {
                'version': global_context.get('version', 1),
                'high_level_intent': safe_truncate(global_context.get('high_level_intent', ''), 200),
                'document_context': safe_truncate(global_context.get('document_context', ''), 200),
                'top_entities': (global_context.get('global_entities', []) or [])[:10],
                'top_keywords': (global_context.get('global_keywords', []) or [])[:10]
            }
        
        final_report = merge_supervisor.merge(
            mapper_results,
            reduced_results,
            metadata
        )
        
        # Save merged report
        merged_path = merge_supervisor.save_merged_report(final_report, Config.OUTPUT_DIR)
        
        quality = final_report.get('quality_metrics', {})
        print(f"\n✅ Merge completed")
        print(f"   📊 Overall Quality: {quality.get('overall_quality_score', 0):.1f}/100 ({quality.get('quality_rating', 'N/A')})")
        print(f"   📈 Coverage: {quality.get('coverage_score', 0):.1f}%")
        print(f"   💾 Final Report: {merged_path}")
        
    except Exception as e:
        print(f"\n❌ Merge Supervisor failed: {e}")
        logger.exception("Merge Supervisor stage failed")
        return 1
    
    # ========================================================================
    # STAGE 8: REPORT GENERATOR - Create PDF + JSON Reports
    # ========================================================================
    print("\n" + "=" * 80)
    print("📄 [STAGE 8/9] REPORT GENERATOR: Creating final reports...")
    print("=" * 80)
    
    try:
        # Include global context in report if available
        report_metadata = metadata.copy()
        if residual_used and global_context and isinstance(global_context, dict):
            report_metadata['global_context'] = {
                'version': global_context.get('version', 1),
                'high_level_intent': safe_truncate(global_context.get('high_level_intent', ''), 200),
                'document_context': safe_truncate(global_context.get('document_context', ''), 200),
                'top_entities': (global_context.get('global_entities', []) or [])[:10],
                'top_keywords': (global_context.get('global_keywords', []) or [])[:10]
            }
        
        report_files = generate_analysis_report(
            mapper_results,
            report_metadata,
            Config.OUTPUT_DIR
        )
        
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
        report_files = {}
    
    # ========================================================================
    # STAGE 9: FINAL SUMMARY AND CLEANUP
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 [STAGE 9/9] FINAL SUMMARY")
    print("=" * 80)
    
    total_elapsed = time.time() - pipeline_start
    
    # Processing stats
    processing_stats = reduced_results.get('processing_stats', {})
    print(f"\n📈 PROCESSING STATISTICS:")
    print(f"   Total Pages: {metadata.get('num_pages', 0)}")
    print(f"   Pages Processed: {summary.get('total_pages_processed', 0)}")
    print(f"   SubMasters: {summary.get('total_submasters', 0)}")
    print(f"   Workers: {summary.get('total_workers', 0)}")
    print(f"   LLM Success Rate: {summary.get('llm_success_rate', 0):.1f}%")
    
    # Entity/Keyword stats
    consolidated = reduced_results.get('consolidated_analysis', {})
    print(f"\n🔍 EXTRACTION STATISTICS:")
    print(f"   Unique Entities: {consolidated.get('total_unique_entities', 0)}")
    print(f"   Unique Keywords: {consolidated.get('total_unique_keywords', 0)}")
    
    top_entities = consolidated.get('top_entities', [])
    if top_entities and len(top_entities) > 0:
        print(f"   Top Entity: {top_entities[0].get('entity', 'N/A')} ({top_entities[0].get('count', 0)})")
    
    top_keywords = consolidated.get('top_keywords', [])
    if top_keywords and len(top_keywords) > 0:
        print(f"   Top Keyword: {top_keywords[0].get('keyword', 'N/A')} ({top_keywords[0].get('count', 0)})")
    
    # Global context stats
    if residual_used and global_context and isinstance(global_context, dict):
        print(f"\n🤖 RESIDUAL AGENT CONTEXT:")
        print(f"   Version: {global_context.get('version', 1)}")
        
        # FIXED: Safe sections extraction
        section_overview = global_context.get('section_overview')
        section_count = 0
        if isinstance(section_overview, dict):
            sections = section_overview.get('sections', [])
            section_count = len(sections) if isinstance(sections, list) else 0
        
        print(f"   Sections: {section_count}")
        
        entities = global_context.get('global_entities', [])
        keywords = global_context.get('global_keywords', [])
        print(f"   Global Entities: {len(entities) if isinstance(entities, list) else 0}")
        print(f"   Global Keywords: {len(keywords) if isinstance(keywords, list) else 0}")
    
    # Quality metrics
    quality = final_report.get('quality_metrics', {})
    print(f"\n✅ QUALITY METRICS:")
    print(f"   Overall Score: {quality.get('overall_quality_score', 0):.1f}/100")
    print(f"   Rating: {quality.get('quality_rating', 'N/A')}")
    print(f"   Coverage: {quality.get('coverage_score', 0):.1f}%")
    print(f"   Success Rate: {quality.get('success_rate', 0):.1f}%")
    
    # Timing breakdown
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"   Total Pipeline: {total_elapsed:.2f}s")
    print(f"   Mapper: {mapper_result.get('elapsed_time', 0):.2f}s")
    print(f"   Orchestrator: {summary.get('elapsed_time', 0):.2f}s")
    print(f"   Reducer: {processing_stats.get('elapsed_time', 0):.2f}s")
    
    # FIXED: Safe access to merge_time
    merge_stats = final_report.get('processing_statistics', {})
    merge_time = merge_stats.get('merge_time', 0) if isinstance(merge_stats, dict) else 0
    print(f"   Merger: {merge_time:.2f}s")
    print(f"   Pages/Second: {summary.get('pages_per_second', 0):.2f}")
    
    # Output files
    print(f"\n📁 OUTPUT FILES:")
    print(f"   Metadata: {metadata_path}")
    print(f"   Reduced Results: {reduced_results.get('output_path', 'N/A')}")
    print(f"   Final Report: {merged_path}")
    if 'json' in report_files:
        print(f"   Analysis JSON: {report_files['json']}")
    if 'pdf' in report_files:
        print(f"   Analysis PDF: {report_files['pdf']}")
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    
    # Cleanup Ray
    try:
        import ray
        if ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✅ Ray shutdown complete")
    except Exception as e:
        logger.warning(f"Ray shutdown warning: {e}")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Usage: python run_complete_pipeline.py <pdf_path> [config.json]")
        print("\nExample:")
        print("  python run_complete_pipeline.py data/paper.pdf")
        print("  python run_complete_pipeline.py data/paper.pdf config.json")
        print("\nEnvironment Variables:")
        print("  ENABLE_RESIDUAL_AGENT=true/false  - Enable ResidualAgent (default: true)")
        print("  ENABLE_VECTOR_DB=true/false       - Enable Vector DB (default: false)")
        print("  LLM_MODEL=model-name               - LLM model to use (default: mistral-small-latest)")
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
