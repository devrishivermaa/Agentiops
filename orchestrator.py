"""
Orchestrator: Coordinates SubMaster execution with ResidualAgent integration.

Features:
- Spawns and manages SubMaster agents
- Integrates ResidualAgent for global context distribution
- Emits real-time events for monitoring
- Handles errors and provides comprehensive reporting
"""

import ray
import time
from typing import Dict, Any, Optional, List
from agents.sub_master import SubMaster
from agents.residual_agent import ResidualAgentActor
from utils.logger import get_logger

# Import event emission (optional - graceful fallback if API not available)
try:
    from api.events import (
        event_bus,
        EventType,
        emit_agent_event,
        emit_submaster_progress,
        emit_pipeline_step,
    )
    EVENTS_ENABLED = True
except ImportError:
    EVENTS_ENABLED = False

logger = get_logger("Orchestrator")


def emit_event(event_type, pipeline_id, data=None, agent_id=None, agent_type=None):
    """Emit event if API layer is available."""
    if EVENTS_ENABLED and pipeline_id:
        try:
            event_bus.emit_simple(
                event_type, pipeline_id, data or {}, agent_id=agent_id, agent_type=agent_type
            )
        except Exception as e:
            logger.debug(f"Event emission failed: {e}")


def start_ray_if_needed():
    """Initialize Ray if not already running."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4, logging_level="ERROR")
        logger.info("✅ Ray initialized")


def run_submasters(
    plan: Dict,
    metadata: Dict,
    pipeline_id: Optional[str] = None,
    use_residual_agent: bool = True
) -> Dict[str, Any]:
    """
    Spawn SubMasters and execute processing with event emission.
    
    Args:
        plan: SubMaster execution plan from MasterAgent
        metadata: Document metadata from Mapper
        pipeline_id: Optional pipeline ID for event emission
        use_residual_agent: Whether to use ResidualAgent for context (default: True)
        
    Returns:
        Dict containing:
            - results: Dict mapping sm_id to result/error
            - summary: Execution summary
            - residual_context: Global context from ResidualAgent (if used)
    """
    return spawn_submasters_and_run(plan, metadata, pipeline_id, use_residual_agent)


def spawn_submasters_and_run(
    plan: Dict,
    metadata: Dict,
    pipeline_id: Optional[str] = None,
    use_residual_agent: bool = True
) -> Dict[str, Any]:
    """
    Spawn SubMasters and execute processing with optional ResidualAgent.
    
    Main orchestration workflow:
    1. Initialize Ray
    2. Create ResidualAgent (if enabled)
    3. Generate and distribute global context
    4. Spawn SubMaster actors with ResidualAgent handle
    5. Initialize SubMasters
    6. Execute processing in parallel
    7. Collect and return results
    
    Args:
        plan: Execution plan from MasterAgent
        metadata: Document metadata from Mapper
        pipeline_id: Pipeline ID for event tracking
        use_residual_agent: Whether to use ResidualAgent
        
    Returns:
        Dict with results, summary, and optional residual context
    """
    
    start_time = time.time()
    start_ray_if_needed()
    
    actors = {}
    results = {}
    total_workers = 0
    residual_agent = None
    global_context = {}
    
    submasters_list = plan.get("submasters", [])
    num_submasters = len(submasters_list)
    
    logger.info(f"🚀 Orchestrator starting: {num_submasters} SubMasters")
    
    # ==================== STAGE 1: RESIDUAL AGENT ====================
    
    if use_residual_agent:
        try:
            logger.info("📦 Creating ResidualAgent...")
            
            # Emit pipeline step
            if EVENTS_ENABLED and pipeline_id:
                emit_pipeline_step(pipeline_id, "residual_init", "started")
            
            # Create ResidualAgent
            residual_agent = ResidualAgentActor.remote(
                model=metadata.get("preferred_model"),
                persist=True,
                max_retries=3,
                pipeline_id=pipeline_id
            )
            
            logger.info("✅ ResidualAgent created")
            
            # Generate and distribute global context
            logger.info("🧠 Generating global context...")
            
            context_result = ray.get(
                residual_agent.generate_and_distribute.remote(
                    metadata=metadata,
                    master_plan=plan,
                    wait_for_updates_seconds=10
                ),
                timeout=60
            )
            
            global_context = context_result
            
            logger.info(
                f"✅ Global context generated (v{global_context.get('version', 1)}, "
                f"{len(global_context.get('section_overview', {}).get('sections', []))} sections)"
            )
            
            if EVENTS_ENABLED and pipeline_id:
                emit_pipeline_step(
                    pipeline_id,
                    "residual_init",
                    "completed",
                    {
                        "context_version": global_context.get("version", 1),
                        "sections_count": len(global_context.get("section_overview", {}).get("sections", []))
                    }
                )
        
        except Exception as e:
            logger.error(f"⚠️ ResidualAgent initialization failed: {e}")
            logger.info("Continuing without ResidualAgent...")
            residual_agent = None
            
            if EVENTS_ENABLED and pipeline_id:
                emit_pipeline_step(pipeline_id, "residual_init", "failed", {"error": str(e)})
    else:
        logger.info("ResidualAgent disabled by configuration")
    
    # ==================== STAGE 2: SPAWN SUBMASTERS ====================
    
    logger.info(f"📦 Spawning {num_submasters} SubMasters...")
    
    if EVENTS_ENABLED and pipeline_id:
        emit_pipeline_step(pipeline_id, "submaster_spawn", "started", {"count": num_submasters})
    
    # Create SubMaster actors with ResidualAgent handle
    for sm in submasters_list:
        sm_id = sm["submaster_id"]
        try:
            actor = SubMaster.options(name=sm_id).remote(
                plan_piece=sm,
                metadata=metadata,
                pipeline_id=pipeline_id,
                residual_agent=residual_agent  # ← PASS RESIDUAL AGENT
            )
            actors[sm_id] = actor
            
            # Emit spawn event
            emit_event(
                EventType.SUBMASTER_SPAWNED,
                pipeline_id,
                {
                    "submaster_id": sm_id,
                    "role": sm.get("role", ""),
                    "page_range": sm.get("page_range", []),
                    "assigned_sections": sm.get("assigned_sections", []),
                    "has_residual_agent": residual_agent is not None,
                },
                agent_id=sm_id,
                agent_type="submaster",
            )
            
            logger.info(
                f"✅ Spawned {sm_id}: {sm.get('role', 'N/A')[:60]} "
                f"(pages {sm.get('page_range', [])})"
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to spawn {sm_id}: {e}")
            results[sm_id] = {"status": "error", "error": f"Spawn failed: {e}"}
            
            emit_event(
                EventType.SUBMASTER_FAILED,
                pipeline_id,
                {"submaster_id": sm_id, "error": str(e), "phase": "spawn"},
                agent_id=sm_id,
                agent_type="submaster",
            )
    
    if EVENTS_ENABLED and pipeline_id:
        emit_pipeline_step(
            pipeline_id,
            "submaster_spawn",
            "completed",
            {"spawned": len(actors), "failed": num_submasters - len(actors)}
        )
    
    # ==================== STAGE 3: INITIALIZE SUBMASTERS ====================
    
    logger.info(f"🔧 Initializing {len(actors)} SubMasters...")
    
    if EVENTS_ENABLED and pipeline_id:
        emit_pipeline_step(pipeline_id, "submaster_init", "started", {"count": len(actors)})
    
    init_futures = [a.initialize.remote() for a in actors.values()]
    
    try:
        init_results = ray.get(init_futures, timeout=120)
        
        # Track worker count and emit events
        for init_result in init_results:
            sm_id = init_result.get("sm_id")
            num_workers = init_result.get("num_workers", 0)
            total_workers += num_workers
            
            emit_event(
                EventType.SUBMASTER_INITIALIZED,
                pipeline_id,
                {
                    "submaster_id": sm_id,
                    "num_workers": num_workers,
                    "status": init_result.get("status"),
                },
                agent_id=sm_id,
                agent_type="submaster",
            )
            
            logger.debug(f"✅ {sm_id} initialized with {num_workers} workers")
        
        logger.info(f"✅ All SubMasters initialized ({total_workers} total workers)")
        
        if EVENTS_ENABLED and pipeline_id:
            emit_pipeline_step(
                pipeline_id,
                "submaster_init",
                "completed",
                {"initialized": len(init_results), "total_workers": total_workers}
            )
    
    except Exception as e:
        logger.error(f"❌ SubMaster initialization failed: {e}")
        
        if EVENTS_ENABLED and pipeline_id:
            emit_pipeline_step(pipeline_id, "submaster_init", "failed", {"error": str(e)})
        
        return {
            "results": results,
            "summary": {
                "status": "failed",
                "error": f"Initialization failed: {e}",
                "elapsed_time": time.time() - start_time
            },
            "residual_context": global_context
        }
    
    # ==================== STAGE 4: EXECUTE PROCESSING ====================
    
    logger.info(f"🚀 Starting parallel processing for {len(actors)} SubMasters...")
    
    if EVENTS_ENABLED and pipeline_id:
        emit_pipeline_step(pipeline_id, "execution", "started", {"submasters": len(actors)})
    
    # Emit processing started for each SubMaster
    for sm_id in actors:
        emit_event(
            EventType.SUBMASTER_PROCESSING,
            pipeline_id,
            {"submaster_id": sm_id},
            agent_id=sm_id,
            agent_type="submaster",
        )
    
    # Submit all processing tasks
    process_futures = {sm_id: actor.process.remote() for sm_id, actor in actors.items()}
    
    # Collect results as they complete
    completed_count = 0
    failed_count = 0
    
    for sm_id, future in process_futures.items():
        try:
            output = ray.get(future, timeout=1800)  # 30 min timeout per SubMaster
            results[sm_id] = {"status": "ok", "output": output}
            completed_count += 1
            
            # Emit completion event
            emit_event(
                EventType.SUBMASTER_COMPLETED,
                pipeline_id,
                {
                    "submaster_id": sm_id,
                    "total_pages": output.get("total_pages", 0),
                    "llm_successes": output.get("llm_successes", 0),
                    "llm_failures": output.get("llm_failures", 0),
                    "elapsed_time": output.get("elapsed_time", 0),
                    "used_global_context": output.get("used_global_context", False),
                },
                agent_id=sm_id,
                agent_type="submaster",
            )
            
            logger.info(
                f"✅ {sm_id} completed: "
                f"{output.get('total_pages', 0)} pages, "
                f"{output.get('llm_successes', 0)} successes, "
                f"{output.get('llm_failures', 0)} failures "
                f"({output.get('elapsed_time', 0):.1f}s)"
            )
        
        except ray.exceptions.GetTimeoutError:
            logger.error(f"❌ {sm_id} timed out after 30 minutes")
            results[sm_id] = {"status": "error", "error": "Processing timeout"}
            failed_count += 1
            
            emit_event(
                EventType.SUBMASTER_FAILED,
                pipeline_id,
                {"submaster_id": sm_id, "error": "Timeout", "phase": "processing"},
                agent_id=sm_id,
                agent_type="submaster",
            )
        
        except Exception as e:
            logger.error(f"❌ {sm_id} failed: {e}")
            results[sm_id] = {"status": "error", "error": str(e)}
            failed_count += 1
            
            emit_event(
                EventType.SUBMASTER_FAILED,
                pipeline_id,
                {"submaster_id": sm_id, "error": str(e), "phase": "processing"},
                agent_id=sm_id,
                agent_type="submaster",
            )
    
    elapsed_time = time.time() - start_time
    
    logger.info(
        f"🎉 Processing complete: {completed_count}/{len(results)} succeeded, "
        f"{failed_count} failed ({elapsed_time:.1f}s total)"
    )
    
    if EVENTS_ENABLED and pipeline_id:
        emit_pipeline_step(
            pipeline_id,
            "execution",
            "completed",
            {
                "completed": completed_count,
                "failed": failed_count,
                "total": len(results),
                "elapsed_time": elapsed_time
            }
        )
    
    # ==================== STAGE 5: GENERATE SUMMARY ====================
    
    summary = _generate_execution_summary(
        results=results,
        elapsed_time=elapsed_time,
        total_workers=total_workers,
        used_residual_agent=residual_agent is not None
    )
    
    # Get final global context snapshot if ResidualAgent was used
    if residual_agent:
        try:
            final_context = ray.get(residual_agent.get_snapshot.remote(), timeout=10)
            global_context = final_context
            logger.info(f"✅ Retrieved final global context (v{final_context.get('version', 1)})")
        except Exception as e:
            logger.warning(f"Failed to get final context snapshot: {e}")
    
    return {
        "results": results,
        "summary": summary,
        "residual_context": global_context,
        "residual_agent_used": residual_agent is not None
    }


def _generate_execution_summary(
    results: Dict[str, Any],
    elapsed_time: float,
    total_workers: int,
    used_residual_agent: bool
) -> Dict[str, Any]:
    """
    Generate execution summary from results.
    
    Args:
        results: Dict mapping sm_id to result/error
        elapsed_time: Total execution time
        total_workers: Total number of workers spawned
        used_residual_agent: Whether ResidualAgent was used
        
    Returns:
        Summary dict with statistics
    """
    
    total_submasters = len(results)
    successful = sum(1 for r in results.values() if r.get("status") == "ok")
    failed = total_submasters - successful
    
    # Aggregate statistics from successful SubMasters
    total_pages = 0
    total_chars = 0
    total_entities = 0
    total_keywords = 0
    total_llm_successes = 0
    total_llm_failures = 0
    
    for result in results.values():
        if result.get("status") == "ok":
            output = result.get("output", {})
            total_pages += output.get("total_pages", 0)
            total_chars += output.get("total_chars", 0)
            total_entities += output.get("total_entities", 0)
            total_keywords += output.get("total_keywords", 0)
            total_llm_successes += output.get("llm_successes", 0)
            total_llm_failures += output.get("llm_failures", 0)
    
    # Calculate rates
    success_rate = (successful / total_submasters * 100) if total_submasters > 0 else 0
    llm_success_rate = (
        (total_llm_successes / (total_llm_successes + total_llm_failures) * 100)
        if (total_llm_successes + total_llm_failures) > 0
        else 0
    )
    
    summary = {
        "status": "completed" if failed == 0 else "partial",
        "total_submasters": total_submasters,
        "successful_submasters": successful,
        "failed_submasters": failed,
        "success_rate": round(success_rate, 1),
        "total_workers": total_workers,
        "total_pages_processed": total_pages,
        "total_characters": total_chars,
        "total_entities_extracted": total_entities,
        "total_keywords_extracted": total_keywords,
        "llm_successes": total_llm_successes,
        "llm_failures": total_llm_failures,
        "llm_success_rate": round(llm_success_rate, 1),
        "elapsed_time": round(elapsed_time, 2),
        "pages_per_second": round(total_pages / elapsed_time, 2) if elapsed_time > 0 else 0,
        "used_residual_agent": used_residual_agent,
        "timestamp": time.time()
    }
    
    logger.info(
        f"📊 Summary: {successful}/{total_submasters} SubMasters, "
        f"{total_pages} pages, {total_llm_successes} LLM successes, "
        f"{llm_success_rate:.1f}% LLM success rate"
    )
    
    return summary


def get_orchestrator_status() -> Dict[str, Any]:
    """
    Get orchestrator status.
    
    Returns:
        Status dict with Ray and system information
    """
    
    status = {
        "ray_initialized": ray.is_initialized(),
        "timestamp": time.time()
    }
    
    if ray.is_initialized():
        try:
            ray_status = ray.cluster_resources()
            status["ray_resources"] = {
                "cpus": ray_status.get("CPU", 0),
                "memory": ray_status.get("memory", 0),
                "gpus": ray_status.get("GPU", 0)
            }
        except Exception as e:
            status["ray_error"] = str(e)
    
    return status


def shutdown_orchestrator():
    """Shutdown Ray and cleanup resources."""
    if ray.is_initialized():
        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("✅ Ray shutdown complete")
    else:
        logger.info("Ray was not initialized")
