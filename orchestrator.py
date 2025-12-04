# orchestrator.py
"""Simplified orchestrator for SubMaster execution with event emission."""

import ray
from typing import Dict, Any, Optional
from agents.sub_master import SubMaster
from utils.logger import get_logger

# Import event emission (optional - graceful fallback if API not available)
try:
    from api.events import (
        event_bus,
        EventType,
        emit_agent_event,
        emit_submaster_progress,
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
        ray.init(ignore_reinit_error=True, num_cpus=4)
        logger.info("Ray initialized")


def run_submasters(plan: Dict, metadata: Dict, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Spawn SubMasters and execute processing with event emission.
    
    Args:
        plan: SubMaster execution plan from MasterAgent
        metadata: Document metadata from Mapper
        pipeline_id: Optional pipeline ID for event emission
        
    Returns:
        Dict mapping sm_id to result/error
    """
    return spawn_submasters_and_run(plan, metadata, pipeline_id)


def spawn_submasters_and_run(
    plan: Dict, 
    metadata: Dict, 
    pipeline_id: Optional[str] = None
) -> Dict[str, Any]:
    """Spawn SubMasters and execute processing."""
    start_ray_if_needed()
    
    actors = {}
    results = {}
    total_workers = 0
    num_workers_per_sm = metadata.get("num_workers_per_submaster", 3)
    submasters_list = plan.get("submasters", [])
    
    logger.info(f"Spawning {len(submasters_list)} SubMasters...")
    
    # Create SubMaster actors
    for sm in submasters_list:
        sm_id = sm["submaster_id"]
        try:
            actor = SubMaster.options(name=sm_id).remote(sm, metadata, pipeline_id)
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
                },
                agent_id=sm_id,
                agent_type="submaster",
            )
            
            logger.info(f"✅ Spawned {sm_id}: {sm.get('role', 'N/A')[:60]}")
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
    
    # Initialize all actors
    logger.info("Initializing SubMasters...")
    init_futures = [a.initialize.remote() for a in actors.values()]
    init_results = ray.get(init_futures)
    
    # Track worker count
    for init_result in init_results:
        total_workers += init_result.get("num_workers", 0)
        sm_id = init_result.get("sm_id")
        
        emit_event(
            EventType.SUBMASTER_INITIALIZED,
            pipeline_id,
            {
                "submaster_id": sm_id,
                "num_workers": init_result.get("num_workers", 0),
                "status": init_result.get("status"),
            },
            agent_id=sm_id,
            agent_type="submaster",
        )
    
    logger.info(f"✅ All SubMasters initialized ({total_workers} total workers)")
    
    # Process in parallel
    logger.info("Starting parallel processing...")
    
    # Emit processing started for each
    for sm_id in actors:
        emit_event(
            EventType.SUBMASTER_PROCESSING,
            pipeline_id,
            {"submaster_id": sm_id},
            agent_id=sm_id,
            agent_type="submaster",
        )
    
    process_futures = {sm_id: actor.process.remote() for sm_id, actor in actors.items()}
    
    # Collect results
    completed_count = 0
    for sm_id, future in process_futures.items():
        try:
            output = ray.get(future)
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
                },
                agent_id=sm_id,
                agent_type="submaster",
            )
            
            logger.info(f"✅ {sm_id} completed successfully")
        except Exception as e:
            logger.error(f"❌ {sm_id} failed: {e}")
            results[sm_id] = {"status": "error", "error": str(e)}
            
            emit_event(
                EventType.SUBMASTER_FAILED,
                pipeline_id,
                {"submaster_id": sm_id, "error": str(e), "phase": "processing"},
                agent_id=sm_id,
                agent_type="submaster",
            )
    
    logger.info(f"Processing complete: {completed_count}/{len(results)} SubMasters succeeded")
    return results
