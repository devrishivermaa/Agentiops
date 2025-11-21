# orchestrator.py
"""Simplified orchestrator for SubMaster execution."""

import ray
from agents.sub_master import SubMaster
from utils.logger import get_logger

logger = get_logger("Orchestrator")

def start_ray_if_needed():
    """Initialize Ray if not already running."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
        logger.info("Ray initialized")

def spawn_submasters_and_run(plan, metadata):
    """Spawn SubMasters and execute processing."""
    start_ray_if_needed()
    
    actors = {}
    results = {}
    
    logger.info(f"Spawning {len(plan.get('submasters', []))} SubMasters...")
    
    # Create SubMaster actors
    for sm in plan.get("submasters", []):
        sm_id = sm["submaster_id"]
        try:
            actor = SubMaster.options(name=sm_id).remote(sm, metadata)
            actors[sm_id] = actor
            logger.info(f"✅ Spawned {sm_id}: {sm.get('role', 'N/A')[:60]}")
        except Exception as e:
            logger.error(f"❌ Failed to spawn {sm_id}: {e}")
            results[sm_id] = {"status": "error", "error": f"Spawn failed: {e}"}
    
    # Initialize all actors
    logger.info("Initializing SubMasters...")
    init_futures = [a.initialize.remote() for a in actors.values()]
    ray.get(init_futures)
    logger.info("✅ All SubMasters initialized")
    
    # Process in parallel
    logger.info("Starting parallel processing...")
    process_futures = {sm_id: actor.process.remote() for sm_id, actor in actors.items()}
    
    # Collect results
    for sm_id, future in process_futures.items():
        try:
            output = ray.get(future)
            results[sm_id] = {"status": "ok", "output": output}
            logger.info(f"✅ {sm_id} completed successfully")
        except Exception as e:
            logger.error(f"❌ {sm_id} failed: {e}")
            results[sm_id] = {"status": "error", "error": str(e)}
    
    logger.info(f"Processing complete: {len(results)} SubMasters")
    return results
