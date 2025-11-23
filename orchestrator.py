# orchestrator.py

import ray
import time
from agents.sub_master import SubMaster
from utils.logger import get_logger

logger = get_logger("Orchestrator")


def start_ray_if_needed():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
        logger.info("Ray initialized")


def spawn_submasters_and_run(plan, metadata, residual_handle=None):
    start_ray_if_needed()

    actors = {}
    results = {}

    logger.info(f"Spawning {len(plan.get('submasters', []))} SubMasters...")

    # -----------------------------
    # 1. Spawn SubMasters
    # -----------------------------
    for sm in plan.get("submasters", []):
        sm_id = sm["submaster_id"]
        actor = SubMaster.remote(sm, metadata, residual_handle)
        actors[sm_id] = actor
        logger.info(f"Spawned: {sm_id}")

    # -----------------------------
    # 2. Register SubMasters
    # -----------------------------
    if residual_handle:
        ray.get(residual_handle.register_submasters.remote(list(actors.values())))
        logger.info("SubMasters registered to ResidualAgent")

    # -----------------------------
    # 3. Initialize SubMasters
    # -----------------------------
    logger.info("Initializing SubMasters (workers will be created)...")
    init_futures = [a.initialize.remote() for a in actors.values()]
    init_results = ray.get(init_futures)
    logger.info(f"All SubMasters initialized: {init_results}")

    # -----------------------------
    # 4. Register workers
    # -----------------------------
    worker_handles = []
    for sm_id, a in actors.items():
        handles = ray.get(a.get_worker_handles.remote())
        worker_handles.extend(handles)
        logger.info(f"{sm_id} has {len(handles)} workers")

    if residual_handle:
        ray.get(residual_handle.register_workers.remote(worker_handles))
        logger.info(f"Registered {len(worker_handles)} workers to ResidualAgent")

    # -----------------------------
    # 5. Generate + distribute global context
    # -----------------------------
    if residual_handle:
        logger.info("Generating global context via ResidualAgent...")
        final_gc = ray.get(
            residual_handle.generate_and_distribute.remote(
                metadata,
                plan,
                wait_for_updates_seconds=6
            )
        )
        logger.info(f"Global context generated with keys: {list(final_gc.keys())}")
        
        # Explicitly ensure all SubMasters have the context
        logger.info("Explicitly broadcasting context to all SubMasters...")
        context_futures = []
        for sm_id, a in actors.items():
            fut = a.set_global_context.remote(final_gc)
            context_futures.append((sm_id, fut))
        
        # Wait for all SubMasters to confirm context received
        for sm_id, fut in context_futures:
            try:
                result = ray.get(fut)
                logger.info(f"{sm_id} context confirmation: {result}")
            except Exception as e:
                logger.error(f"{sm_id} failed to receive context: {e}")
        
        logger.info("Global context distribution completed")
        
        # Give workers a moment to process the context
        time.sleep(1)

    # -----------------------------
    # 6. Start SubMasters processing
    # -----------------------------
    logger.info("Starting SubMaster processing...")
    process_futures = {sid: a.process.remote() for sid, a in actors.items()}

    for sid, fut in process_futures.items():
        try:
            out = ray.get(fut)
            results[sid] = {"status": "ok", "output": out}
            logger.info(f"{sid} completed: {out.get('output', {}).get('context_usage', 'N/A')} context usage")
        except Exception as e:
            results[sid] = {"status": "error", "error": str(e)}
            logger.error(f"{sid} failed: {e}")

    return results