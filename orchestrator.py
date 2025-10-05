# orchestrator.py
import ray
from agents.sub_master import SubMaster
from utils.logger import get_logger

logger = get_logger("Orchestrator")

def start_ray_if_needed():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

def spawn_submasters_and_run(plan, metadata):
    start_ray_if_needed()
    actors = {}
    results = {}

    for sm in plan.get("submasters", []):
        sm_id = sm["submaster_id"]
        actor = SubMaster.options(name=sm_id).remote(sm, metadata)
        actors[sm_id] = actor
        logger.info(f"Spawned {sm_id} (role={sm.get('role')})")

    init_futs = [a.initialize.remote() for a in actors.values()]
    ray.get(init_futs)

    proc_futs = {sm_id: a.process.remote() for sm_id, a in actors.items()}

    for sm_id, fut in proc_futs.items():
        try:
            res = ray.get(fut)
            results[sm_id] = {"status": "ok", "output": res}
            logger.info(f"{sm_id} finished.")
        except Exception as e:
            results[sm_id] = {"status": "error", "error": str(e)}
            logger.error(f"{sm_id} failed: {e}")

    return results
