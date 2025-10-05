# agents/sub_master.py
import time, random, uuid
import ray
from utils.logger import get_logger

logger = get_logger("SubMaster")


@ray.remote
class SubMaster:
    def __init__(self, plan_piece, metadata):
        self.sm_id = plan_piece.get("submaster_id", f"SM-{uuid.uuid4().hex[:6].upper()}")
        self.role = plan_piece.get("role", "generic")
        self.sections = plan_piece.get("assigned_sections", [])
        self.pages = plan_piece.get("page_range", [1, 1])
        self.meta = metadata
        self.status = "initialized"
        logger.info(f"[{self.sm_id}] Init: role={self.role}, pages={self.pages}")

    def initialize(self):
        self.status = "ready"
        return {"sm_id": self.sm_id, "status": "ready"}

    def process(self):
        self.status = "running"
        output = []

        # Handle multiple page ranges
        if len(self.pages) % 2 != 0:
            raise ValueError(f"Invalid page_range for {self.sm_id}: {self.pages}")

        # Iterate over each (start, end) pair
        for i in range(0, len(self.pages), 2):
            start, end = self.pages[i], self.pages[i + 1]
            for page in range(start, end + 1):
                time.sleep(random.uniform(0.05, 0.15))  # simulate processing
                output.append({"page": page, "summary": f"Processed text of page {page}"})

        self.status = "completed"
        logger.info(f"[{self.sm_id}] Completed {len(output)} pages.")

        return {"sm_id": self.sm_id, "role": self.role, "results": output}
