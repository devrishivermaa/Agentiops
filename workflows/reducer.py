# workflows/reducer.py
"""
Reducer workflow: Aggregates SubMaster results into final output.
Emits events through API EventBus for real-time visualization.
"""

from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger("Reducer")


def _emit_event_safe(event_type_name, pipeline_id, data=None):
    """Safely emit events from Reducer"""
    try:
        from api.events import event_bus, EventType
        event_type = getattr(EventType, event_type_name, None)
        if event_type:
            event_bus.emit_simple(
                event_type,
                pipeline_id,
                data or {},
                agent_id="reducer",
                agent_type="reducer",
            )
    except Exception as e:
        logger.debug(f"Event emission skipped: {e}")


class Reducer:
    """
    Reducer aggregates and consolidates results from SubMasters.
    """
    
    def __init__(self, pipeline_id: str = None):
        self.pipeline_id = pipeline_id or "unknown"
        logger.info("Reducer initialized")
    
    def reduce(self, submaster_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate SubMaster results.
        
        Args:
            submaster_results: List of SubMaster outputs
            
        Returns:
            Aggregated final result
        """
        logger.info(f"Reducing {len(submaster_results)} SubMaster results")
        
        # Emit started event
        _emit_event_safe("REDUCER_STARTED", self.pipeline_id, {
            "num_results": len(submaster_results)
        })
        
        # Aggregate results with progress tracking
        combined_results = []
        total_pages = 0
        successful_results = 0
        failed_results = 0
        
        for idx, result in enumerate(submaster_results):
            # Emit progress event
            _emit_event_safe("REDUCER_AGGREGATING", self.pipeline_id, {
                "current": idx + 1,
                "total": len(submaster_results),
                "progress_percent": round(((idx + 1) / len(submaster_results)) * 100, 1)
            })
            
            if result.get("status") == "ok":
                successful_results += 1
                output = result.get("output", {})
                total_pages += output.get("output", {}).get("total_pages", 0)
                combined_results.append(output)
            else:
                failed_results += 1
                combined_results.append(result)
        
        # Build final result
        final_result = {
            "status": "completed",
            "num_submasters": len(submaster_results),
            "successful": successful_results,
            "failed": failed_results,
            "total_pages_processed": total_pages,
            "combined_results": combined_results
        }
        
        # Emit completed event
        _emit_event_safe("REDUCER_COMPLETED", self.pipeline_id, {
            "num_submasters": len(submaster_results),
            "successful": successful_results,
            "failed": failed_results,
            "total_pages": total_pages,
            "status": "completed"
        })
        
        logger.info(f"Reducer completed: {successful_results}/{len(submaster_results)} successful")
        
        return final_result
