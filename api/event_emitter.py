"""
Event emitter helper for Ray actors.

Since Ray actors run in separate processes, they cannot share the EventBus
singleton with the main FastAPI process. This module provides a helper
function that emits events via HTTP to the FastAPI internal endpoint.
"""

import logging
import os
import requests
from typing import Any, Dict, Optional
from threading import Thread
from queue import Queue

logger = logging.getLogger(__name__)

# API base URL - can be configured via environment
API_BASE_URL = os.getenv("AGENTOPS_API_URL", "http://localhost:8000")

# Event queue for async emission
_event_queue: Queue = Queue()
_emit_thread_started = False


def _emit_worker():
    """Background thread to emit events asynchronously"""
    while True:
        try:
            event_data = _event_queue.get()
            if event_data is None:
                break
                
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/internal/emit",
                    json=event_data,
                    timeout=5,
                )
                if response.status_code != 200:
                    logger.debug(f"Event emission failed: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Event emission request failed: {e}")
        except Exception as e:
            logger.debug(f"Event emission worker error: {e}")


def _ensure_emit_thread():
    """Start the emit worker thread if not already started"""
    global _emit_thread_started
    if not _emit_thread_started:
        _emit_thread_started = True
        thread = Thread(target=_emit_worker, daemon=True)
        thread.start()


def emit_event(
    event_type: str,
    pipeline_id: str,
    data: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    async_emit: bool = True,
):
    """
    Emit an event to the API server.
    
    Args:
        event_type: The event type name (e.g., "WORKER_SPAWNED")
        pipeline_id: The pipeline ID
        data: Event data dictionary
        agent_id: The agent's ID
        agent_type: The agent type (e.g., "worker", "submaster")
        async_emit: If True, emit asynchronously in background thread
    """
    event_data = {
        "event_type": event_type,
        "pipeline_id": pipeline_id,
        "data": data or {},
        "agent_id": agent_id,
        "agent_type": agent_type,
    }
    
    if async_emit:
        _ensure_emit_thread()
        _event_queue.put(event_data)
    else:
        # Synchronous emit
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/internal/emit",
                json=event_data,
                timeout=5,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Event emission failed: {e}")
            return False


def emit_event_safe(event_type, pipeline_id, agent_id, agent_type, data=None):
    """
    Safe wrapper for emitting events from Ray actors.
    
    This function catches all exceptions and logs them as debug messages.
    It's designed to be used in Ray actors where event emission should not
    block or fail the main processing logic.
    """
    try:
        emit_event(
            event_type=event_type,
            pipeline_id=pipeline_id,
            data=data,
            agent_id=agent_id,
            agent_type=agent_type,
            async_emit=True,
        )
    except Exception as e:
        logger.debug(f"Event emission skipped: {e}")
