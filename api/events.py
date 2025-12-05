"""
EventBus: Central event emission and subscription system for AgentOps
Supports both sync and async subscribers, with WebSocket broadcast capability
"""

import asyncio
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """All event types emitted by the AgentOps engine"""
    
    # Pipeline lifecycle events
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_STEP_STARTED = "pipeline.step_started"
    PIPELINE_STEP_COMPLETED = "pipeline.step_completed"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    
    # Master Agent events
    MASTER_PLAN_GENERATING = "master.plan_generating"
    MASTER_PLAN_GENERATED = "master.plan_generated"
    MASTER_AWAITING_FEEDBACK = "master.awaiting_feedback"
    MASTER_PLAN_APPROVED = "master.plan_approved"
    
    # ResidualAgent events
    RESIDUAL_INITIALIZED = "residual.initialized"
    RESIDUAL_CONTEXT_GENERATING = "residual.context_generating"
    RESIDUAL_CONTEXT_GENERATED = "residual.context_generated"
    RESIDUAL_BROADCASTING = "residual.broadcasting"
    RESIDUAL_BROADCAST_COMPLETE = "residual.broadcast_complete"
    RESIDUAL_CONTEXT_ENHANCED = "residual.context_enhanced"
    RESIDUAL_UPDATE_RECEIVED = "residual.update_received"
    
    # SubMaster events
    SUBMASTER_SPAWNED = "submaster.spawned"
    SUBMASTER_INITIALIZED = "submaster.initialized"
    SUBMASTER_PROCESSING = "submaster.processing"
    SUBMASTER_PROGRESS = "submaster.progress"
    SUBMASTER_COMPLETED = "submaster.completed"
    SUBMASTER_FAILED = "submaster.failed"
    SUBMASTER_CONTEXT_RECEIVED = "submaster.context_received"
    SUBMASTER_WORKERS_SPAWNED = "submaster.workers_spawned"
    
    # Worker events
    WORKER_SPAWNED = "worker.spawned"
    WORKER_INITIALIZED = "worker.initialized"
    WORKER_PROCESSING = "worker.processing"
    WORKER_PAGE_STARTED = "worker.page_started"
    WORKER_PAGE_COMPLETED = "worker.page_completed"
    WORKER_COMPLETED = "worker.completed"
    WORKER_FAILED = "worker.failed"
    WORKER_CONTEXT_RECEIVED = "worker.context_received"
    
    # Reducer events (basic aggregation)
    REDUCER_STARTED = "reducer.started"
    REDUCER_AGGREGATING = "reducer.aggregating"
    REDUCER_COMPLETED = "reducer.completed"
    
    # Reducer SubMaster events (full reducer pipeline)
    REDUCER_SUBMASTER_STARTED = "reducer_submaster.started"
    REDUCER_SUBMASTER_PROCESSING = "reducer_submaster.processing"
    REDUCER_SUBMASTER_PROGRESS = "reducer_submaster.progress"
    REDUCER_SUBMASTER_COMPLETED = "reducer_submaster.completed"
    REDUCER_SUBMASTER_FAILED = "reducer_submaster.failed"
    
    # Reducer Worker events
    REDUCER_WORKER_SPAWNED = "reducer_worker.spawned"
    REDUCER_WORKER_PROCESSING = "reducer_worker.processing"
    REDUCER_WORKER_COMPLETED = "reducer_worker.completed"
    
    # Reducer Residual Agent events
    REDUCER_RESIDUAL_STARTED = "reducer_residual.started"
    REDUCER_RESIDUAL_CONTEXT_UPDATING = "reducer_residual.context_updating"
    REDUCER_RESIDUAL_CONTEXT_UPDATED = "reducer_residual.context_updated"
    REDUCER_RESIDUAL_PLAN_CREATING = "reducer_residual.plan_creating"
    REDUCER_RESIDUAL_PLAN_CREATED = "reducer_residual.plan_created"
    REDUCER_RESIDUAL_COMPLETED = "reducer_residual.completed"
    
    # Master Merger events
    MASTER_MERGER_STARTED = "master_merger.started"
    MASTER_MERGER_SYNTHESIZING = "master_merger.synthesizing"
    MASTER_MERGER_EXECUTIVE_SUMMARY = "master_merger.executive_summary"
    MASTER_MERGER_DETAILED_SYNTHESIS = "master_merger.detailed_synthesis"
    MASTER_MERGER_INSIGHTS = "master_merger.insights"
    MASTER_MERGER_COMPLETED = "master_merger.completed"
    MASTER_MERGER_FAILED = "master_merger.failed"
    
    # PDF Generation events
    PDF_GENERATION_STARTED = "pdf.generation_started"
    PDF_GENERATION_COMPLETED = "pdf.generation_completed"
    PDF_GENERATION_FAILED = "pdf.generation_failed"
    
    # System events
    SYSTEM_STATS = "system.stats"
    RATE_LIMIT_WARNING = "ratelimit.warning"
    RATE_LIMIT_QUOTA_REACHED = "ratelimit.quota_reached"


@dataclass
class Event:
    """Base event structure for all AgentOps events"""
    event_type: EventType
    pipeline_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None  # "master", "submaster", "worker"
    
    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Recursively convert non-JSON-serializable types to strings"""
        if isinstance(obj, dict):
            return {k: Event._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Event._sanitize_for_json(item) for item in obj]
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'ObjectId':
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        else:
            return obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization"""
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp,
            "data": Event._sanitize_for_json(self.data),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())


class EventBus:
    """
    Central event bus for the AgentOps engine.
    
    Supports:
    - Sync subscribers (called in separate thread to avoid blocking)
    - Async subscribers (for WebSocket broadcasting)
    - Event history for late-joining clients
    - Pipeline-specific subscriptions
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single EventBus instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Subscribers: event_type -> list of callbacks
        self._sync_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # WebSocket connections: pipeline_id -> set of queues
        self._ws_queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        
        # Event history per pipeline (limited to last 1000 events)
        self._event_history: Dict[str, List[Event]] = defaultdict(list)
        self._max_history = 1000
        
        # Active pipelines tracking
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Thread pool for sync subscribers
        self._event_queue = Queue()
        self._worker_thread = threading.Thread(target=self._process_sync_events, daemon=True)
        self._worker_thread.start()
        
        # Async event loop reference (set when FastAPI starts)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        logger.info("EventBus initialized")
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the async event loop for broadcasting to WebSocket clients"""
        self._loop = loop
    
    def subscribe(self, event_type: EventType, callback: Callable, is_async: bool = False):
        """
        Subscribe to an event type.
        
        Args:
            event_type: The event type to subscribe to
            callback: Function to call when event is emitted
            is_async: Whether callback is async (coroutine)
        """
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if is_async:
            self._async_subscribers[key].append(callback)
        else:
            self._sync_subscribers[key].append(callback)
        logger.debug(f"Subscribed to {key} (async={is_async})")
    
    def subscribe_all(self, callback: Callable, is_async: bool = False):
        """Subscribe to all event types"""
        for event_type in EventType:
            self.subscribe(event_type, callback, is_async)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Remove a subscriber"""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if callback in self._sync_subscribers[key]:
            self._sync_subscribers[key].remove(callback)
        if callback in self._async_subscribers[key]:
            self._async_subscribers[key].remove(callback)
    
    def emit(self, event: Event):
        """
        Emit an event to all subscribers.
        
        This is the main method called by agents to publish events.
        Thread-safe and non-blocking for the caller.
        """
        # Store in history
        self._store_event(event)
        
        # Queue for sync subscribers (processed in worker thread)
        self._event_queue.put(event)
        
        # Broadcast to WebSocket clients
        self._broadcast_to_ws(event)
        
        # Handle async subscribers
        self._notify_async_subscribers(event)
        
        logger.debug(f"Event emitted: {event.event_type} for pipeline {event.pipeline_id}")
    
    def emit_simple(
        self,
        event_type: EventType,
        pipeline_id: str,
        data: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ):
        """Convenience method to emit events without creating Event object"""
        event = Event(
            event_type=event_type,
            pipeline_id=pipeline_id,
            data=data or {},
            agent_id=agent_id,
            agent_type=agent_type,
        )
        self.emit(event)
    
    def _store_event(self, event: Event):
        """Store event in history, maintaining max limit"""
        history = self._event_history[event.pipeline_id]
        history.append(event)
        if len(history) > self._max_history:
            self._event_history[event.pipeline_id] = history[-self._max_history:]
    
    def _process_sync_events(self):
        """Worker thread to process sync subscriber callbacks"""
        while True:
            try:
                event = self._event_queue.get()
                key = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
                
                for callback in self._sync_subscribers.get(key, []):
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in sync subscriber: {e}")
                
                # Also call wildcard subscribers (subscribed with "*")
                for callback in self._sync_subscribers.get("*", []):
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in wildcard subscriber: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
    
    def _notify_async_subscribers(self, event: Event):
        """Notify async subscribers using the event loop"""
        if not self._loop:
            return
            
        key = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
        
        for callback in self._async_subscribers.get(key, []):
            try:
                asyncio.run_coroutine_threadsafe(callback(event), self._loop)
            except Exception as e:
                logger.error(f"Error in async subscriber: {e}")
    
    def _broadcast_to_ws(self, event: Event):
        """Broadcast event to WebSocket clients subscribed to this pipeline"""
        if not self._loop:
            return
            
        pipeline_id = event.pipeline_id
        
        # Broadcast to pipeline-specific subscribers
        for queue in self._ws_queues.get(pipeline_id, set()):
            try:
                asyncio.run_coroutine_threadsafe(queue.put(event), self._loop)
            except Exception as e:
                logger.error(f"Error broadcasting to WS queue: {e}")
        
        # Also broadcast to "all" subscribers
        for queue in self._ws_queues.get("*", set()):
            try:
                asyncio.run_coroutine_threadsafe(queue.put(event), self._loop)
            except Exception as e:
                logger.error(f"Error broadcasting to WS queue: {e}")
    
    def register_ws_client(self, pipeline_id: str = "*") -> asyncio.Queue:
        """
        Register a WebSocket client to receive events.
        
        Args:
            pipeline_id: Pipeline to subscribe to, or "*" for all pipelines
            
        Returns:
            Queue that will receive Event objects
        """
        queue = asyncio.Queue()
        self._ws_queues[pipeline_id].add(queue)
        logger.info(f"WebSocket client registered for pipeline: {pipeline_id}")
        return queue
    
    def unregister_ws_client(self, queue: asyncio.Queue, pipeline_id: str = "*"):
        """Remove a WebSocket client"""
        if pipeline_id in self._ws_queues:
            self._ws_queues[pipeline_id].discard(queue)
            logger.info(f"WebSocket client unregistered from pipeline: {pipeline_id}")
    
    def get_history(self, pipeline_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history for a pipeline"""
        events = self._event_history.get(pipeline_id, [])[-limit:]
        return [e.to_dict() for e in events]
    
    def register_pipeline(self, pipeline_id: str, metadata: Dict[str, Any]):
        """Register an active pipeline"""
        self._active_pipelines[pipeline_id] = {
            "id": pipeline_id,
            "started_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "status": "running",
        }
    
    def update_pipeline_status(self, pipeline_id: str, status: str, result: Optional[Dict] = None):
        """Update pipeline status"""
        if pipeline_id in self._active_pipelines:
            self._active_pipelines[pipeline_id]["status"] = status
            if result:
                self._active_pipelines[pipeline_id]["result"] = result
    
    def get_active_pipelines(self) -> List[Dict[str, Any]]:
        """Get list of active pipelines"""
        return list(self._active_pipelines.values())
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline info by ID"""
        return self._active_pipelines.get(pipeline_id)
    
    def clear_pipeline(self, pipeline_id: str):
        """Clear pipeline data (call when pipeline completes)"""
        if pipeline_id in self._active_pipelines:
            del self._active_pipelines[pipeline_id]
        if pipeline_id in self._event_history:
            del self._event_history[pipeline_id]
        if pipeline_id in self._ws_queues:
            del self._ws_queues[pipeline_id]


# Global singleton instance
event_bus = EventBus()


# Convenience functions for emitting events from agents
def emit_pipeline_started(pipeline_id: str, file_path: str, metadata: Dict[str, Any]):
    """Emit pipeline started event"""
    event_bus.register_pipeline(pipeline_id, metadata)
    event_bus.emit_simple(
        EventType.PIPELINE_STARTED,
        pipeline_id,
        {"file_path": file_path, "metadata": metadata},
    )


def emit_pipeline_step(pipeline_id: str, step: str, status: str, data: Optional[Dict] = None):
    """Emit pipeline step event"""
    event_type = EventType.PIPELINE_STEP_STARTED if status == "started" else EventType.PIPELINE_STEP_COMPLETED
    event_bus.emit_simple(event_type, pipeline_id, {"step": step, "status": status, **(data or {})})


def emit_pipeline_completed(pipeline_id: str, result: Dict[str, Any]):
    """Emit pipeline completed event"""
    event_bus.update_pipeline_status(pipeline_id, "completed", result)
    event_bus.emit_simple(EventType.PIPELINE_COMPLETED, pipeline_id, result)


def emit_pipeline_failed(pipeline_id: str, error: str, step: Optional[str] = None):
    """Emit pipeline failed event"""
    event_bus.update_pipeline_status(pipeline_id, "failed", {"error": error})
    event_bus.emit_simple(EventType.PIPELINE_FAILED, pipeline_id, {"error": error, "step": step})


def emit_agent_event(
    event_type: EventType,
    pipeline_id: str,
    agent_id: str,
    agent_type: str,
    data: Optional[Dict] = None,
):
    """Emit agent-specific event"""
    event_bus.emit_simple(
        event_type,
        pipeline_id,
        data or {},
        agent_id=agent_id,
        agent_type=agent_type,
    )


def emit_submaster_progress(
    pipeline_id: str,
    submaster_id: str,
    current_page: int,
    total_pages: int,
    worker_id: Optional[str] = None,
):
    """Emit SubMaster progress event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_PROGRESS,
        pipeline_id,
        {
            "current_page": current_page,
            "total_pages": total_pages,
            "progress_percent": round((current_page / total_pages) * 100, 1) if total_pages > 0 else 0,
            "worker_id": worker_id,
        },
        agent_id=submaster_id,
        agent_type="submaster",
    )


def emit_worker_event(
    event_type: EventType,
    pipeline_id: str,
    worker_id: str,
    submaster_id: str,
    data: Optional[Dict] = None,
):
    """Emit worker event with parent submaster context"""
    event_data = {"submaster_id": submaster_id, **(data or {})}
    event_bus.emit_simple(
        event_type,
        pipeline_id,
        event_data,
        agent_id=worker_id,
        agent_type="worker",
    )


# ResidualAgent event helpers
def emit_residual_event(
    event_type: EventType,
    pipeline_id: str,
    residual_id: str,
    data: Optional[Dict] = None,
):
    """Emit ResidualAgent event"""
    event_bus.emit_simple(
        event_type,
        pipeline_id,
        data or {},
        agent_id=residual_id,
        agent_type="residual",
    )


def emit_residual_context_generating(pipeline_id: str, residual_id: str):
    """Emit context generation started event"""
    emit_residual_event(
        EventType.RESIDUAL_CONTEXT_GENERATING,
        pipeline_id,
        residual_id,
        {"status": "generating"}
    )


def emit_residual_context_generated(pipeline_id: str, residual_id: str, context_keys: list):
    """Emit context generated event"""
    emit_residual_event(
        EventType.RESIDUAL_CONTEXT_GENERATED,
        pipeline_id,
        residual_id,
        {"context_keys": context_keys, "status": "generated"}
    )


def emit_residual_broadcasting(pipeline_id: str, residual_id: str, target: str, count: int):
    """Emit broadcasting event"""
    emit_residual_event(
        EventType.RESIDUAL_BROADCASTING,
        pipeline_id,
        residual_id,
        {"target": target, "count": count}
    )


def emit_residual_broadcast_complete(pipeline_id: str, residual_id: str, target: str):
    """Emit broadcast complete event"""
    emit_residual_event(
        EventType.RESIDUAL_BROADCAST_COMPLETE,
        pipeline_id,
        residual_id,
        {"target": target, "status": "complete"}
    )


def emit_residual_update_received(pipeline_id: str, residual_id: str, source: str, author: str):
    """Emit update received event"""
    emit_residual_event(
        EventType.RESIDUAL_UPDATE_RECEIVED,
        pipeline_id,
        residual_id,
        {"source": source, "author": author}
    )


# Reducer event helpers
def emit_reducer_started(pipeline_id: str, num_results: int):
    """Emit reducer started event"""
    event_bus.emit_simple(
        EventType.REDUCER_STARTED,
        pipeline_id,
        {"num_results": num_results},
        agent_id="reducer",
        agent_type="reducer",
    )


def emit_reducer_aggregating(pipeline_id: str, current: int, total: int):
    """Emit reducer aggregating event"""
    event_bus.emit_simple(
        EventType.REDUCER_AGGREGATING,
        pipeline_id,
        {"current": current, "total": total, "progress_percent": round((current / total) * 100, 1) if total > 0 else 0},
        agent_id="reducer",
        agent_type="reducer",
    )


def emit_reducer_completed(pipeline_id: str, result_summary: Dict):
    """Emit reducer completed event"""
    event_bus.emit_simple(
        EventType.REDUCER_COMPLETED,
        pipeline_id,
        result_summary,
        agent_id="reducer",
        agent_type="reducer",
    )


# SubMaster spawning events
def emit_submaster_spawned(pipeline_id: str, submaster_id: str, role: str, sections: list, page_range: list):
    """Emit submaster spawned event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_SPAWNED,
        pipeline_id,
        {"role": role, "sections": sections, "page_range": page_range},
        agent_id=submaster_id,
        agent_type="submaster",
    )


def emit_submaster_initialized(pipeline_id: str, submaster_id: str, num_workers: int):
    """Emit submaster initialized event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_INITIALIZED,
        pipeline_id,
        {"num_workers": num_workers, "status": "ready"},
        agent_id=submaster_id,
        agent_type="submaster",
    )


def emit_submaster_processing(pipeline_id: str, submaster_id: str, num_pages: int):
    """Emit submaster processing event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_PROCESSING,
        pipeline_id,
        {"num_pages": num_pages, "status": "processing"},
        agent_id=submaster_id,
        agent_type="submaster",
    )


def emit_submaster_completed(pipeline_id: str, submaster_id: str, results_count: int, elapsed: float):
    """Emit submaster completed event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_COMPLETED,
        pipeline_id,
        {"results_count": results_count, "elapsed_seconds": elapsed, "status": "completed"},
        agent_id=submaster_id,
        agent_type="submaster",
    )


def emit_submaster_failed(pipeline_id: str, submaster_id: str, error: str):
    """Emit submaster failed event"""
    event_bus.emit_simple(
        EventType.SUBMASTER_FAILED,
        pipeline_id,
        {"error": error, "status": "failed"},
        agent_id=submaster_id,
        agent_type="submaster",
    )


# Worker spawning and processing events
def emit_worker_spawned(pipeline_id: str, worker_id: str, submaster_id: str):
    """Emit worker spawned event"""
    emit_worker_event(
        EventType.WORKER_SPAWNED,
        pipeline_id,
        worker_id,
        submaster_id,
        {"status": "spawned"}
    )


def emit_worker_initialized(pipeline_id: str, worker_id: str, submaster_id: str):
    """Emit worker initialized event"""
    emit_worker_event(
        EventType.WORKER_INITIALIZED,
        pipeline_id,
        worker_id,
        submaster_id,
        {"status": "ready"}
    )


def emit_worker_page_started(pipeline_id: str, worker_id: str, submaster_id: str, page: int):
    """Emit worker page processing started event"""
    emit_worker_event(
        EventType.WORKER_PAGE_STARTED,
        pipeline_id,
        worker_id,
        submaster_id,
        {"page": page, "status": "processing"}
    )


def emit_worker_page_completed(pipeline_id: str, worker_id: str, submaster_id: str, page: int, from_cache: bool):
    """Emit worker page processing completed event"""
    emit_worker_event(
        EventType.WORKER_PAGE_COMPLETED,
        pipeline_id,
        worker_id,
        submaster_id,
        {"page": page, "from_cache": from_cache, "status": "completed"}
    )
