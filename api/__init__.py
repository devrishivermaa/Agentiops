"""
API Layer for AgentOps Engine
Provides REST endpoints and WebSocket streaming for real-time agent monitoring
"""

from .events import (
    EventBus,
    event_bus,
    EventType,
    Event,
    emit_pipeline_started,
    emit_pipeline_step,
    emit_pipeline_completed,
    emit_pipeline_failed,
    emit_agent_event,
    emit_submaster_progress,
    emit_worker_event,
)
from .models import (
    PipelineEvent,
    AgentEvent,
    PipelineStatus,
    AgentStatus,
    AgentType,
    PipelineRunCreate,
    PipelineRunResponse,
    PipelineListResponse,
)
from .pipeline_manager import pipeline_manager, PipelineManager

__all__ = [
    # Events
    "EventBus",
    "event_bus",
    "EventType",
    "Event",
    "emit_pipeline_started",
    "emit_pipeline_step",
    "emit_pipeline_completed",
    "emit_pipeline_failed",
    "emit_agent_event",
    "emit_submaster_progress",
    "emit_worker_event",
    # Models
    "PipelineEvent",
    "AgentEvent",
    "PipelineStatus",
    "AgentStatus",
    "AgentType",
    "PipelineRunCreate",
    "PipelineRunResponse",
    "PipelineListResponse",
    # Pipeline Manager
    "pipeline_manager",
    "PipelineManager",
]
