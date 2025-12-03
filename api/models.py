"""
Pydantic models for the AgentOps API layer
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ==================== Enums ====================

class PipelineStatus(str, Enum):
    """Pipeline run status"""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Agent lifecycle status"""
    SPAWNED = "spawned"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    """Types of agents in the system"""
    MASTER = "master"
    SUBMASTER = "submaster"
    WORKER = "worker"
    RESIDUAL = "residual"


# ==================== Event Models ====================

class PipelineEvent(BaseModel):
    """Event emitted during pipeline execution"""
    event_type: str
    pipeline_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentEvent(BaseModel):
    """Event specific to an agent"""
    event_type: str
    pipeline_id: str
    agent_id: str
    agent_type: AgentType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


# ==================== Pipeline Models ====================

class PipelineRunCreate(BaseModel):
    """Request to start a new pipeline run"""
    file_path: str = Field(..., description="Path to the PDF file to process")
    user_notes: Optional[str] = Field(None, description="Optional user notes for processing")
    auto_approve: bool = Field(False, description="Auto-approve SubMaster plan without user feedback")
    max_parallel_submasters: Optional[int] = Field(None, ge=1, le=10, description="Max parallel SubMasters")
    num_workers_per_submaster: Optional[int] = Field(None, ge=1, le=10, description="Workers per SubMaster")


class PipelineRunFromMetadata(BaseModel):
    """Request to start pipeline from existing metadata"""
    metadata_path: str = Field(..., description="Path to metadata JSON file")
    auto_approve: bool = Field(False, description="Auto-approve SubMaster plan")


class PipelineApproval(BaseModel):
    """Request to approve/reject SubMaster plan"""
    approved: bool
    feedback: Optional[str] = Field(None, description="Optional feedback for plan modification")


class SubMasterPlanItem(BaseModel):
    """A single SubMaster in the execution plan"""
    submaster_id: str
    role: str
    assigned_sections: List[str]
    page_range: List[int]  # [start, end]
    estimated_workload: str


class SubMasterPlan(BaseModel):
    """Complete SubMaster execution plan"""
    status: str
    num_submasters: int
    distribution_strategy: str
    submasters: List[SubMasterPlanItem]


class PipelineRunResponse(BaseModel):
    """Response for pipeline run status"""
    pipeline_id: str
    status: PipelineStatus
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    progress_percent: float = 0.0
    num_submasters: Optional[int] = None
    num_workers: Optional[int] = None
    submaster_plan: Optional[SubMasterPlan] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineListResponse(BaseModel):
    """List of pipeline runs"""
    pipelines: List[PipelineRunResponse]
    total: int


# ==================== Agent Models ====================

class WorkerInfo(BaseModel):
    """Worker agent information"""
    worker_id: str
    submaster_id: str
    status: AgentStatus
    assigned_page: Optional[int] = None
    processing_time: Optional[float] = None


class SubMasterInfo(BaseModel):
    """SubMaster agent information"""
    submaster_id: str
    role: str
    status: AgentStatus
    assigned_sections: List[str]
    page_range: List[int]
    num_workers: int
    workers: List[WorkerInfo] = Field(default_factory=list)
    pages_processed: int = 0
    total_pages: int = 0
    progress_percent: float = 0.0


class MasterAgentInfo(BaseModel):
    """Master agent information"""
    agent_id: str = "master"
    status: AgentStatus
    plan_generated: bool = False
    plan_approved: bool = False


class AgentHierarchy(BaseModel):
    """Complete agent hierarchy for a pipeline"""
    pipeline_id: str
    master: Optional[MasterAgentInfo] = None
    submasters: List[SubMasterInfo] = Field(default_factory=list)
    total_workers: int = 0


class AgentListResponse(BaseModel):
    """Response containing agent information"""
    pipeline_id: str
    agents: AgentHierarchy


# ==================== Stats Models ====================

class RateLimitStats(BaseModel):
    """Rate limiter statistics"""
    requests_last_minute: int
    max_per_minute: int
    requests_last_day: int
    max_per_day: int
    percent_daily_quota_used: float


class SystemStats(BaseModel):
    """System resource statistics"""
    elapsed_time: float
    cpu_percent: float
    memory_mb: float
    memory_delta_mb: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OverallStats(BaseModel):
    """Combined statistics"""
    active_pipelines: int
    total_submasters: int
    total_workers: int
    rate_limit: Optional[RateLimitStats] = None
    system: Optional[SystemStats] = None


# ==================== Report Models ====================

class PageResult(BaseModel):
    """Result for a single page analysis"""
    page: int
    section: str
    char_count: int
    text_preview: str
    worker_id: str
    summary: str
    entities: List[str]
    keywords: List[str]
    key_points: List[str]
    technical_terms: List[str]
    status: str
    processing_time: float


class SubMasterResult(BaseModel):
    """Result from a SubMaster"""
    sm_id: str
    role: str
    assigned_sections: List[str]
    page_range: List[int]
    num_workers: int
    results: List[PageResult]
    total_pages: int
    total_chars: int
    total_entities: int
    total_keywords: int
    llm_successes: int
    llm_failures: int
    aggregate_summary: str
    elapsed_time: float


class PipelineResult(BaseModel):
    """Complete pipeline result"""
    pipeline_id: str
    file_name: str
    total_pages: int
    submasters: List[SubMasterResult]
    overall_summary: Optional[str] = None
    total_processing_time: float


# ==================== WebSocket Models ====================

class WSMessage(BaseModel):
    """WebSocket message wrapper"""
    type: str  # "event", "ping", "pong", "subscribe", "unsubscribe"
    payload: Optional[Dict[str, Any]] = None


class WSSubscribe(BaseModel):
    """WebSocket subscription request"""
    pipeline_id: str = "*"  # "*" for all pipelines


class WSEventPayload(BaseModel):
    """WebSocket event payload"""
    event_type: str
    pipeline_id: str
    timestamp: str
    data: Dict[str, Any]
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
