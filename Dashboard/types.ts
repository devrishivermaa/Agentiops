export enum AgentType {
  MASTER = "master",
  SUBMASTER = "submaster",
  WORKER = "worker",
  RESIDUAL = "residual",
  REDUCER = "reducer",
  // New Reducer Pipeline Agent Types
  REDUCER_SUBMASTER = "reducer_submaster",
  REDUCER_WORKER = "reducer_worker",
  REDUCER_RESIDUAL = "reducer_residual",
  MASTER_MERGER = "master_merger",
  PDF_GENERATOR = "pdf_generator",
}

export enum AgentStatus {
  SPAWNED = "spawned",
  INITIALIZING = "initializing",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
  WAITING = "waiting",
}

// Session workflow status
export enum SessionStatus {
  AWAITING_INTENT = "awaiting_intent",
  GENERATING_METADATA = "generating_metadata",
  AWAITING_APPROVAL = "awaiting_approval",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
}

export interface SessionState {
  sessionId: string | null;
  status: SessionStatus;
  fileName: string | null;
  filePath: string | null;
  pipelineId: string | null;
  metadata: Record<string, any> | null;
  metadataPath: string | null;
  highLevelIntent: string | null;
  documentContext: string | null;
  error: string | null;
}

export interface AgentNode {
  id: string;
  type: AgentType;
  status: AgentStatus;
  label: string;
  parentId?: string;
  metadata?: {
    role?: string;
    pages?: [number, number] | number; // Range for submaster, single for worker
    progress?: number;
    processingTime?: number;
    context?: string;
    summary?: string;
    entities?: string[];
    keywords?: string[];
    // Reducer pipeline metadata
    submasterCount?: number;
    numResults?: number;
    elapsedTime?: number;
    contextSize?: number;
    planSize?: number;
    resultSize?: number;
    pdfPath?: string;
    status?: string;
    current?: number;
    total?: number;
  };
  children?: string[]; // IDs of children
  events: AgentEventLog[]; // All events for this agent
  startTime?: number;
  endTime?: number;
}

export interface AgentEventLog {
  timestamp: Date;
  eventType: string;
  message: string;
  data?: Record<string, any>;
}

export interface AgentEvent {
  event_type: string;
  pipeline_id: string;
  timestamp: string;
  data: Record<string, any>;
  agent_id?: string;
  agent_type?: AgentType;
}

export interface PipelineState {
  id: string;
  status: "idle" | "running" | "completed" | "failed" | "paused";
  startTime?: number;
  endTime?: number;
  fileName?: string;
  currentStep: string;
}

export interface EventLogEntry {
  id: string;
  timestamp: Date;
  eventType: string;
  message: string;
  agentId?: string;
  severity: "info" | "success" | "warning" | "error";
}

// WebSocket Message Types - matches backend api/events.py
export const EventTypes = {
  // Pipeline lifecycle
  PIPELINE_STARTED: "pipeline.started",
  PIPELINE_STEP_STARTED: "pipeline.step_started",
  PIPELINE_STEP_COMPLETED: "pipeline.step_completed",
  PIPELINE_COMPLETED: "pipeline.completed",
  PIPELINE_FAILED: "pipeline.failed",

  // Master Agent
  MASTER_PLAN_GENERATING: "master.plan_generating",
  MASTER_PLAN_GENERATED: "master.plan_generated",
  MASTER_AWAITING_FEEDBACK: "master.awaiting_feedback",
  MASTER_PLAN_APPROVED: "master.plan_approved",

  // Residual Agent
  RESIDUAL_INITIALIZED: "residual.initialized",
  RESIDUAL_CONTEXT_GENERATING: "residual.context_generating",
  RESIDUAL_CONTEXT_GENERATED: "residual.context_generated",
  RESIDUAL_BROADCASTING: "residual.broadcasting",
  RESIDUAL_BROADCAST_COMPLETE: "residual.broadcast_complete",
  RESIDUAL_CONTEXT_ENHANCED: "residual.context_enhanced",
  RESIDUAL_UPDATE_RECEIVED: "residual.update_received",

  // SubMaster
  SUBMASTER_SPAWNED: "submaster.spawned",
  SUBMASTER_INITIALIZED: "submaster.initialized",
  SUBMASTER_PROCESSING: "submaster.processing",
  SUBMASTER_PROGRESS: "submaster.progress",
  SUBMASTER_COMPLETED: "submaster.completed",
  SUBMASTER_FAILED: "submaster.failed",
  SUBMASTER_CONTEXT_RECEIVED: "submaster.context_received",
  SUBMASTER_WORKERS_SPAWNED: "submaster.workers_spawned",

  // Worker
  WORKER_SPAWNED: "worker.spawned",
  WORKER_INITIALIZED: "worker.initialized",
  WORKER_PROCESSING: "worker.processing",
  WORKER_PAGE_STARTED: "worker.page_started",
  WORKER_PAGE_COMPLETED: "worker.page_completed",
  WORKER_COMPLETED: "worker.completed",
  WORKER_FAILED: "worker.failed",
  WORKER_CONTEXT_RECEIVED: "worker.context_received",

  // Reducer (basic aggregation)
  REDUCER_STARTED: "reducer.started",
  REDUCER_AGGREGATING: "reducer.aggregating",
  REDUCER_COMPLETED: "reducer.completed",

  // Reducer SubMaster (full reducer pipeline)
  REDUCER_SUBMASTER_STARTED: "reducer_submaster.started",
  REDUCER_SUBMASTER_PROCESSING: "reducer_submaster.processing",
  REDUCER_SUBMASTER_PROGRESS: "reducer_submaster.progress",
  REDUCER_SUBMASTER_COMPLETED: "reducer_submaster.completed",
  REDUCER_SUBMASTER_FAILED: "reducer_submaster.failed",

  // Reducer Worker
  REDUCER_WORKER_SPAWNED: "reducer_worker.spawned",
  REDUCER_WORKER_PROCESSING: "reducer_worker.processing",
  REDUCER_WORKER_COMPLETED: "reducer_worker.completed",

  // Reducer Residual Agent
  REDUCER_RESIDUAL_STARTED: "reducer_residual.started",
  REDUCER_RESIDUAL_CONTEXT_UPDATING: "reducer_residual.context_updating",
  REDUCER_RESIDUAL_CONTEXT_UPDATED: "reducer_residual.context_updated",
  REDUCER_RESIDUAL_PLAN_CREATING: "reducer_residual.plan_creating",
  REDUCER_RESIDUAL_PLAN_CREATED: "reducer_residual.plan_created",
  REDUCER_RESIDUAL_COMPLETED: "reducer_residual.completed",

  // Master Merger
  MASTER_MERGER_STARTED: "master_merger.started",
  MASTER_MERGER_SYNTHESIZING: "master_merger.synthesizing",
  MASTER_MERGER_EXECUTIVE_SUMMARY: "master_merger.executive_summary",
  MASTER_MERGER_DETAILED_SYNTHESIS: "master_merger.detailed_synthesis",
  MASTER_MERGER_INSIGHTS: "master_merger.insights",
  MASTER_MERGER_COMPLETED: "master_merger.completed",
  MASTER_MERGER_FAILED: "master_merger.failed",

  // PDF Generation
  PDF_GENERATION_STARTED: "pdf.generation_started",
  PDF_GENERATION_COMPLETED: "pdf.generation_completed",
  PDF_GENERATION_FAILED: "pdf.generation_failed",

  // System
  SYSTEM_STATS: "system.stats",
  ERROR: "error",
};

// ==================== Chat/RAG Types ====================

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: ChatSource[];
  isLoading?: boolean;
}

export interface ChatSource {
  text: string;
  score: number;
  doc_id: string;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  ragAvailable: boolean;
}
