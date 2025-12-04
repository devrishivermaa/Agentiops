export enum AgentType {
  MASTER = "master",
  SUBMASTER = "submaster",
  WORKER = "worker"
}

export enum AgentStatus {
  SPAWNED = "spawned",
  INITIALIZING = "initializing",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
  WAITING = "waiting"
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
  };
  children?: string[]; // IDs of children
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

// WebSocket Message Types
export const EventTypes = {
  PIPELINE_STARTED: "pipeline.started",
  PIPELINE_COMPLETED: "pipeline.completed",
  MASTER_PLAN_GENERATING: "master.plan_generating",
  MASTER_PLAN_GENERATED: "master.plan_generated",
  SUBMASTER_SPAWNED: "submaster.spawned",
  SUBMASTER_PROGRESS: "submaster.progress",
  SUBMASTER_COMPLETED: "submaster.completed",
  WORKER_SPAWNED: "worker.spawned",
  WORKER_PROCESSING: "worker.processing",
  WORKER_COMPLETED: "worker.completed",
  ERROR: "error"
};