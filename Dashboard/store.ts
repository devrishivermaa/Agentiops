import { create } from 'zustand';
import { AgentNode, AgentStatus, AgentType, EventLogEntry, PipelineState, AgentEvent, EventTypes } from './types';

interface AppState {
  // Connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // Pipeline
  pipeline: PipelineState;
  setPipeline: (pipeline: Partial<PipelineState>) => void;
  resetPipeline: () => void;

  // Agents
  agents: Record<string, AgentNode>;
  addAgent: (agent: AgentNode) => void;
  updateAgent: (id: string, updates: Partial<AgentNode>) => void;

  // Events
  events: EventLogEntry[];
  addEvent: (event: EventLogEntry) => void;
  
  // Actions
  processIncomingEvent: (event: AgentEvent) => void;
}

export const useStore = create<AppState>((set, get) => ({
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  pipeline: {
    id: "",
    status: "idle",
    currentStep: "Ready",
  },
  setPipeline: (updates) => set((state) => ({ pipeline: { ...state.pipeline, ...updates } })),
  resetPipeline: () => set({
    pipeline: { id: "", status: "idle", currentStep: "Ready" },
    agents: {},
    events: []
  }),

  agents: {},
  addAgent: (agent) => set((state) => ({ agents: { ...state.agents, [agent.id]: agent } })),
  updateAgent: (id, updates) => set((state) => {
    const agent = state.agents[id];
    if (!agent) return state;
    return { agents: { ...state.agents, [id]: { ...agent, ...updates } } };
  }),

  events: [],
  addEvent: (event) => set((state) => ({ events: [...state.events, event] })),

  processIncomingEvent: (event: AgentEvent) => {
    const { addEvent, updateAgent, addAgent, setPipeline } = get();
    const timestamp = new Date();

    // Log the event
    let severity: "info" | "success" | "warning" | "error" = "info";
    let message = event.event_type;

    if (event.event_type.includes("failed") || event.event_type.includes("error")) severity = "error";
    if (event.event_type.includes("completed")) severity = "success";

    // Format readable message based on type
    if (event.event_type === EventTypes.PIPELINE_STARTED) message = `Pipeline started for ${event.data.file}`;
    if (event.event_type === EventTypes.SUBMASTER_SPAWNED) message = `Spawned SubMaster: ${event.data.role}`;
    if (event.event_type === EventTypes.WORKER_SPAWNED) message = `Spawned Worker for page ${event.data.page}`;

    addEvent({
      id: Math.random().toString(36).substring(7),
      timestamp,
      eventType: event.event_type,
      message,
      agentId: event.agent_id,
      severity
    });

    // Handle State Logic
    switch (event.event_type) {
      case EventTypes.PIPELINE_STARTED:
        setPipeline({ 
          id: event.pipeline_id, 
          status: "running", 
          fileName: event.data.file,
          startTime: Date.now() 
        });
        break;

      case EventTypes.PIPELINE_COMPLETED:
        setPipeline({ status: "completed", endTime: Date.now(), currentStep: "Finished" });
        break;

      case EventTypes.MASTER_PLAN_GENERATING:
        // Ensure master exists
        if (!get().agents["master"]) {
          addAgent({
            id: "master",
            type: AgentType.MASTER,
            status: AgentStatus.PROCESSING,
            label: "Master Orchestrator",
            metadata: { role: "Planner" }
          });
        } else {
            updateAgent("master", { status: AgentStatus.PROCESSING });
        }
        setPipeline({ currentStep: "Generating Plan" });
        break;

      case EventTypes.MASTER_PLAN_GENERATED:
        updateAgent("master", { status: AgentStatus.COMPLETED });
        setPipeline({ currentStep: "Spawning SubMasters" });
        break;

      case EventTypes.SUBMASTER_SPAWNED:
        addAgent({
          id: event.agent_id!,
          type: AgentType.SUBMASTER,
          status: AgentStatus.INITIALIZING,
          label: event.data.role || "SubMaster",
          parentId: "master",
          metadata: { 
            role: event.data.role,
            pages: event.data.page_range,
            progress: 0 
          }
        });
        break;

      case EventTypes.WORKER_SPAWNED:
        addAgent({
          id: event.agent_id!,
          type: AgentType.WORKER,
          status: AgentStatus.WAITING,
          label: `Worker P${event.data.page}`,
          parentId: event.data.submaster_id,
          metadata: {
            pages: event.data.page,
            progress: 0
          }
        });
        break;

      case EventTypes.WORKER_PROCESSING:
        updateAgent(event.agent_id!, { status: AgentStatus.PROCESSING });
        // Also set SubMaster to processing if not already
        const worker = get().agents[event.agent_id!];
        if (worker && worker.parentId) {
             updateAgent(worker.parentId, { status: AgentStatus.PROCESSING });
        }
        break;

      case EventTypes.WORKER_COMPLETED:
        updateAgent(event.agent_id!, { status: AgentStatus.COMPLETED });
        break;

      case EventTypes.SUBMASTER_PROGRESS:
        updateAgent(event.agent_id!, { 
            metadata: { ...get().agents[event.agent_id!]?.metadata, progress: event.data.percent } 
        });
        break;

      case EventTypes.SUBMASTER_COMPLETED:
        updateAgent(event.agent_id!, { status: AgentStatus.COMPLETED, metadata: { progress: 100 } });
        break;
    }
  }
}));