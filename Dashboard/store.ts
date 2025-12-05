import { create } from "zustand";
import {
  AgentNode,
  AgentStatus,
  AgentType,
  EventLogEntry,
  PipelineState,
  AgentEvent,
  EventTypes,
  SessionState,
  SessionStatus,
  AgentEventLog,
  ChatMessage,
  ChatState,
  ChatSource,
} from "./types";

// API Base URL
const API_BASE = "http://localhost:8000";

interface AppState {
  // Connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // Session (new workflow)
  session: SessionState;
  setSession: (session: Partial<SessionState>) => void;
  resetSession: () => void;

  // Pipeline
  pipeline: PipelineState;
  setPipeline: (pipeline: Partial<PipelineState>) => void;
  resetPipeline: () => void;

  // Agents
  agents: Record<string, AgentNode>;
  addAgent: (agent: AgentNode) => void;
  updateAgent: (id: string, updates: Partial<AgentNode>) => void;
  addAgentEvent: (agentId: string, event: AgentEventLog) => void;

  // Events
  events: EventLogEntry[];
  addEvent: (event: EventLogEntry) => void;

  // Download state
  outputPaths: {
    reportPath?: string;
    jsonPath?: string;
  };
  setOutputPaths: (paths: { reportPath?: string; jsonPath?: string }) => void;

  // Actions
  processIncomingEvent: (event: AgentEvent) => void;

  // API Actions
  uploadFile: (file: File) => Promise<{ sessionId: string; fileName: string }>;
  submitIntent: (
    sessionId: string,
    intent: string,
    context?: string
  ) => Promise<any>;
  updateMetadata: (
    sessionId: string,
    metadata: Record<string, any>
  ) => Promise<void>;
  approveAndProcess: (
    sessionId: string,
    approved: boolean,
    modifiedMetadata?: Record<string, any>
  ) => Promise<any>;
  downloadReport: () => Promise<void>;
  downloadJson: () => Promise<void>;

  // Chat/RAG state
  chat: ChatState;
  sendChatMessage: (question: string) => Promise<void>;
  clearChat: () => void;
  loadChatHistory: () => Promise<void>;
}

const initialSession: SessionState = {
  sessionId: null,
  status: SessionStatus.AWAITING_INTENT,
  fileName: null,
  filePath: null,
  pipelineId: null,
  metadata: null,
  metadataPath: null,
  highLevelIntent: null,
  documentContext: null,
  error: null,
};

export const useStore = create<AppState>((set, get) => ({
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  // Session state
  session: { ...initialSession },
  setSession: (updates) =>
    set((state) => ({ session: { ...state.session, ...updates } })),
  resetSession: () => set({ session: { ...initialSession } }),

  pipeline: {
    id: "",
    status: "idle",
    currentStep: "Ready",
  },
  setPipeline: (updates) =>
    set((state) => ({ pipeline: { ...state.pipeline, ...updates } })),
  resetPipeline: () =>
    set({
      pipeline: { id: "", status: "idle", currentStep: "Ready" },
      agents: {},
      events: [],
      outputPaths: {},
    }),

  agents: {},
  addAgent: (agent) =>
    set((state) => ({ agents: { ...state.agents, [agent.id]: agent } })),
  updateAgent: (id, updates) =>
    set((state) => {
      const agent = state.agents[id];
      if (!agent) return state;
      return { agents: { ...state.agents, [id]: { ...agent, ...updates } } };
    }),
  addAgentEvent: (agentId, event) =>
    set((state) => {
      const agent = state.agents[agentId];
      if (!agent) return state;
      return {
        agents: {
          ...state.agents,
          [agentId]: {
            ...agent,
            events: [...(agent.events || []), event],
          },
        },
      };
    }),

  events: [],
  addEvent: (event) => set((state) => ({ events: [...state.events, event] })),

  // Output paths for downloads
  outputPaths: {},
  setOutputPaths: (paths) =>
    set((state) => ({ outputPaths: { ...state.outputPaths, ...paths } })),

  // API Actions
  uploadFile: async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/session/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Upload failed");
    }

    const data = await response.json();

    set((state) => ({
      session: {
        ...state.session,
        sessionId: data.session_id,
        fileName: data.file_name,
        filePath: data.file_path,
        status: SessionStatus.AWAITING_INTENT,
      },
    }));

    return { sessionId: data.session_id, fileName: data.file_name };
  },

  submitIntent: async (sessionId: string, intent: string, context?: string) => {
    set((state) => ({
      session: { ...state.session, status: SessionStatus.GENERATING_METADATA },
    }));

    const response = await fetch(
      `${API_BASE}/api/session/${sessionId}/intent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          high_level_intent: intent,
          document_context: context || null,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      set((state) => ({
        session: {
          ...state.session,
          status: SessionStatus.FAILED,
          error: error.detail,
        },
      }));
      throw new Error(error.detail || "Failed to submit intent");
    }

    const data = await response.json();

    set((state) => ({
      session: {
        ...state.session,
        metadata: data.metadata,
        metadataPath: data.metadata_path,
        highLevelIntent: intent,
        documentContext: context || null,
        status: SessionStatus.AWAITING_APPROVAL,
      },
    }));

    return data;
  },

  updateMetadata: async (sessionId: string, metadata: Record<string, any>) => {
    const response = await fetch(
      `${API_BASE}/api/session/${sessionId}/metadata`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metadata),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to update metadata");
    }

    set((state) => ({
      session: { ...state.session, metadata },
    }));
  },

  approveAndProcess: async (
    sessionId: string,
    approved: boolean,
    modifiedMetadata?: Record<string, any>
  ) => {
    set((state) => ({
      session: { ...state.session, status: SessionStatus.PROCESSING },
    }));

    const response = await fetch(
      `${API_BASE}/api/session/${sessionId}/approve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          approved,
          modified_metadata: modifiedMetadata || null,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      set((state) => ({
        session: {
          ...state.session,
          status: SessionStatus.FAILED,
          error: error.detail,
        },
      }));
      throw new Error(error.detail || "Failed to start processing");
    }

    const data = await response.json();

    set((state) => ({
      session: { ...state.session, pipelineId: data.pipeline_id },
      pipeline: {
        ...state.pipeline,
        id: data.pipeline_id,
        status: "running",
        fileName: state.session.fileName || undefined,
        currentStep: "Processing",
      },
    }));

    return data;
  },

  processIncomingEvent: (event: AgentEvent) => {
    const {
      addEvent,
      updateAgent,
      addAgent,
      setPipeline,
      addAgentEvent,
      setOutputPaths,
      events,
    } = get();
    const timestamp = new Date();

    // Debug logging
    console.log(`[Store] Processing event: ${event.event_type}`, event);

    // Define important events that should be logged to the event panel
    const importantEvents = new Set([
      EventTypes.PIPELINE_STARTED,
      EventTypes.PIPELINE_COMPLETED,
      EventTypes.PIPELINE_FAILED,
      EventTypes.MASTER_PLAN_GENERATING,
      EventTypes.MASTER_PLAN_GENERATED,
      EventTypes.RESIDUAL_INITIALIZED,
      EventTypes.RESIDUAL_CONTEXT_GENERATED,
      EventTypes.RESIDUAL_BROADCAST_COMPLETE,
      EventTypes.SUBMASTER_SPAWNED,
      EventTypes.SUBMASTER_COMPLETED,
      EventTypes.SUBMASTER_FAILED,
      EventTypes.WORKER_COMPLETED,
      EventTypes.WORKER_FAILED,
      EventTypes.REDUCER_STARTED,
      EventTypes.REDUCER_COMPLETED,
      // Reducer Pipeline Events (new)
      EventTypes.REDUCER_SUBMASTER_STARTED,
      EventTypes.REDUCER_SUBMASTER_COMPLETED,
      EventTypes.REDUCER_SUBMASTER_FAILED,
      EventTypes.REDUCER_RESIDUAL_STARTED,
      EventTypes.REDUCER_RESIDUAL_CONTEXT_UPDATED,
      EventTypes.REDUCER_RESIDUAL_PLAN_CREATED,
      EventTypes.REDUCER_RESIDUAL_COMPLETED,
      EventTypes.MASTER_MERGER_STARTED,
      EventTypes.MASTER_MERGER_COMPLETED,
      EventTypes.MASTER_MERGER_FAILED,
      EventTypes.PDF_GENERATION_STARTED,
      EventTypes.PDF_GENERATION_COMPLETED,
      EventTypes.PDF_GENERATION_FAILED,
    ]);

    // Check for duplicate events (same type + agent within 2 seconds)
    const isDuplicate = events.some(
      (e) =>
        e.eventType === event.event_type &&
        e.agentId === event.agent_id &&
        timestamp.getTime() - e.timestamp.getTime() < 2000
    );

    // Log the event only if important and not duplicate
    let severity: "info" | "success" | "warning" | "error" = "info";
    let message = event.event_type;

    if (
      event.event_type.includes("failed") ||
      event.event_type.includes("error")
    )
      severity = "error";
    if (event.event_type.includes("completed")) severity = "success";
    if (event.event_type.includes("warning")) severity = "warning";

    // Format readable message based on type
    switch (event.event_type) {
      case EventTypes.PIPELINE_STARTED:
        message = `Pipeline started for ${event.data.file || "document"}`;
        break;
      case EventTypes.PIPELINE_COMPLETED:
        message = `Pipeline completed successfully`;
        break;
      case EventTypes.PIPELINE_FAILED:
        message = `Pipeline failed: ${event.data.error || "Unknown error"}`;
        break;
      case EventTypes.MASTER_PLAN_GENERATING:
        message = `Master generating execution plan`;
        break;
      case EventTypes.MASTER_PLAN_GENERATED:
        message = `Plan ready: ${event.data.submaster_count || "?"} submasters`;
        break;
      case EventTypes.RESIDUAL_INITIALIZED:
        message = `Residual agent initialized`;
        break;
      case EventTypes.RESIDUAL_CONTEXT_GENERATED:
        message = `Global context generated`;
        break;
      case EventTypes.RESIDUAL_BROADCAST_COMPLETE:
        message = `Context broadcast complete`;
        break;
      case EventTypes.SUBMASTER_SPAWNED:
        message = `SubMaster spawned: ${event.data.role || event.agent_id}`;
        break;
      case EventTypes.SUBMASTER_COMPLETED:
        message = `SubMaster completed: ${event.agent_id}`;
        break;
      case EventTypes.SUBMASTER_FAILED:
        message = `SubMaster failed: ${event.agent_id}`;
        break;
      case EventTypes.WORKER_COMPLETED:
        message = `Worker completed page ${event.data.page || "?"}`;
        break;
      case EventTypes.WORKER_FAILED:
        message = `Worker failed: ${event.agent_id}`;
        break;
      case EventTypes.REDUCER_STARTED:
        message = `Reducer aggregating results`;
        break;
      case EventTypes.REDUCER_COMPLETED:
        message = `Reducer completed - final output ready`;
        break;
      // Reducer Pipeline Messages
      case EventTypes.REDUCER_SUBMASTER_STARTED:
        message = `🔄 Reducer SubMasters started`;
        break;
      case EventTypes.REDUCER_SUBMASTER_COMPLETED:
        message = `✅ Reducer SubMasters completed`;
        break;
      case EventTypes.REDUCER_SUBMASTER_FAILED:
        message = `❌ Reducer SubMasters failed`;
        break;
      case EventTypes.REDUCER_RESIDUAL_STARTED:
        message = `🧠 Reducer Residual Agent started`;
        break;
      case EventTypes.REDUCER_RESIDUAL_CONTEXT_UPDATING:
        message = `📊 Updating global context from reducer results`;
        break;
      case EventTypes.REDUCER_RESIDUAL_CONTEXT_UPDATED:
        message = `✅ Global context updated (${
          event.data.context_size || "?"
        } chars)`;
        break;
      case EventTypes.REDUCER_RESIDUAL_PLAN_CREATING:
        message = `📝 Creating processing plan`;
        break;
      case EventTypes.REDUCER_RESIDUAL_PLAN_CREATED:
        message = `✅ Processing plan created`;
        break;
      case EventTypes.REDUCER_RESIDUAL_COMPLETED:
        message = `✅ Reducer Residual Agent completed`;
        break;
      case EventTypes.MASTER_MERGER_STARTED:
        message = `🔀 Master Merger started - synthesizing final document`;
        break;
      case EventTypes.MASTER_MERGER_SYNTHESIZING:
        message = `📝 Synthesizing final document...`;
        break;
      case EventTypes.MASTER_MERGER_COMPLETED:
        message = `✅ Master Merger completed - final synthesis ready`;
        break;
      case EventTypes.MASTER_MERGER_FAILED:
        message = `❌ Master Merger failed: ${event.data.error || "Unknown"}`;
        break;
      case EventTypes.PDF_GENERATION_STARTED:
        message = `📄 Generating PDF report...`;
        break;
      case EventTypes.PDF_GENERATION_COMPLETED:
        message = `✅ PDF report generated successfully`;
        break;
      case EventTypes.PDF_GENERATION_FAILED:
        message = `❌ PDF generation failed`;
        break;
    }

    // Only add to event log if important and not duplicate
    if (importantEvents.has(event.event_type) && !isDuplicate) {
      addEvent({
        id: Math.random().toString(36).substring(7),
        timestamp,
        eventType: event.event_type,
        message,
        agentId: event.agent_id,
        severity,
      });
    }

    // Important events for agent-specific logs (more detailed than global)
    const agentImportantEvents = new Set([
      EventTypes.MASTER_PLAN_GENERATING,
      EventTypes.MASTER_PLAN_GENERATED,
      EventTypes.RESIDUAL_INITIALIZED,
      EventTypes.RESIDUAL_CONTEXT_GENERATED,
      EventTypes.RESIDUAL_BROADCAST_COMPLETE,
      EventTypes.SUBMASTER_SPAWNED,
      EventTypes.SUBMASTER_INITIALIZED,
      EventTypes.SUBMASTER_CONTEXT_RECEIVED,
      EventTypes.SUBMASTER_COMPLETED,
      EventTypes.SUBMASTER_FAILED,
      EventTypes.WORKER_SPAWNED,
      EventTypes.WORKER_INITIALIZED,
      EventTypes.WORKER_COMPLETED,
      EventTypes.WORKER_FAILED,
    ]);

    // Add event to agent's event log if agent exists and event is important
    if (event.agent_id && get().agents[event.agent_id]) {
      const agentEvents = get().agents[event.agent_id]?.events || [];
      const isAgentDuplicate = agentEvents.some(
        (e) =>
          e.eventType === event.event_type &&
          timestamp.getTime() - new Date(e.timestamp).getTime() < 2000
      );

      if (agentImportantEvents.has(event.event_type) && !isAgentDuplicate) {
        addAgentEvent(event.agent_id, {
          timestamp,
          eventType: event.event_type,
          message,
          data: event.data,
        });
      }
    }

    // Handle State Logic
    switch (event.event_type) {
      case EventTypes.PIPELINE_STARTED:
        setPipeline({
          id: event.pipeline_id,
          status: "running",
          fileName: event.data.file,
          startTime: Date.now(),
        });
        break;

      case EventTypes.PIPELINE_COMPLETED:
        setPipeline({
          status: "completed",
          endTime: Date.now(),
          currentStep: "Finished",
        });
        // Also update session status so RAG chat knows processing is complete
        set((state) => ({
          session: {
            ...state.session,
            status: SessionStatus.COMPLETED,
          },
        }));
        // Store output paths if available
        if (event.data.report_path) {
          setOutputPaths({ reportPath: event.data.report_path });
        }
        if (event.data.json_path) {
          setOutputPaths({ jsonPath: event.data.json_path });
        }
        break;

      case EventTypes.PIPELINE_FAILED:
        setPipeline({
          status: "failed",
          endTime: Date.now(),
          currentStep: "Failed",
        });
        // Update session status on failure
        set((state) => ({
          session: {
            ...state.session,
            status: SessionStatus.FAILED,
          },
        }));
        break;

      case EventTypes.MASTER_PLAN_GENERATING:
        // Ensure master exists
        if (!get().agents["master"]) {
          addAgent({
            id: "master",
            type: AgentType.MASTER,
            status: AgentStatus.PROCESSING,
            label: "Master Orchestrator",
            metadata: { role: "Planner" },
            events: [
              {
                timestamp,
                eventType: event.event_type,
                message: "Started generating plan",
              },
            ],
            startTime: Date.now(),
          });
        } else {
          updateAgent("master", { status: AgentStatus.PROCESSING });
        }
        setPipeline({ currentStep: "Generating Plan" });
        break;

      case EventTypes.MASTER_PLAN_GENERATED:
        updateAgent("master", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["master"]?.metadata,
            submasterCount: event.data.submaster_count,
          },
          endTime: Date.now(),
        });
        setPipeline({ currentStep: "Spawning SubMasters" });
        break;

      case EventTypes.RESIDUAL_INITIALIZED:
        addAgent({
          id: event.agent_id || "residual",
          type: AgentType.RESIDUAL,
          status: AgentStatus.INITIALIZING,
          label: "Residual Agent",
          parentId: "master",
          metadata: { role: "Global Context Coordinator" },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Initialized",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.RESIDUAL_CONTEXT_GENERATING:
        updateAgent(event.agent_id || "residual", {
          status: AgentStatus.PROCESSING,
        });
        break;

      case EventTypes.RESIDUAL_CONTEXT_GENERATED:
        updateAgent(event.agent_id || "residual", {
          metadata: {
            ...get().agents[event.agent_id || "residual"]?.metadata,
            context: event.data.context_preview,
          },
        });
        break;

      case EventTypes.RESIDUAL_BROADCASTING:
        updateAgent(event.agent_id || "residual", {
          status: AgentStatus.PROCESSING,
        });
        break;

      case EventTypes.RESIDUAL_BROADCAST_COMPLETE:
        updateAgent(event.agent_id || "residual", {
          status: AgentStatus.COMPLETED,
          endTime: Date.now(),
        });
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
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: `Spawned with role: ${event.data.role}`,
              data: event.data,
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.SUBMASTER_INITIALIZED:
        updateAgent(event.agent_id!, {
          status: AgentStatus.WAITING,
        });
        break;

      case EventTypes.SUBMASTER_CONTEXT_RECEIVED:
        // Just update state, event logging handled above
        break;

      case EventTypes.SUBMASTER_PROCESSING:
        updateAgent(event.agent_id!, { status: AgentStatus.PROCESSING });
        break;

      case EventTypes.SUBMASTER_WORKERS_SPAWNED:
        // Just update state, event logging handled above
        break;

      case EventTypes.WORKER_SPAWNED:
        // Workers may not have a page assigned at spawn time
        const workerId =
          event.agent_id || `worker-${Math.random().toString(36).substr(2, 6)}`;
        const workerLabel = event.data.page
          ? `Worker P${event.data.page}`
          : event.data.worker_id || workerId;
        addAgent({
          id: workerId,
          type: AgentType.WORKER,
          status: AgentStatus.WAITING,
          label: workerLabel,
          parentId: event.data.submaster_id,
          metadata: {
            pages: event.data.page,
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: event.data.page
                ? `Spawned for page ${event.data.page}`
                : "Worker spawned",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.WORKER_INITIALIZED:
        updateAgent(event.agent_id!, { status: AgentStatus.WAITING });
        break;

      case EventTypes.WORKER_CONTEXT_RECEIVED:
        // Just track state, event logging handled above
        break;

      case EventTypes.WORKER_PROCESSING:
      case EventTypes.WORKER_PAGE_STARTED:
        updateAgent(event.agent_id!, { status: AgentStatus.PROCESSING });
        // Also set SubMaster to processing if not already
        const worker = get().agents[event.agent_id!];
        if (worker && worker.parentId) {
          updateAgent(worker.parentId, { status: AgentStatus.PROCESSING });
        }
        break;

      case EventTypes.WORKER_PAGE_COMPLETED:
      case EventTypes.WORKER_COMPLETED:
        updateAgent(event.agent_id!, {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents[event.agent_id!]?.metadata,
            summary: event.data.summary,
            entities: event.data.entities,
            keywords: event.data.keywords,
          },
          endTime: Date.now(),
        });
        break;

      case EventTypes.WORKER_FAILED:
        updateAgent(event.agent_id!, {
          status: AgentStatus.FAILED,
          endTime: Date.now(),
        });
        break;

      case EventTypes.SUBMASTER_PROGRESS:
        updateAgent(event.agent_id!, {
          metadata: {
            ...get().agents[event.agent_id!]?.metadata,
            progress: event.data.percent,
          },
        });
        break;

      case EventTypes.SUBMASTER_COMPLETED:
        updateAgent(event.agent_id!, {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents[event.agent_id!]?.metadata,
            progress: 100,
            summary: event.data.summary,
          },
          endTime: Date.now(),
        });
        break;

      case EventTypes.SUBMASTER_FAILED:
        updateAgent(event.agent_id!, {
          status: AgentStatus.FAILED,
          endTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_STARTED:
        setPipeline({ currentStep: "Aggregating Results" });
        // Create reducer node in the tree
        addAgent({
          id: "reducer",
          type: AgentType.REDUCER,
          status: AgentStatus.PROCESSING,
          label: "Reducer",
          parentId: "master",
          metadata: {
            role: "Result Aggregator",
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Started aggregating results from all submasters",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_AGGREGATING:
        // Update reducer progress
        updateAgent("reducer", {
          metadata: {
            ...get().agents["reducer"]?.metadata,
            progress: event.data?.progress_percent || 0,
            current: event.data?.current,
            total: event.data?.total,
          },
        });
        addAgentEvent("reducer", {
          timestamp,
          eventType: event.event_type,
          message: `Aggregating result ${event.data?.current}/${event.data?.total}`,
        });
        break;

      case EventTypes.REDUCER_COMPLETED:
        setPipeline({ currentStep: "Generating Report" });
        // Update reducer node to completed
        updateAgent("reducer", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["reducer"]?.metadata,
            progress: 100,
          },
          endTime: Date.now(),
        });
        if (event.data.report_path) {
          setOutputPaths({ reportPath: event.data.report_path });
        }
        if (event.data.json_path) {
          setOutputPaths({ jsonPath: event.data.json_path });
        }
        break;

      // ============================================
      // REDUCER PIPELINE EVENTS (FULL UNIFIED FLOW)
      // ============================================

      case EventTypes.REDUCER_SUBMASTER_STARTED:
        setPipeline({ currentStep: "Reducer SubMasters" });
        // Create reducer submasters node
        addAgent({
          id: "reducer_submasters",
          type: AgentType.REDUCER_SUBMASTER,
          status: AgentStatus.PROCESSING,
          label: "Reducer SubMasters",
          parentId: "master",
          metadata: {
            role: "Aggregate & Enhance Mapper Results",
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Started processing mapper results",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_SUBMASTER_PROCESSING:
        updateAgent("reducer_submasters", {
          status: AgentStatus.PROCESSING,
        });
        break;

      case EventTypes.REDUCER_SUBMASTER_PROGRESS:
        updateAgent("reducer_submasters", {
          metadata: {
            ...get().agents["reducer_submasters"]?.metadata,
            progress: event.data?.progress_percent || 0,
          },
        });
        break;

      case EventTypes.REDUCER_SUBMASTER_COMPLETED:
        setPipeline({ currentStep: "Reducer Residual Agent" });
        updateAgent("reducer_submasters", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["reducer_submasters"]?.metadata,
            progress: 100,
            numResults: event.data?.num_results,
            elapsedTime: event.data?.elapsed_time,
          },
          endTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_SUBMASTER_FAILED:
        updateAgent("reducer_submasters", {
          status: AgentStatus.FAILED,
          endTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_STARTED:
        setPipeline({ currentStep: "Building Global Context" });
        // Create reducer residual agent node
        addAgent({
          id: "reducer_residual",
          type: AgentType.REDUCER_RESIDUAL,
          status: AgentStatus.PROCESSING,
          label: "Reducer Residual Agent",
          parentId: "reducer_submasters",
          metadata: {
            role: "Global Context Builder",
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Started building global context",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_CONTEXT_UPDATING:
        updateAgent("reducer_residual", {
          metadata: {
            ...get().agents["reducer_residual"]?.metadata,
            progress: 25,
            status: "Updating context from reducer results",
          },
        });
        addAgentEvent("reducer_residual", {
          timestamp,
          eventType: event.event_type,
          message: "Updating context from reducer results",
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_CONTEXT_UPDATED:
        updateAgent("reducer_residual", {
          metadata: {
            ...get().agents["reducer_residual"]?.metadata,
            progress: 50,
            contextSize: event.data?.context_size,
          },
        });
        addAgentEvent("reducer_residual", {
          timestamp,
          eventType: event.event_type,
          message: `Global context updated (${event.data?.context_size} chars)`,
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_PLAN_CREATING:
        updateAgent("reducer_residual", {
          metadata: {
            ...get().agents["reducer_residual"]?.metadata,
            progress: 75,
            status: "Creating processing plan",
          },
        });
        addAgentEvent("reducer_residual", {
          timestamp,
          eventType: event.event_type,
          message: "Creating processing plan",
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_PLAN_CREATED:
        updateAgent("reducer_residual", {
          metadata: {
            ...get().agents["reducer_residual"]?.metadata,
            progress: 90,
            planSize: event.data?.plan_size,
          },
        });
        addAgentEvent("reducer_residual", {
          timestamp,
          eventType: event.event_type,
          message: `Processing plan created (${event.data?.plan_size} chars)`,
        });
        break;

      case EventTypes.REDUCER_RESIDUAL_COMPLETED:
        setPipeline({ currentStep: "Master Merger" });
        updateAgent("reducer_residual", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["reducer_residual"]?.metadata,
            progress: 100,
            elapsedTime: event.data?.elapsed_time,
          },
          endTime: Date.now(),
        });
        break;

      case EventTypes.MASTER_MERGER_STARTED:
        setPipeline({ currentStep: "Final Synthesis" });
        // Create master merger agent node
        addAgent({
          id: "master_merger",
          type: AgentType.MASTER_MERGER,
          status: AgentStatus.PROCESSING,
          label: "Master Merger",
          parentId: "reducer_residual",
          metadata: {
            role: "Final Document Synthesis",
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Started final document synthesis",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.MASTER_MERGER_SYNTHESIZING:
        updateAgent("master_merger", {
          metadata: {
            ...get().agents["master_merger"]?.metadata,
            progress: 30,
            status: "Synthesizing final document",
          },
        });
        addAgentEvent("master_merger", {
          timestamp,
          eventType: event.event_type,
          message: "Synthesizing final document...",
        });
        break;

      case EventTypes.MASTER_MERGER_EXECUTIVE_SUMMARY:
        updateAgent("master_merger", {
          metadata: {
            ...get().agents["master_merger"]?.metadata,
            progress: 50,
            status: "Creating executive summary",
          },
        });
        addAgentEvent("master_merger", {
          timestamp,
          eventType: event.event_type,
          message: "Creating executive summary",
        });
        break;

      case EventTypes.MASTER_MERGER_DETAILED_SYNTHESIS:
        updateAgent("master_merger", {
          metadata: {
            ...get().agents["master_merger"]?.metadata,
            progress: 70,
            status: "Creating detailed synthesis",
          },
        });
        addAgentEvent("master_merger", {
          timestamp,
          eventType: event.event_type,
          message: "Creating detailed synthesis",
        });
        break;

      case EventTypes.MASTER_MERGER_INSIGHTS:
        updateAgent("master_merger", {
          metadata: {
            ...get().agents["master_merger"]?.metadata,
            progress: 85,
            status: "Generating insights and conclusions",
          },
        });
        addAgentEvent("master_merger", {
          timestamp,
          eventType: event.event_type,
          message: "Generating insights and conclusions",
        });
        break;

      case EventTypes.MASTER_MERGER_COMPLETED:
        setPipeline({ currentStep: "Generating PDF" });
        updateAgent("master_merger", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["master_merger"]?.metadata,
            progress: 100,
            elapsedTime: event.data?.elapsed_time,
            resultSize: event.data?.result_size,
          },
          endTime: Date.now(),
        });
        break;

      case EventTypes.MASTER_MERGER_FAILED:
        updateAgent("master_merger", {
          status: AgentStatus.FAILED,
          endTime: Date.now(),
        });
        break;

      case EventTypes.PDF_GENERATION_STARTED:
        setPipeline({ currentStep: "PDF Generation" });
        // Create PDF generator node
        addAgent({
          id: "pdf_generator",
          type: AgentType.PDF_GENERATOR,
          status: AgentStatus.PROCESSING,
          label: "PDF Generator",
          parentId: "master_merger",
          metadata: {
            role: "Final Report Generator",
            progress: 0,
          },
          events: [
            {
              timestamp,
              eventType: event.event_type,
              message: "Started generating PDF report",
            },
          ],
          startTime: Date.now(),
        });
        break;

      case EventTypes.PDF_GENERATION_COMPLETED:
        setPipeline({ currentStep: "Complete" });
        updateAgent("pdf_generator", {
          status: AgentStatus.COMPLETED,
          metadata: {
            ...get().agents["pdf_generator"]?.metadata,
            progress: 100,
            pdfPath: event.data?.pdf_path,
          },
          endTime: Date.now(),
        });
        if (event.data?.pdf_path) {
          setOutputPaths({ reportPath: event.data.pdf_path });
        }
        break;

      case EventTypes.PDF_GENERATION_FAILED:
        updateAgent("pdf_generator", {
          status: AgentStatus.FAILED,
          endTime: Date.now(),
        });
        break;
    }
  },

  // Download functions
  downloadReport: async () => {
    const { outputPaths, session } = get();
    if (!outputPaths.reportPath && !session.pipelineId) {
      console.error("No report path available");
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/pipeline/${session.pipelineId}/download/report`,
        { method: "GET" }
      );

      if (!response.ok) throw new Error("Failed to download report");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis_report_${session.pipelineId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
    }
  },

  downloadJson: async () => {
    const { outputPaths, session } = get();
    if (!outputPaths.jsonPath && !session.pipelineId) {
      console.error("No JSON path available");
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/pipeline/${session.pipelineId}/download/json`,
        { method: "GET" }
      );

      if (!response.ok) throw new Error("Failed to download JSON");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis_results_${session.pipelineId}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
    }
  },

  // ==================== Chat/RAG State ====================
  chat: {
    messages: [],
    isLoading: false,
    error: null,
    ragAvailable: false,
  },

  sendChatMessage: async (question: string) => {
    const { session } = get();
    const sessionId = session.sessionId;

    // Add user message immediately
    const userMessageId = `msg_${Date.now()}_user`;
    const userMessage: ChatMessage = {
      id: userMessageId,
      role: "user",
      content: question,
      timestamp: new Date(),
    };

    // Add loading message for assistant
    const loadingMessageId = `msg_${Date.now()}_assistant`;
    const loadingMessage: ChatMessage = {
      id: loadingMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true,
    };

    set((state) => ({
      chat: {
        ...state.chat,
        messages: [...state.chat.messages, userMessage, loadingMessage],
        isLoading: true,
        error: null,
      },
    }));

    try {
      // Try session-specific chat first, fallback to standalone
      let endpoint = sessionId
        ? `${API_BASE}/api/session/${sessionId}/chat`
        : `${API_BASE}/api/chat`;

      let response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 5,
        }),
      });

      // If session chat fails (e.g., session not found or not completed), try standalone
      if (!response.ok && sessionId) {
        console.log("Session chat failed, trying standalone chat...");
        response = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
            top_k: 5,
          }),
        });
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Chat failed");
      }

      const data = await response.json();

      // Update the loading message with actual response
      const assistantMessage: ChatMessage = {
        id: loadingMessageId,
        role: "assistant",
        content: data.answer,
        timestamp: new Date(),
        sources: data.sources || [],
        isLoading: false,
      };

      set((state) => ({
        chat: {
          ...state.chat,
          messages: state.chat.messages.map((msg) =>
            msg.id === loadingMessageId ? assistantMessage : msg
          ),
          isLoading: false,
          ragAvailable: true,
        },
      }));
    } catch (error: any) {
      // Remove loading message and add error
      set((state) => ({
        chat: {
          ...state.chat,
          messages: state.chat.messages.filter(
            (msg) => msg.id !== loadingMessageId
          ),
          isLoading: false,
          error: error.message || "Failed to send message",
          ragAvailable: error.message?.includes("Vector store not found")
            ? false
            : state.chat.ragAvailable,
        },
      }));
    }
  },

  clearChat: () => {
    const { session } = get();

    // Clear on server if session exists
    if (session.sessionId) {
      fetch(`${API_BASE}/api/session/${session.sessionId}/chat/history`, {
        method: "DELETE",
      }).catch(console.error);
    }

    set((state) => ({
      chat: {
        ...state.chat,
        messages: [],
        error: null,
      },
    }));
  },

  loadChatHistory: async () => {
    const { session } = get();
    if (!session.sessionId) return;

    try {
      const response = await fetch(
        `${API_BASE}/api/session/${session.sessionId}/chat/history`
      );

      if (!response.ok) return;

      const data = await response.json();

      const messages: ChatMessage[] = data.messages.map(
        (msg: { role: string; content: string }, index: number) => ({
          id: `history_${index}`,
          role: msg.role as "user" | "assistant",
          content: msg.content,
          timestamp: new Date(),
        })
      );

      set((state) => ({
        chat: {
          ...state.chat,
          messages,
          ragAvailable: true,
        },
      }));
    } catch (error) {
      console.error("Failed to load chat history:", error);
    }
  },
}));
