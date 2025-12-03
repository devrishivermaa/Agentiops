# Frontend Builder Prompt: AgentOps Real-Time Visualization Dashboard

## 🎯 Project Overview

Build a **React + TypeScript** real-time dashboard for visualizing an AI agent orchestration engine. The frontend connects to a FastAPI backend via REST APIs and WebSocket for live event streaming. Users should see the entire pipeline process: uploading documents, master agent generating plans, sub-agents spawning, workers processing pages, and results being generated—all animated in real-time.

---

## 🏗️ Tech Stack Requirements

- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **State Management**: Zustand or React Context + useReducer
- **WebSocket**: Native WebSocket or socket.io-client
- **Animations**: Framer Motion for agent spawning/processing animations
- **Charts**: Recharts for progress visualization
- **Icons**: Lucide React
- **Build Tool**: Vite

---

## 🌐 Backend API Reference

**Base URL**: `http://localhost:8000`

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/files` | List available PDF files |
| POST | `/api/upload` | Upload PDF file (multipart/form-data) |
| POST | `/api/upload-and-process` | Upload and immediately start processing |
| POST | `/api/pipeline/start` | Start pipeline with file_path |
| GET | `/api/pipeline/{pipeline_id}` | Get pipeline status |
| GET | `/api/pipeline/{pipeline_id}/events` | Get event history for pipeline |
| GET | `/api/pipeline/{pipeline_id}/agents` | Get agent hierarchy |
| POST | `/api/pipeline/{pipeline_id}/approve` | Approve/reject SubMaster plan |
| GET | `/api/pipelines` | List all pipelines |
| GET | `/api/stats` | Get system statistics |

### WebSocket Endpoint

**URL**: `ws://localhost:8000/api/ws`

Connect to receive real-time events. Events are JSON objects with this structure:

```typescript
interface AgentEvent {
  event_type: string;
  pipeline_id: string;
  timestamp: string;
  data: Record<string, any>;
  agent_id?: string;
  agent_type?: "master" | "submaster" | "worker";
}
```

### Event Types to Handle

```typescript
enum EventType {
  // Pipeline lifecycle
  PIPELINE_STARTED = "pipeline.started",
  PIPELINE_STEP_STARTED = "pipeline.step_started",
  PIPELINE_STEP_COMPLETED = "pipeline.step_completed",
  PIPELINE_COMPLETED = "pipeline.completed",
  PIPELINE_FAILED = "pipeline.failed",
  
  // Master Agent
  MASTER_PLAN_GENERATING = "master.plan_generating",
  MASTER_PLAN_GENERATED = "master.plan_generated",
  MASTER_AWAITING_FEEDBACK = "master.awaiting_feedback",
  MASTER_PLAN_APPROVED = "master.plan_approved",
  
  // SubMaster Agents
  SUBMASTER_SPAWNED = "submaster.spawned",
  SUBMASTER_INITIALIZED = "submaster.initialized",
  SUBMASTER_PROCESSING = "submaster.processing",
  SUBMASTER_PROGRESS = "submaster.progress",
  SUBMASTER_COMPLETED = "submaster.completed",
  SUBMASTER_FAILED = "submaster.failed",
  
  // Worker Agents
  WORKER_SPAWNED = "worker.spawned",
  WORKER_PROCESSING = "worker.processing",
  WORKER_COMPLETED = "worker.completed",
  WORKER_FAILED = "worker.failed",
  
  // System
  SYSTEM_STATS = "system.stats",
  RATE_LIMIT_WARNING = "ratelimit.warning",
}
```

---

## 📐 Application Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🤖 AgentOps Dashboard                    [Connection: 🟢 Live] │
├──────────────────────┬──────────────────────────────────────────┤
│                      │                                          │
│   SIDEBAR            │   MAIN CONTENT AREA                      │
│   ─────────          │   ─────────────────                      │
│                      │                                          │
│   📁 Upload File     │   Switches between:                      │
│                      │   1. Upload & Start View                 │
│   📋 Active Pipes    │   2. Agent Visualization (Tree/Graph)    │
│   ├─ Pipeline #1     │   3. Pipeline Details                    │
│   └─ Pipeline #2     │   4. Results View                        │
│                      │                                          │
│   📊 History         │                                          │
│                      │                                          │
│   ⚙️ Settings        │                                          │
│                      │                                          │
├──────────────────────┴──────────────────────────────────────────┤
│  EVENT LOG (Collapsible) - Real-time scrolling event feed      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 Component Specifications

### 1. **FileUploader Component**

```typescript
// Features:
// - Drag & drop zone for PDF files
// - File validation (PDF only, max size)
// - Upload progress indicator
// - Option to "Upload & Process" or just "Upload"
// - Display list of already uploaded files from GET /api/files

interface FileUploaderProps {
  onUploadComplete: (filePath: string, fileName: string) => void;
  onStartPipeline: (filePath: string, autoApprove: boolean) => void;
}
```

**Visual Design**:
- Large dropzone with dashed border
- Icon changes on drag-over
- Progress bar during upload
- Success/error toast notifications

---

### 2. **AgentVisualization Component** (⭐ CRITICAL)

This is the main visualization showing the agent hierarchy in real-time.

```typescript
interface AgentNode {
  id: string;
  type: "master" | "submaster" | "worker";
  status: "spawned" | "initializing" | "processing" | "completed" | "failed";
  label: string;
  children?: AgentNode[];
  metadata?: {
    role?: string;
    pages?: number[];
    progress?: number;
  };
}
```

**Visual Layout** (Hierarchical Tree):

```
                    ┌──────────────────┐
                    │   🧠 MASTER      │
                    │   Agent          │
                    │   [Generating..] │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
   │ 📋 SubMaster│    │ 📋 SubMaster│    │ 📋 SubMaster│
   │ Introduction│    │ Methodology │    │ Conclusion  │
   │ [Processing]│    │ [Waiting]   │    │ [Waiting]   │
   │ ████░░░ 60% │    │             │    │             │
   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                  │                  │
    ┌─────┼─────┐      ┌─────┼─────┐      ┌─────┼─────┐
    │     │     │      │     │     │      │     │     │
   🔧    🔧    🔧     🔧    🔧    🔧     🔧    🔧    🔧
   W1    W2    W3     W4    W5    W6     W7    W8    W9
  [✓]   [⟳]   [○]   [○]   [○]   [○]    [○]   [○]   [○]
```

**Animation Requirements**:
- Nodes **fade in with scale animation** when spawned
- **Pulsing glow effect** when processing
- **Checkmark animation** when completed
- **Connection lines animate** when data flows between agents
- Progress bars smoothly animate

**Status Colors**:
- 🟡 Yellow: Spawned/Initializing
- 🔵 Blue: Processing (pulsing)
- 🟢 Green: Completed
- 🔴 Red: Failed

---

### 3. **PipelineProgress Component**

Shows overall pipeline progress with step indicators.

```
Step 1          Step 2          Step 3          Step 4          Step 5
Extract    →    Plan Gen   →    Approve   →    Process   →    Report
  ✓              ✓               ⟳              ○              ○
[Complete]    [Complete]     [Waiting]      [Pending]      [Pending]
```

**Steps**:
1. PDF Extraction - "pipeline.step_started" with step="extraction"
2. Plan Generation - "master.plan_generating" / "master.plan_generated"
3. Plan Approval - "master.awaiting_feedback" (optional if auto_approve=true)
4. Agent Processing - SubMasters + Workers processing
5. Report Generation - Final report compilation

---

### 4. **PlanApprovalModal Component**

When `master.awaiting_feedback` event arrives, show modal for user to approve/reject.

```typescript
interface SubMasterPlan {
  status: string;
  num_submasters: number;
  distribution_strategy: string;
  submasters: Array<{
    submaster_id: string;
    role: string;
    assigned_sections: string[];
    page_range: [number, number];
    estimated_workload: string;
  }>;
}
```

**Modal Content**:
- Display plan summary
- List each SubMaster's role and page assignments
- "Approve" button (green)
- "Reject with Feedback" button + textarea
- POST to `/api/pipeline/{id}/approve` with `{ approved: boolean, feedback?: string }`

---

### 5. **EventLog Component**

Real-time scrolling log of all events.

```typescript
interface EventLogEntry {
  timestamp: string;
  eventType: string;
  agentType?: string;
  agentId?: string;
  message: string;
  severity: "info" | "success" | "warning" | "error";
}
```

**Features**:
- Auto-scroll to bottom (toggleable)
- Color-coded by event type
- Filter by event type or agent
- Collapsible panel at bottom of screen
- Timestamp formatting (relative: "2s ago")

---

### 6. **ResultsView Component**

Displays final processing results when pipeline completes.

**Sections**:
- Summary statistics (pages processed, time taken, agents used)
- Generated report/analysis content
- Downloadable report (if available)
- Option to start new pipeline

---

## 🔌 WebSocket Integration

### Connection Manager Hook

```typescript
// hooks/useWebSocket.ts

interface UseWebSocketOptions {
  url: string;
  onEvent: (event: AgentEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  reconnectAttempts?: number;
}

function useWebSocket(options: UseWebSocketOptions) {
  // Implement:
  // - Auto-reconnect with exponential backoff
  // - Connection status tracking
  // - Heartbeat/ping handling
  // - Event parsing and dispatching
}
```

### Event Handler Pattern

```typescript
// Process incoming events and update state

function handleEvent(event: AgentEvent, dispatch: Dispatch) {
  switch (event.event_type) {
    case "pipeline.started":
      dispatch({ type: "PIPELINE_STARTED", payload: event });
      break;
      
    case "submaster.spawned":
      dispatch({ 
        type: "ADD_SUBMASTER", 
        payload: {
          id: event.agent_id,
          pipelineId: event.pipeline_id,
          role: event.data.role,
          pages: event.data.page_range,
        }
      });
      break;
      
    case "worker.spawned":
      dispatch({
        type: "ADD_WORKER",
        payload: {
          id: event.agent_id,
          submasterId: event.data.submaster_id,
          page: event.data.page,
        }
      });
      // Trigger spawn animation
      break;
      
    case "worker.completed":
      dispatch({
        type: "UPDATE_WORKER_STATUS",
        payload: { id: event.agent_id, status: "completed" }
      });
      // Trigger completion animation
      break;
      
    // ... handle all event types
  }
}
```

---

## 📊 State Management Schema

```typescript
interface AppState {
  // Connection
  wsConnected: boolean;
  wsReconnecting: boolean;
  
  // Files
  availableFiles: Array<{
    filename: string;
    file_path: string;
    size_mb: number;
    modified_at: string;
  }>;
  
  // Pipelines
  pipelines: Map<string, PipelineState>;
  activePipelineId: string | null;
  
  // Events
  eventLog: EventLogEntry[];
  
  // UI
  sidebarCollapsed: boolean;
  eventLogExpanded: boolean;
}

interface PipelineState {
  id: string;
  status: "pending" | "running" | "awaiting_approval" | "completed" | "failed";
  fileName: string;
  startedAt: string;
  completedAt?: string;
  currentStep: string;
  progress: number;
  
  // Agent hierarchy
  masterAgent: MasterAgentState | null;
  submasters: Map<string, SubMasterState>;
  
  // Plan (for approval flow)
  plan: SubMasterPlan | null;
  planApproved: boolean;
  
  // Results
  result: any | null;
  error: string | null;
}

interface SubMasterState {
  id: string;
  role: string;
  status: AgentStatus;
  pageRange: [number, number];
  progress: number;
  workers: Map<string, WorkerState>;
}

interface WorkerState {
  id: string;
  page: number;
  status: AgentStatus;
}
```

---

## 🎬 Animation Specifications

### Agent Spawn Animation (Framer Motion)

```typescript
const spawnVariants = {
  hidden: { 
    scale: 0, 
    opacity: 0,
    y: -20 
  },
  visible: { 
    scale: 1, 
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 260,
      damping: 20,
      duration: 0.5
    }
  }
};

// Usage
<motion.div
  variants={spawnVariants}
  initial="hidden"
  animate="visible"
>
  <AgentNode ... />
</motion.div>
```

### Processing Pulse Animation

```css
@keyframes processingPulse {
  0%, 100% { 
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4);
  }
  50% { 
    box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
  }
}

.agent-processing {
  animation: processingPulse 2s ease-in-out infinite;
}
```

### Connection Line Animation

```typescript
// Animated path for data flow
const ConnectionLine = ({ from, to, active }) => (
  <motion.path
    d={`M ${from.x} ${from.y} Q ${midX} ${midY} ${to.x} ${to.y}`}
    stroke={active ? "#3B82F6" : "#E5E7EB"}
    strokeWidth={2}
    fill="none"
    initial={{ pathLength: 0 }}
    animate={{ pathLength: 1 }}
    transition={{ duration: 0.5 }}
  />
);
```

---

## 📱 Responsive Design Requirements

- **Desktop** (>1024px): Full layout with sidebar + main content + event log
- **Tablet** (768-1024px): Collapsible sidebar, stacked event log
- **Mobile** (<768px): Bottom navigation, full-screen views, simplified agent tree

---

## 🎯 User Flows

### Flow 1: Upload and Process

1. User lands on dashboard
2. Drags PDF to upload zone → Upload animation
3. Clicks "Start Processing" → Pipeline created
4. Sees Master Agent appear → "Generating plan..."
5. Plan modal appears (if not auto-approve) → Reviews and approves
6. SubMasters spawn one by one with animation
7. Workers spawn under each SubMaster
8. Progress bars update in real-time
9. Agents complete and show checkmarks
10. Results view appears with summary

### Flow 2: Monitor Existing Pipeline

1. User opens dashboard with pipeline running
2. WebSocket connects, receives event history
3. UI reconstructs current state from events
4. Live updates continue

---

## 🔧 API Integration Examples

### Upload File

```typescript
async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch("http://localhost:8000/api/upload", {
    method: "POST",
    body: formData,
  });
  
  return response.json();
}
```

### Start Pipeline

```typescript
async function startPipeline(
  filePath: string, 
  autoApprove: boolean = true
): Promise<PipelineResponse> {
  const response = await fetch("http://localhost:8000/api/pipeline/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_path: filePath,
      auto_approve: autoApprove,
    }),
  });
  
  return response.json();
}
```

### Approve Plan

```typescript
async function approvePlan(
  pipelineId: string, 
  approved: boolean,
  feedback?: string
): Promise<void> {
  await fetch(`http://localhost:8000/api/pipeline/${pipelineId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved, feedback }),
  });
}
```

---

## 📁 Suggested File Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── EventLogPanel.tsx
│   ├── agents/
│   │   ├── AgentVisualization.tsx    # Main tree view
│   │   ├── AgentNode.tsx             # Individual agent node
│   │   ├── MasterAgentNode.tsx
│   │   ├── SubMasterNode.tsx
│   │   ├── WorkerNode.tsx
│   │   └── ConnectionLines.tsx
│   ├── pipeline/
│   │   ├── PipelineProgress.tsx
│   │   ├── PipelineCard.tsx
│   │   └── PlanApprovalModal.tsx
│   ├── upload/
│   │   ├── FileUploader.tsx
│   │   └── FileList.tsx
│   └── results/
│       └── ResultsView.tsx
├── hooks/
│   ├── useWebSocket.ts
│   ├── usePipeline.ts
│   └── useAgents.ts
├── store/
│   ├── index.ts                      # Zustand store
│   ├── pipelineSlice.ts
│   └── eventSlice.ts
├── types/
│   ├── api.ts                        # API response types
│   ├── events.ts                     # Event types
│   └── agents.ts                     # Agent state types
├── lib/
│   ├── api.ts                        # API client functions
│   └── utils.ts
├── App.tsx
└── main.tsx
```

---

## ✅ Acceptance Criteria

1. **Real-time Connection**: WebSocket connects on load, shows connection status
2. **File Upload**: Drag-drop upload works, shows progress
3. **Agent Visualization**: Tree shows all agents with correct hierarchy
4. **Animations**: Agents animate when spawning/processing/completing
5. **Progress Tracking**: Overall progress bar + per-agent progress
6. **Event Log**: Shows real-time event stream, filterable
7. **Plan Approval**: Modal appears when needed, approval works
8. **Results**: Final results display clearly
9. **Error Handling**: Failed states shown properly, reconnection works
10. **Responsive**: Works on desktop and tablet

---

## 🚀 Getting Started Commands

```bash
# Create project
npm create vite@latest agentops-dashboard -- --template react-ts

# Install dependencies
cd agentops-dashboard
npm install tailwindcss postcss autoprefixer
npm install @radix-ui/react-dialog @radix-ui/react-progress
npm install framer-motion
npm install lucide-react
npm install recharts
npm install zustand
npm install clsx tailwind-merge

# Initialize Tailwind
npx tailwindcss init -p

# Install shadcn/ui
npx shadcn@latest init
npx shadcn@latest add button card dialog progress toast
```

---

## 📝 Sample Event Sequence for Testing

When you connect to the WebSocket, you'll receive events like this in order:

```json
{"event_type": "pipeline.started", "pipeline_id": "abc123", "data": {"file": "document.pdf"}}
{"event_type": "pipeline.step_started", "pipeline_id": "abc123", "data": {"step": "extraction"}}
{"event_type": "pipeline.step_completed", "pipeline_id": "abc123", "data": {"step": "extraction"}}
{"event_type": "master.plan_generating", "pipeline_id": "abc123", "agent_id": "master", "agent_type": "master"}
{"event_type": "master.plan_generated", "pipeline_id": "abc123", "data": {"num_submasters": 3}}
{"event_type": "master.plan_approved", "pipeline_id": "abc123"}
{"event_type": "submaster.spawned", "pipeline_id": "abc123", "agent_id": "sm_1", "agent_type": "submaster", "data": {"role": "Introduction Analyst", "page_range": [1, 5]}}
{"event_type": "submaster.spawned", "pipeline_id": "abc123", "agent_id": "sm_2", "agent_type": "submaster", "data": {"role": "Methodology Expert", "page_range": [6, 12]}}
{"event_type": "worker.spawned", "pipeline_id": "abc123", "agent_id": "w_1", "agent_type": "worker", "data": {"submaster_id": "sm_1", "page": 1}}
{"event_type": "worker.spawned", "pipeline_id": "abc123", "agent_id": "w_2", "agent_type": "worker", "data": {"submaster_id": "sm_1", "page": 2}}
{"event_type": "worker.processing", "pipeline_id": "abc123", "agent_id": "w_1", "data": {"page": 1}}
{"event_type": "worker.completed", "pipeline_id": "abc123", "agent_id": "w_1", "data": {"page": 1, "processing_time": 2.3}}
{"event_type": "submaster.progress", "pipeline_id": "abc123", "agent_id": "sm_1", "data": {"completed": 1, "total": 5, "percent": 20}}
{"event_type": "submaster.completed", "pipeline_id": "abc123", "agent_id": "sm_1"}
{"event_type": "pipeline.step_completed", "pipeline_id": "abc123", "data": {"step": "processing"}}
{"event_type": "pipeline.completed", "pipeline_id": "abc123", "data": {"total_time": 45.2, "pages_processed": 15}}
```

---

## 💡 Pro Tips

1. **Use React Query or SWR** for REST API calls with caching
2. **Debounce rapid events** to prevent animation jank
3. **Use CSS transforms** for animations (GPU accelerated)
4. **Implement event batching** for bulk updates
5. **Add keyboard shortcuts** (e.g., `Esc` to close modals)
6. **Store WebSocket connection in a ref** to prevent reconnection on re-renders
7. **Use `requestAnimationFrame`** for smooth progress bar updates

---

Build this dashboard to give users full visibility into the AI agent processing pipeline! 🚀
