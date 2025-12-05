import React, { useEffect, useState } from "react";
import { useStore } from "./store";
import { AgentVisualization } from "./components/AgentVisualization";
import { EventLog } from "./components/EventLog";
import { NewPipelinePage } from "./components/NewPipelinePage";
import { RagChat } from "./components/RagChat";
import {
  LayoutDashboard,
  Settings,
  Activity,
  Wifi,
  WifiOff,
  Play,
  MessageSquare,
} from "lucide-react";
import { simulatePipeline } from "./services/mockSimulation";

export default function App() {
  const {
    wsConnected,
    setWsConnected,
    processIncomingEvent,
    pipeline,
    session,
    resetPipeline,
    resetSession,
  } = useStore();

  const [activeTab, setActiveTab] = useState<"upload" | "dashboard" | "chat">(
    "upload"
  );

  // WebSocket Connection Logic
  useEffect(() => {
    let socket: WebSocket;
    let reconnectTimer: any;

    const connect = () => {
      // Connect to the API WebSocket
      try {
        const wsUrl = session.pipelineId
          ? `ws://localhost:8000/api/ws/${session.pipelineId}`
          : "ws://localhost:8000/api/ws";

        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          setWsConnected(true);
          console.log("WebSocket Connected");
        };

        socket.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log("WebSocket message received:", data.type, data);

          // Handle both direct events and wrapped events
          if (data.type === "event" && data.payload) {
            console.log("Processing event:", data.payload.event_type);
            processIncomingEvent(data.payload);
          } else if (data.type === "history" && data.payload?.events) {
            // Process event history
            console.log(
              "Processing history:",
              data.payload.events.length,
              "events"
            );
            data.payload.events.forEach((evt: any) =>
              processIncomingEvent(evt)
            );
          } else {
            console.log("Processing direct event:", data.event_type);
            processIncomingEvent(data);
          }
        };

        socket.onclose = () => {
          setWsConnected(false);
          // Simple reconnect logic
          reconnectTimer = setTimeout(connect, 3000);
        };

        socket.onerror = () => {
          // Silently fail for demo purposes if backend not running
          socket.close();
        };
      } catch (e) {
        console.warn("WS Connection failed", e);
      }
    };

    connect();

    return () => {
      if (socket) socket.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [session.pipelineId]);

  // Handler for uploading file (Simulated - legacy)
  const handleUpload = (file: File) => {
    resetPipeline();
    setActiveTab("dashboard");

    // Simulate API upload delay
    setTimeout(() => {
      // Start simulation
      simulatePipeline(
        "pipe_" + Math.random().toString(36).substr(2, 9),
        (event) => {
          processIncomingEvent(event);
        }
      );
    }, 500);
  };

  const handleSimulate = () => {
    resetPipeline();
    resetSession();
    setActiveTab("dashboard");
    simulatePipeline("pipe_sim_" + Date.now(), processIncomingEvent);
  };

  // Handler for when pipeline actually starts via API
  const handlePipelineStarted = () => {
    setActiveTab("dashboard");
  };

  return (
    <div className="flex h-screen bg-background text-zinc-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-16 md:w-64 flex flex-col border-r border-zinc-800 bg-zinc-900 shrink-0 z-20">
        <div className="h-16 flex items-center px-4 md:px-6 border-b border-surfaceHighlight">
          <Activity className="w-6 h-6 text-primary" />
          <span className="ml-3 font-bold text-lg hidden md:block tracking-tight">
            AgentOps
          </span>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          <button
            onClick={() => setActiveTab("upload")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
              activeTab === "upload"
                ? "bg-primary/10 text-primary"
                : "text-zinc-400 hover:bg-zinc-800"
            }`}
          >
            <LayoutDashboard size={20} />
            <span className="hidden md:block">New Pipeline</span>
          </button>

          <button
            onClick={() => setActiveTab("dashboard")}
            disabled={pipeline.status === "idle"}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
              activeTab === "dashboard"
                ? "bg-primary/10 text-primary"
                : "text-zinc-400 hover:bg-zinc-800"
            } ${
              pipeline.status === "idle" ? "opacity-50 cursor-not-allowed" : ""
            }`}
          >
            <Activity size={20} />
            <span className="hidden md:block">Live View</span>
          </button>

          <button
            onClick={() => setActiveTab("chat")}
            disabled={pipeline.status !== "completed"}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
              activeTab === "chat"
                ? "bg-violet-500/10 text-violet-400"
                : "text-zinc-400 hover:bg-zinc-800"
            } ${
              pipeline.status !== "completed"
                ? "opacity-50 cursor-not-allowed"
                : ""
            }`}
          >
            <MessageSquare size={20} />
            <span className="hidden md:block">Chat (RAG)</span>
          </button>
        </nav>

        <div className="p-4 border-t border-surfaceHighlight">
          <div className="flex items-center gap-3 px-3 py-2 text-zinc-500">
            {wsConnected ? (
              <Wifi className="text-green-500" size={16} />
            ) : (
              <WifiOff className="text-red-500" size={16} />
            )}
            <span className="text-xs hidden md:block">
              {wsConnected ? "System Online" : "Offline Mode"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* Header */}
        <header className="h-16 border-b border-surfaceHighlight bg-background/50 backdrop-blur flex items-center justify-between px-6 z-10">
          <div>
            <h1 className="text-xl font-semibold">
              {activeTab === "upload"
                ? "Initialize Pipeline"
                : activeTab === "chat"
                ? "Document Chat"
                : `Pipeline: ${pipeline.fileName || "Active Session"}`}
            </h1>
            <p className="text-xs text-zinc-500">
              {activeTab === "chat"
                ? "Ask questions about the processed document using RAG"
                : pipeline.status === "idle"
                ? "Ready to process"
                : `Status: ${pipeline.status.toUpperCase()} • Step: ${
                    pipeline.currentStep
                  }`}
            </p>
          </div>

          <div className="flex gap-4">
            {/* Additional header actions can go here */}
          </div>
        </header>

        {/* Viewport */}
        <div className="flex-1 overflow-hidden flex flex-col md:flex-row relative">
          <div className="flex-1 relative overflow-hidden">
            {activeTab === "upload" && (
              <NewPipelinePage
                onUpload={handleUpload}
                onSimulate={handleSimulate}
                isProcessing={pipeline.status === "running"}
                isConnected={wsConnected}
                onPipelineStarted={handlePipelineStarted}
              />
            )}
            {activeTab === "dashboard" && <AgentVisualization />}
            {activeTab === "chat" && <RagChat />}
          </div>

          {/* Event Log Panel - only show in dashboard */}
          {activeTab === "dashboard" && <EventLog />}
        </div>
      </main>
    </div>
  );
}
