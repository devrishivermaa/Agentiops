import React, { useEffect, useState } from "react";
import { useStore } from "./store";
import { AgentVisualization } from "./components/AgentVisualization";
import { EventLog } from "./components/EventLog";
import { FileUploader } from "./components/FileUploader";
import {
  LayoutDashboard,
  Settings,
  History,
  Activity,
  Wifi,
  WifiOff,
  Play,
} from "lucide-react";
import { simulatePipeline } from "./services/mockSimulation";

export default function App() {
  const {
    wsConnected,
    setWsConnected,
    processIncomingEvent,
    pipeline,
    resetPipeline,
  } = useStore();

  const [activeTab, setActiveTab] = useState<"upload" | "dashboard">("upload");

  // WebSocket Connection Logic
  useEffect(() => {
    let socket: WebSocket;
    let reconnectTimer: any;

    const connect = () => {
      // In a real scenario, this connects to ws://localhost:8000/api/ws
      // For this demo, we will simulate a failed connection gracefully if backend isn't there
      // or just stay disconnected until "Simulation" is triggered.
      try {
        socket = new WebSocket("ws://localhost:8000/api/ws");

        socket.onopen = () => {
          setWsConnected(true);
          console.log("WebSocket Connected");
        };

        socket.onmessage = (event) => {
          const data = JSON.parse(event.data);
          processIncomingEvent(data);
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
  }, []);

  // Handler for uploading file (Simulated)
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
    setActiveTab("dashboard");
    simulatePipeline("pipe_sim_" + Date.now(), processIncomingEvent);
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

          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 transition-colors">
            <History size={20} />
            <span className="hidden md:block">History</span>
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
                : `Pipeline: ${pipeline.fileName || "Active Session"}`}
            </h1>
            <p className="text-xs text-zinc-500">
              {pipeline.status === "idle"
                ? "Ready to process"
                : `Status: ${pipeline.status.toUpperCase()} • Step: ${
                    pipeline.currentStep
                  }`}
            </p>
          </div>

          <div className="flex gap-4">
            {!wsConnected && activeTab === "upload" && (
              <button
                onClick={handleSimulate}
                className="flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-4 py-2 rounded-md text-sm border border-zinc-700 transition-colors"
              >
                <Play size={14} />
                Run Simulation
              </button>
            )}
          </div>
        </header>

        {/* Viewport */}
        <div className="flex-1 overflow-hidden flex flex-col md:flex-row relative">
          <div className="flex-1 relative overflow-hidden bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-opacity-5">
            {activeTab === "upload" && (
              <FileUploader
                onUpload={handleUpload}
                isProcessing={pipeline.status === "running"}
              />
            )}
            {activeTab === "dashboard" && <AgentVisualization />}
          </div>

          {/* Event Log Panel */}
          <EventLog />
        </div>
      </main>
    </div>
  );
}
