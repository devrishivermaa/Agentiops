import React, { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "../store";
import { AgentNode, AgentType, AgentStatus } from "../types";
import {
  BrainCircuit,
  Loader2,
  Users,
  Zap,
  Clock,
  CheckCircle2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  FileText,
  FileJson,
  Globe,
  Layers,
  Timer,
  Activity,
  TrendingUp,
  GitMerge,
  FileOutput,
  Brain,
  ChevronDown,
  ChevronUp,
  BarChart3,
} from "lucide-react";
import { TreeNode } from "./agent/TreeNode";
import { AgentDetailPanel } from "./agent/AgentDetailPanel";

// Format elapsed time nicely
function formatElapsedTime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

export const AgentVisualization: React.FC = () => {
  const agents = useStore((state) => state.agents);
  const pipeline = useStore((state) => state.pipeline);
  const downloadReport = useStore((state) => state.downloadReport);
  const downloadJson = useStore((state) => state.downloadJson);
  const outputPaths = useStore((state) => state.outputPaths);
  const agentList = Object.values(agents) as AgentNode[];
  const [selectedAgent, setSelectedAgent] = useState<AgentNode | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [metricsExpanded, setMetricsExpanded] = useState(false);

  const master = agentList.find((a) => a.type === AgentType.MASTER);
  const residual = agentList.find((a) => a.type === AgentType.RESIDUAL);
  const reducer = agentList.find((a) => a.type === AgentType.REDUCER);
  const submasters = agentList.filter((a) => a.type === AgentType.SUBMASTER);
  const workers = agentList.filter((a) => a.type === AgentType.WORKER);

  // Reducer Pipeline Agents
  const reducerSubmasters = agentList.find(
    (a) => a.type === AgentType.REDUCER_SUBMASTER
  );
  const reducerResidual = agentList.find(
    (a) => a.type === AgentType.REDUCER_RESIDUAL
  );
  const masterMerger = agentList.find(
    (a) => a.type === AgentType.MASTER_MERGER
  );
  const pdfGenerator = agentList.find(
    (a) => a.type === AgentType.PDF_GENERATOR
  );

  // Calculate stats
  const totalAgents = agentList.length;
  const activeAgents = agentList.filter(
    (a) => a.status === AgentStatus.PROCESSING
  ).length;
  const completedAgents = agentList.filter(
    (a) => a.status === AgentStatus.COMPLETED
  ).length;

  // Elapsed time tracker
  useEffect(() => {
    if (!pipeline.startTime) {
      setElapsedTime(0);
      return;
    }

    if (pipeline.status === "completed") {
      // Set final elapsed time
      setElapsedTime(Date.now() - pipeline.startTime);
      return;
    }

    // Update every second while running
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - pipeline.startTime!);
    }, 1000);

    return () => clearInterval(interval);
  }, [pipeline.startTime, pipeline.status]);

  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const lastMousePosition = useRef({ x: 0, y: 0 });

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();

    if (e.ctrlKey || e.metaKey) {
      // Zoom with Ctrl/Cmd + scroll
      const scaleSensitivity = 0.001;
      const newScale = Math.min(
        Math.max(0.1, transform.scale - e.deltaY * scaleSensitivity),
        5
      );
      setTransform((prev) => ({ ...prev, scale: newScale }));
    } else if (e.shiftKey) {
      // Horizontal scroll with Shift + scroll wheel
      setTransform((prev) => ({
        ...prev,
        x: prev.x - e.deltaY,
      }));
    } else {
      // Pan with wheel (supports both horizontal and vertical)
      setTransform((prev) => ({
        ...prev,
        x: prev.x - e.deltaX,
        y: prev.y - e.deltaY,
      }));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      // Left click
      isDragging.current = true;
      lastMousePosition.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) {
      const dx = e.clientX - lastMousePosition.current.x;
      const dy = e.clientY - lastMousePosition.current.y;
      setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
      lastMousePosition.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleZoomIn = () => {
    setTransform((prev) => ({
      ...prev,
      scale: Math.min(prev.scale + 0.25, 3),
    }));
  };

  const handleZoomOut = () => {
    setTransform((prev) => ({
      ...prev,
      scale: Math.max(prev.scale - 0.25, 0.25),
    }));
  };

  const handleResetView = () => {
    setTransform({ x: 0, y: 0, scale: 1 });
  };

  return (
    <div
      className="relative w-full h-full overflow-hidden bg-zinc-950 select-none"
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Gradient Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950 via-zinc-900/50 to-zinc-950 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/5 via-transparent to-transparent pointer-events-none" />

      {/* Grid pattern */}
      <div
        className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.02)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none"
        style={{
          transform: `translate(${transform.x % 64}px, ${
            transform.y % 64
          }px) scale(${transform.scale})`,
          transformOrigin: "0 0",
        }}
      />

      {/* Live Stats Bar - Compact with Collapsible Details */}
      <div className="absolute top-0 left-0 right-0 z-20 bg-zinc-900/90 backdrop-blur-xl border-b border-zinc-800/50">
        <div className="px-4 md:px-6 py-2">
          {/* Compact Header Row - Always Visible */}
          <div className="flex items-center justify-between gap-4">
            {/* Left: Compact Stats + Toggle */}
            <div className="flex items-center gap-3">
              {/* Toggle Button */}
              <motion.button
                onClick={() => setMetricsExpanded(!metricsExpanded)}
                className="flex items-center gap-2 px-3 py-2 bg-zinc-800/60 hover:bg-zinc-700/60 rounded-xl border border-zinc-700/50 transition-all"
                whileTap={{ scale: 0.95 }}
              >
                <BarChart3 size={16} className="text-zinc-400" />
                <span className="text-sm font-medium text-zinc-300">
                  Metrics
                </span>
                <motion.div
                  animate={{ rotate: metricsExpanded ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronDown size={14} className="text-zinc-500" />
                </motion.div>
              </motion.button>

              {/* Mini Stats (always visible) */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-zinc-500">{totalAgents} agents</span>
                <span className="text-zinc-600">•</span>
                <span className="text-blue-400">{activeAgents} active</span>
                <span className="text-zinc-600">•</span>
                <span className="text-green-400">{completedAgents} done</span>
                {pipeline.startTime && (
                  <>
                    <span className="text-zinc-600">•</span>
                    <span
                      className={
                        pipeline.status === "completed"
                          ? "text-violet-400"
                          : "text-amber-400"
                      }
                    >
                      {formatElapsedTime(elapsedTime)}
                    </span>
                  </>
                )}
              </div>

              {/* Status indicator */}
              {pipeline.status !== "idle" && (
                <div
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg ${
                    pipeline.status === "completed"
                      ? "bg-green-500/10 border border-green-500/20"
                      : pipeline.status === "failed"
                      ? "bg-red-500/10 border border-red-500/20"
                      : "bg-blue-500/10 border border-blue-500/20"
                  }`}
                >
                  {pipeline.status === "running" && (
                    <Loader2 size={12} className="text-blue-400 animate-spin" />
                  )}
                  <span
                    className={`text-xs font-medium capitalize ${
                      pipeline.status === "completed"
                        ? "text-green-400"
                        : pipeline.status === "failed"
                        ? "text-red-400"
                        : "text-blue-400"
                    }`}
                  >
                    {pipeline.status}
                  </span>
                </div>
              )}
            </div>

            {/* Right: Download Buttons (always visible) */}
            <div className="flex items-center gap-2">
              {pipeline.status === "completed" && (
                <motion.div
                  className="flex items-center gap-2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                >
                  <button
                    onClick={downloadReport}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-primary/20 to-violet-500/20 hover:from-primary/30 hover:to-violet-500/30 rounded-xl border border-primary/30 transition-all group"
                  >
                    <FileText
                      size={16}
                      className="text-primary group-hover:scale-110 transition-transform"
                    />
                    <span className="text-sm font-medium text-primary hidden sm:block">
                      Report
                    </span>
                    <Download size={14} className="text-primary/70" />
                  </button>
                  <button
                    onClick={downloadJson}
                    className="flex items-center gap-2 px-4 py-2 bg-zinc-800/60 hover:bg-zinc-700/60 rounded-xl border border-zinc-700/50 transition-all group"
                  >
                    <FileJson
                      size={16}
                      className="text-zinc-400 group-hover:scale-110 transition-transform"
                    />
                    <span className="text-sm font-medium text-zinc-300 hidden sm:block">
                      JSON
                    </span>
                    <Download size={14} className="text-zinc-500" />
                  </button>
                </motion.div>
              )}
            </div>
          </div>

          {/* Collapsible Detailed Metrics */}
          <AnimatePresence>
            {metricsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="pt-3 mt-3 border-t border-zinc-800/50">
                  {/* Detailed Stats Row */}
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    {/* Total Agents */}
                    <motion.div
                      className="flex items-center gap-2 px-3 py-2 bg-zinc-800/60 rounded-xl border border-zinc-700/50"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="w-8 h-8 rounded-lg bg-zinc-700/50 flex items-center justify-center">
                        <Users size={16} className="text-zinc-300" />
                      </div>
                      <div>
                        <div className="text-lg font-bold text-zinc-100">
                          {totalAgents}
                        </div>
                        <div className="text-[10px] text-zinc-500 uppercase tracking-wide">
                          Total Agents
                        </div>
                      </div>
                    </motion.div>

                    {/* Active */}
                    <motion.div
                      className="flex items-center gap-2 px-3 py-2 bg-blue-500/10 rounded-xl border border-blue-500/20"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2, delay: 0.05 }}
                    >
                      <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                        <Zap size={16} className="text-blue-400" />
                      </div>
                      <div>
                        <div className="text-lg font-bold text-blue-400">
                          {activeAgents}
                        </div>
                        <div className="text-[10px] text-blue-400/70 uppercase tracking-wide">
                          Active
                        </div>
                      </div>
                    </motion.div>

                    {/* Completed */}
                    <motion.div
                      className="flex items-center gap-2 px-3 py-2 bg-green-500/10 rounded-xl border border-green-500/20"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2, delay: 0.1 }}
                    >
                      <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
                        <CheckCircle2 size={16} className="text-green-400" />
                      </div>
                      <div>
                        <div className="text-lg font-bold text-green-400">
                          {completedAgents}
                        </div>
                        <div className="text-[10px] text-green-400/70 uppercase tracking-wide">
                          Completed
                        </div>
                      </div>
                    </motion.div>

                    {/* Elapsed Time */}
                    {pipeline.startTime && (
                      <motion.div
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl border ${
                          pipeline.status === "completed"
                            ? "bg-violet-500/10 border-violet-500/20"
                            : "bg-amber-500/10 border-amber-500/20"
                        }`}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.2, delay: 0.15 }}
                      >
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                            pipeline.status === "completed"
                              ? "bg-violet-500/20"
                              : "bg-amber-500/20"
                          }`}
                        >
                          <Timer
                            size={16}
                            className={
                              pipeline.status === "completed"
                                ? "text-violet-400"
                                : "text-amber-400"
                            }
                          />
                        </div>
                        <div>
                          <div
                            className={`text-lg font-bold font-mono ${
                              pipeline.status === "completed"
                                ? "text-violet-400"
                                : "text-amber-400"
                            }`}
                          >
                            {formatElapsedTime(elapsedTime)}
                          </div>
                          <div
                            className={`text-[10px] uppercase tracking-wide ${
                              pipeline.status === "completed"
                                ? "text-violet-400/70"
                                : "text-amber-400/70"
                            }`}
                          >
                            {pipeline.status === "completed"
                              ? "Total Time"
                              : "Elapsed"}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </div>

                  {/* Architecture Pills */}
                  <div className="flex flex-wrap items-center gap-2">
                    {residual && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        onClick={() => setSelectedAgent(residual)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-500/10 rounded-lg border border-cyan-500/20 hover:bg-cyan-500/20 transition-all"
                      >
                        <Globe size={14} className="text-cyan-400" />
                        <span className="text-xs font-medium text-cyan-400">
                          Residual
                        </span>
                      </motion.button>
                    )}
                    {reducer && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.05 }}
                        onClick={() => setSelectedAgent(reducer)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 rounded-lg border border-emerald-500/20 hover:bg-emerald-500/20 transition-all"
                      >
                        <Layers size={14} className="text-emerald-400" />
                        <span className="text-xs font-medium text-emerald-400">
                          Reducer
                        </span>
                      </motion.button>
                    )}
                    {reducerSubmasters && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        onClick={() => setSelectedAgent(reducerSubmasters)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 rounded-lg border border-purple-500/20 hover:bg-purple-500/20 transition-all"
                      >
                        <Layers size={14} className="text-purple-400" />
                        <span className="text-xs font-medium text-purple-400">
                          Reducer SM
                        </span>
                      </motion.button>
                    )}
                    {reducerResidual && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.15 }}
                        onClick={() => setSelectedAgent(reducerResidual)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-pink-500/10 rounded-lg border border-pink-500/20 hover:bg-pink-500/20 transition-all"
                      >
                        <Brain size={14} className="text-pink-400" />
                        <span className="text-xs font-medium text-pink-400">
                          Context
                        </span>
                      </motion.button>
                    )}
                    {masterMerger && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        onClick={() => setSelectedAgent(masterMerger)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-500/10 rounded-lg border border-orange-500/20 hover:bg-orange-500/20 transition-all"
                      >
                        <GitMerge size={14} className="text-orange-400" />
                        <span className="text-xs font-medium text-orange-400">
                          Merger
                        </span>
                      </motion.button>
                    )}
                    {pdfGenerator && (
                      <motion.button
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.25 }}
                        onClick={() => setSelectedAgent(pdfGenerator)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-500/10 rounded-lg border border-rose-500/20 hover:bg-rose-500/20 transition-all"
                      >
                        <FileOutput size={14} className="text-rose-400" />
                        <span className="text-xs font-medium text-rose-400">
                          PDF
                        </span>
                      </motion.button>
                    )}
                    {submasters.length > 0 && (
                      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/10 rounded-lg border border-amber-500/20">
                        <Activity size={14} className="text-amber-400" />
                        <span className="text-xs font-medium text-amber-400">
                          {submasters.length} SubMasters
                        </span>
                      </div>
                    )}
                    {workers.length > 0 && (
                      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800/60 rounded-lg border border-zinc-700/50">
                        <TrendingUp size={14} className="text-zinc-400" />
                        <span className="text-xs font-medium text-zinc-300">
                          {workers.length} Workers
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Current Step (shown below when running) */}
          {pipeline.status === "running" && pipeline.currentStep && (
            <motion.div
              className="mt-2 pt-2 border-t border-zinc-800/50"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
            >
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                <span className="text-zinc-500">Current:</span>
                <span className="text-zinc-300 font-medium">
                  {pipeline.currentStep}
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Zoom Controls */}
      <div className="absolute bottom-6 right-6 z-20 flex flex-col gap-2">
        <div className="flex flex-col bg-zinc-900/90 backdrop-blur-md rounded-lg border border-zinc-700/50 overflow-hidden shadow-lg">
          <button
            onClick={handleZoomIn}
            className="p-2.5 hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors border-b border-zinc-700/50"
            title="Zoom In"
          >
            <ZoomIn size={18} />
          </button>
          <div className="px-2 py-1.5 text-xs text-center text-zinc-500 font-mono border-b border-zinc-700/50 bg-zinc-800/50">
            {Math.round(transform.scale * 100)}%
          </div>
          <button
            onClick={handleZoomOut}
            className="p-2.5 hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
            title="Zoom Out"
          >
            <ZoomOut size={18} />
          </button>
        </div>
        <button
          onClick={handleResetView}
          className="p-2.5 bg-zinc-900/90 backdrop-blur-md rounded-lg border border-zinc-700/50 hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors shadow-lg"
          title="Reset View"
        >
          <Maximize2 size={18} />
        </button>
      </div>

      {/* Agent Tree Container */}
      <div
        className="w-full h-full flex items-center justify-center origin-center transition-transform duration-75 ease-out"
        style={{
          transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
        }}
      >
        {master ? (
          <div className="min-w-max p-8 md:p-12 flex justify-center pb-32">
            <TreeNode
              node={master}
              allAgents={agentList}
              onSelect={setSelectedAgent}
            />
          </div>
        ) : (
          <div
            className="flex flex-col items-center justify-center text-zinc-500 space-y-6 animate-in fade-in duration-700 scale-100"
            style={{ transform: `scale(${1 / transform.scale})` }}
          >
            <div className="relative">
              <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full" />
              <div className="relative bg-zinc-900/80 p-6 rounded-2xl border border-zinc-800 backdrop-blur">
                <BrainCircuit size={56} className="text-zinc-600" />
                <motion.div
                  className="absolute -bottom-2 -right-2 bg-zinc-900 p-1.5 rounded-full border border-zinc-700"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <Loader2 className="w-5 h-5 text-primary" />
                </motion.div>
              </div>
            </div>
            <div className="text-center space-y-2">
              <p className="text-sm font-medium text-zinc-400">
                Initializing Pipeline
              </p>
              <p className="text-xs text-zinc-600">
                Waiting for agent orchestration...
              </p>
            </div>
          </div>
        )}
      </div>

      <AnimatePresence>
        {selectedAgent && (
          <AgentDetailPanel
            key="agent-detail-panel"
            agent={selectedAgent}
            onClose={() => setSelectedAgent(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};
