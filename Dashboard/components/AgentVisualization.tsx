import React, { useState, useRef } from "react";
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
} from "lucide-react";
import { TreeNode } from "./agent/TreeNode";
import { AgentDetailPanel } from "./agent/AgentDetailPanel";

export const AgentVisualization: React.FC = () => {
  const agents = useStore((state) => state.agents);
  const pipeline = useStore((state) => state.pipeline);
  const agentList = Object.values(agents) as AgentNode[];
  const [selectedAgent, setSelectedAgent] = useState<AgentNode | null>(null);

  const master = agentList.find((a) => a.type === AgentType.MASTER);

  // Calculate stats
  const totalAgents = agentList.length;
  const activeAgents = agentList.filter(
    (a) => a.status === AgentStatus.PROCESSING
  ).length;
  const completedAgents = agentList.filter(
    (a) => a.status === AgentStatus.COMPLETED
  ).length;
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const lastMousePosition = useRef({ x: 0, y: 0 });

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      const scaleSensitivity = 0.001;
      const newScale = Math.min(
        Math.max(0.1, transform.scale - e.deltaY * scaleSensitivity),
        5
      );
      setTransform((prev) => ({ ...prev, scale: newScale }));
    } else {
      // Pan with wheel
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

      {/* Live Stats Bar */}
      <div className="absolute top-0 left-0 right-0 z-20 bg-zinc-900/80 backdrop-blur-md border-b border-zinc-800/50">
        <div className="px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
                <Users size={14} className="text-zinc-400" />
                <span className="text-sm font-medium text-zinc-300">
                  {totalAgents}
                </span>
                <span className="text-xs text-zinc-500">agents</span>
              </div>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <Zap size={14} className="text-blue-400" />
              <span className="text-sm font-medium text-blue-400">
                {activeAgents}
              </span>
              <span className="text-xs text-blue-400/70">active</span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500/10 rounded-lg border border-green-500/20">
              <CheckCircle2 size={14} className="text-green-400" />
              <span className="text-sm font-medium text-green-400">
                {completedAgents}
              </span>
              <span className="text-xs text-green-400/70">done</span>
            </div>
          </div>

          {pipeline.startTime && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Clock size={12} />
              <span>
                Started {new Date(pipeline.startTime).toLocaleTimeString()}
              </span>
            </div>
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
