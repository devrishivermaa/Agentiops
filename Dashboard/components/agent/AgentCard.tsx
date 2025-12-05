import React from "react";
import { motion } from "framer-motion";
import { AgentNode, AgentStatus, AgentType } from "../../types";
import {
  BrainCircuit,
  BookOpen,
  FileText,
  Loader2,
  Check,
  AlertTriangle,
  ChevronRight,
  Cpu,
  Globe,
  Layers,
  GitMerge,
  FileOutput,
  Brain,
  Sparkles,
} from "lucide-react";
import { spawnVariants } from "./animations";

interface AgentCardProps {
  agent: AgentNode;
  onClick: (agent: AgentNode) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({ agent, onClick }) => {
  const isMaster = agent.type === AgentType.MASTER;
  const isSubMaster = agent.type === AgentType.SUBMASTER;
  const isResidual = agent.type === AgentType.RESIDUAL;
  const isReducer = agent.type === AgentType.REDUCER;

  // New Reducer Pipeline Agent Types
  const isReducerSubmaster = agent.type === AgentType.REDUCER_SUBMASTER;
  const isReducerResidual = agent.type === AgentType.REDUCER_RESIDUAL;
  const isMasterMerger = agent.type === AgentType.MASTER_MERGER;
  const isPdfGenerator = agent.type === AgentType.PDF_GENERATOR;

  // Check if this is a reducer pipeline agent
  const isReducerPipelineAgent =
    isReducerSubmaster || isReducerResidual || isMasterMerger || isPdfGenerator;

  const statusConfig = {
    [AgentStatus.SPAWNED]: {
      border: "border-zinc-700/50",
      bg: "bg-zinc-900/80",
      glow: "",
      ring: "ring-zinc-700/30",
    },
    [AgentStatus.INITIALIZING]: {
      border: "border-yellow-500/30",
      bg: "bg-gradient-to-br from-yellow-950/30 to-zinc-900",
      glow: "shadow-[0_0_30px_rgba(234,179,8,0.1)]",
      ring: "ring-yellow-500/20",
    },
    [AgentStatus.PROCESSING]: {
      border: "border-blue-500/40",
      bg: "bg-gradient-to-br from-blue-950/40 to-zinc-900",
      glow: "shadow-[0_0_40px_rgba(59,130,246,0.15)]",
      ring: "ring-blue-500/30",
    },
    [AgentStatus.COMPLETED]: {
      border: "border-green-500/30",
      bg: "bg-gradient-to-br from-green-950/30 to-zinc-900",
      glow: "shadow-[0_0_30px_rgba(34,197,94,0.1)]",
      ring: "ring-green-500/20",
    },
    [AgentStatus.FAILED]: {
      border: "border-red-500/40",
      bg: "bg-gradient-to-br from-red-950/40 to-zinc-900",
      glow: "shadow-[0_0_30px_rgba(239,68,68,0.15)]",
      ring: "ring-red-500/30",
    },
    [AgentStatus.WAITING]: {
      border: "border-zinc-800",
      bg: "bg-zinc-900/60",
      glow: "",
      ring: "ring-zinc-800/50",
    },
  };

  const currentStatus =
    statusConfig[agent.status] || statusConfig[AgentStatus.SPAWNED];

  const StatusIcon = () => {
    if (agent.status === AgentStatus.PROCESSING) {
      return (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        >
          <Loader2 className="w-3.5 h-3.5 text-blue-400" />
        </motion.div>
      );
    }
    if (agent.status === AgentStatus.COMPLETED)
      return <Check className="w-3.5 h-3.5 text-green-400" />;
    if (agent.status === AgentStatus.FAILED)
      return <AlertTriangle className="w-3.5 h-3.5 text-red-400" />;
    if (agent.status === AgentStatus.INITIALIZING)
      return <Cpu className="w-3.5 h-3.5 text-yellow-400 animate-pulse" />;
    return <div className="w-2 h-2 rounded-full bg-zinc-600" />;
  };

  const getStatusLabel = () => {
    switch (agent.status) {
      case AgentStatus.PROCESSING:
        return "Processing";
      case AgentStatus.COMPLETED:
        return "Done";
      case AgentStatus.FAILED:
        return "Failed";
      case AgentStatus.INITIALIZING:
        return "Init";
      case AgentStatus.WAITING:
        return "Waiting";
      default:
        return "Ready";
    }
  };

  return (
    <motion.div
      layoutId={agent.id}
      variants={spawnVariants}
      initial="hidden"
      animate="visible"
      whileHover={{
        scale: 1.03,
        y: -6,
        transition: { type: "spring", stiffness: 400, damping: 25 },
      }}
      whileTap={{ scale: 0.98 }}
      onClick={() => onClick(agent)}
      className={`
        relative flex flex-col items-center p-4 rounded-2xl border backdrop-blur-sm
        transition-colors duration-300 z-10 cursor-pointer group
        ring-1 ${currentStatus.ring}
        ${currentStatus.border} ${currentStatus.bg} ${currentStatus.glow}
        ${
          isMaster
            ? "w-60 min-h-[140px]"
            : isSubMaster
            ? "w-48 min-h-[130px]"
            : isResidual
            ? "w-44 min-h-[120px]"
            : isReducer
            ? "w-52 min-h-[130px]"
            : isReducerSubmaster
            ? "w-52 min-h-[130px]"
            : isReducerResidual
            ? "w-48 min-h-[120px]"
            : isMasterMerger
            ? "w-52 min-h-[130px]"
            : isPdfGenerator
            ? "w-44 min-h-[110px]"
            : "w-32 min-h-[100px]"
        }
      `}
    >
      {/* Animated background pulse for processing */}
      {agent.status === AgentStatus.PROCESSING && (
        <motion.div
          className="absolute inset-0 rounded-2xl bg-blue-500/5"
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}

      {/* Icon Badge */}
      <div
        className={`
        relative rounded-xl p-3 mb-3 ring-1 ring-inset ring-white/10
        ${
          isMaster
            ? "bg-gradient-to-br from-purple-500/20 to-purple-600/10 text-purple-400"
            : isSubMaster
            ? "bg-gradient-to-br from-amber-500/20 to-amber-600/10 text-amber-400"
            : isResidual
            ? "bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 text-cyan-400"
            : isReducer
            ? "bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 text-emerald-400"
            : isReducerSubmaster
            ? "bg-gradient-to-br from-purple-500/20 to-purple-600/10 text-purple-400"
            : isReducerResidual
            ? "bg-gradient-to-br from-pink-500/20 to-pink-600/10 text-pink-400"
            : isMasterMerger
            ? "bg-gradient-to-br from-orange-500/20 to-orange-600/10 text-orange-400"
            : isPdfGenerator
            ? "bg-gradient-to-br from-rose-500/20 to-rose-600/10 text-rose-400"
            : "bg-gradient-to-br from-zinc-700/50 to-zinc-800/50 text-zinc-400"
        }
      `}
      >
        {isMaster ? (
          <BrainCircuit size={26} />
        ) : isSubMaster ? (
          <BookOpen size={22} />
        ) : isResidual ? (
          <Globe size={22} />
        ) : isReducer ? (
          <Layers size={24} />
        ) : isReducerSubmaster ? (
          <Layers size={22} />
        ) : isReducerResidual ? (
          <Brain size={22} />
        ) : isMasterMerger ? (
          <GitMerge size={22} />
        ) : isPdfGenerator ? (
          <FileOutput size={20} />
        ) : (
          <FileText size={18} />
        )}

        {/* Live indicator for processing */}
        {agent.status === AgentStatus.PROCESSING && (
          <motion.div
            className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full"
            animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
      </div>

      {/* Label & Meta */}
      <div className="text-center w-full space-y-1.5">
        <h4
          className={`font-semibold truncate px-1 text-zinc-100 ${
            isMaster ? "text-sm" : "text-xs"
          }`}
        >
          {agent.label}
        </h4>

        <div className="flex items-center justify-center gap-1.5">
          {agent.metadata?.pages && (
            <span className="text-[10px] text-zinc-500 font-medium bg-zinc-800/80 rounded-md px-2 py-0.5 border border-zinc-700/50">
              {Array.isArray(agent.metadata.pages)
                ? `P${agent.metadata.pages[0]}-${agent.metadata.pages[1]}`
                : `P${agent.metadata.pages}`}
            </span>
          )}
        </div>
      </div>

      {/* Status Badge (Top Right) */}
      <div
        className={`
        absolute top-3 right-3 flex items-center gap-1.5 px-2 py-1 rounded-lg
        bg-zinc-900/80 border border-zinc-700/50 backdrop-blur-sm
      `}
      >
        <StatusIcon />
        <span
          className={`text-[10px] font-medium ${
            agent.status === AgentStatus.PROCESSING
              ? "text-blue-400"
              : agent.status === AgentStatus.COMPLETED
              ? "text-green-400"
              : agent.status === AgentStatus.FAILED
              ? "text-red-400"
              : "text-zinc-500"
          }`}
        >
          {getStatusLabel()}
        </span>
      </div>

      {/* Progress Bar for SubMaster/Master/Residual and Reducer Pipeline Agents */}
      {(isSubMaster || isMaster || isResidual || isReducerPipelineAgent) &&
        typeof agent.metadata?.progress === "number" && (
          <div className="w-full px-1 mt-auto pt-3">
            <div className="flex justify-between text-[10px] text-zinc-500 mb-1.5 font-medium">
              <span>Progress</span>
              <span
                className={
                  agent.metadata.progress === 100
                    ? "text-green-400"
                    : "text-zinc-400"
                }
              >
                {Math.round(agent.metadata.progress)}%
              </span>
            </div>
            <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden ring-1 ring-inset ring-zinc-700/50">
              <motion.div
                className={`h-full rounded-full ${
                  agent.metadata.progress === 100
                    ? "bg-gradient-to-r from-green-600 to-green-400"
                    : "bg-gradient-to-r from-blue-600 via-blue-500 to-blue-400"
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${agent.metadata.progress}%` }}
                transition={{ type: "spring", stiffness: 40, damping: 15 }}
              />
            </div>
          </div>
        )}

      {/* Hover hint */}
      <motion.div
        className="absolute -bottom-7 flex items-center gap-1 px-2 py-1 bg-zinc-800/90 rounded-lg border border-zinc-700/50 opacity-0 group-hover:opacity-100 transition-opacity"
        initial={{ y: -4 }}
      >
        <span className="text-[10px] text-zinc-400">View Details</span>
        <ChevronRight size={10} className="text-zinc-500" />
      </motion.div>
    </motion.div>
  );
};
