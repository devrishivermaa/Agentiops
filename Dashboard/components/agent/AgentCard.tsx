import React from 'react';
import { motion } from 'framer-motion';
import { AgentNode, AgentStatus, AgentType } from '../../types';
import { BrainCircuit, BookOpen, FileText, Loader2, Check, AlertTriangle, ChevronRight } from 'lucide-react';
import { spawnVariants } from './animations';

interface AgentCardProps { 
  agent: AgentNode; 
  onClick: (agent: AgentNode) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({ agent, onClick }) => {
  const isMaster = agent.type === AgentType.MASTER;
  const isSubMaster = agent.type === AgentType.SUBMASTER;

  const statusConfig = {
    [AgentStatus.SPAWNED]: { border: "border-zinc-700", bg: "bg-zinc-900", text: "text-zinc-400" },
    [AgentStatus.INITIALIZING]: { border: "border-yellow-600/50", bg: "bg-yellow-950/20", text: "text-yellow-400" },
    [AgentStatus.PROCESSING]: { border: "border-blue-500", bg: "bg-blue-950/30", text: "text-blue-400" },
    [AgentStatus.COMPLETED]: { border: "border-green-600/50", bg: "bg-green-950/20", text: "text-green-400" },
    [AgentStatus.FAILED]: { border: "border-red-600", bg: "bg-red-950/20", text: "text-red-400" },
    [AgentStatus.WAITING]: { border: "border-zinc-800", bg: "bg-zinc-900/50", text: "text-zinc-600" },
  };

  const currentStatus = statusConfig[agent.status] || statusConfig[AgentStatus.SPAWNED];

  const StatusIcon = () => {
    if (agent.status === AgentStatus.PROCESSING) return <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />;
    if (agent.status === AgentStatus.COMPLETED) return <Check className="w-3.5 h-3.5 text-green-400" />;
    if (agent.status === AgentStatus.FAILED) return <AlertTriangle className="w-3.5 h-3.5 text-red-400" />;
    return <div className="w-1.5 h-1.5 rounded-full bg-zinc-600" />;
  };

  return (
    <motion.div
      layoutId={agent.id}
      variants={spawnVariants}
      initial="hidden"
      animate="visible"
      whileHover={{ scale: 1.03, y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => onClick(agent)}
      className={`
        relative flex flex-col items-center p-3 rounded-xl border transition-all duration-300 z-10 cursor-pointer group shadow-lg
        ${currentStatus.border} ${currentStatus.bg}
        ${isMaster ? 'w-56 min-h-[120px]' : isSubMaster ? 'w-44 min-h-[120px]' : 'w-28 min-h-[90px]'}
        ${agent.status === AgentStatus.PROCESSING ? 'shadow-[0_0_20px_rgba(59,130,246,0.15)]' : ''}
      `}
    >
      {/* Icon Badge */}
      <div className={`
        rounded-full p-2.5 mb-3 shadow-inner ring-1 ring-inset ring-white/5
        ${isMaster ? 'bg-purple-500/10 text-purple-400' : isSubMaster ? 'bg-amber-500/10 text-amber-400' : 'bg-zinc-800 text-zinc-400'}
      `}>
        {isMaster ? <BrainCircuit size={24} /> : isSubMaster ? <BookOpen size={20} /> : <FileText size={16} />}
      </div>
      
      {/* Label & Meta */}
      <div className="text-center w-full">
        <h4 className={`font-bold truncate px-2 text-zinc-200 ${isMaster ? 'text-base' : 'text-xs'}`}>
          {agent.label}
        </h4>
        
        {agent.metadata?.pages && (
          <p className="text-[10px] text-zinc-500 mt-1 font-medium bg-black/20 rounded px-1.5 py-0.5 inline-block border border-white/5">
             {Array.isArray(agent.metadata.pages) 
                ? `Pages ${agent.metadata.pages[0]}-${agent.metadata.pages[1]}`
                : `Page ${agent.metadata.pages}`}
          </p>
        )}
      </div>

      {/* Status Indicator (Top Right) */}
      <div className="absolute top-2 right-2 p-1.5 rounded-full bg-black/20 border border-white/5">
        <StatusIcon />
      </div>

      {/* Progress Bar for SubMaster/Master */}
      {(isSubMaster || isMaster) && typeof agent.metadata?.progress === 'number' && (
        <div className="w-full px-3 mt-auto pt-2">
            <div className="flex justify-between text-[9px] text-zinc-500 mb-1">
                <span>Progress</span>
                <span>{Math.round(agent.metadata.progress)}%</span>
            </div>
            <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden border border-white/5">
            <motion.div 
                className="h-full bg-gradient-to-r from-blue-600 to-blue-400"
                initial={{ width: 0 }}
                animate={{ width: `${agent.metadata.progress}%` }}
                transition={{ type: "spring", stiffness: 50 }}
            />
            </div>
        </div>
      )}
      
      {/* Hover 'Expand' hint */}
      <div className="absolute -bottom-6 opacity-0 group-hover:opacity-100 transition-opacity text-[10px] text-zinc-500 flex items-center gap-1">
         View Details <ChevronRight size={10} />
      </div>
    </motion.div>
  );
};