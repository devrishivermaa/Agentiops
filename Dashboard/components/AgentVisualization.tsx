import React, { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { useStore } from '../store';
import { AgentNode, AgentType } from '../types';
import { BrainCircuit, Loader2 } from 'lucide-react';
import { TreeNode } from './agent/TreeNode';
import { AgentDetailPanel } from './agent/AgentDetailPanel';

export const AgentVisualization: React.FC = () => {
  const agents = useStore((state) => state.agents);
  const agentList = Object.values(agents) as AgentNode[];
  const [selectedAgent, setSelectedAgent] = useState<AgentNode | null>(null);
  
  const master = agentList.find(a => a.type === AgentType.MASTER);

  if (!master) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 space-y-4 animate-in fade-in duration-700">
        <div className="relative">
           <BrainCircuit size={64} className="text-zinc-800" />
           <Loader2 className="absolute -bottom-2 -right-2 w-6 h-6 animate-spin text-primary" />
        </div>
        <p className="text-sm tracking-wider uppercase">Waiting for pipeline initialization...</p>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full overflow-auto bg-grid-pattern">
      {/* Background grid effect */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none" />
      
      <div className="min-w-max p-12 flex justify-center pb-32">
        <TreeNode 
          node={master} 
          allAgents={agentList} 
          onSelect={setSelectedAgent} 
        />
      </div>

      <AnimatePresence>
        {selectedAgent && (
          <AgentDetailPanel 
            agent={selectedAgent} 
            onClose={() => setSelectedAgent(null)} 
          />
        )}
      </AnimatePresence>
    </div>
  );
};