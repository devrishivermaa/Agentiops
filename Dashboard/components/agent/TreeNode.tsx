import React from 'react';
import { motion } from 'framer-motion';
import { AgentNode } from '../../types';
import { AgentCard } from './AgentCard';
import { BranchConnector } from './TreeConnector';

interface TreeNodeProps {
  node: AgentNode;
  allAgents: AgentNode[];
  onSelect: (agent: AgentNode) => void;
  depth?: number;
}

export const TreeNode: React.FC<TreeNodeProps> = ({ node, allAgents, onSelect, depth = 0 }) => {
  // Find children
  const children = allAgents.filter(a => a.parentId === node.id);
  const hasChildren = children.length > 0;

  return (
    <div className="flex flex-col items-center relative">
      <AgentCard agent={node} onClick={onSelect} />
      
      {hasChildren && (
        <>
          {/* Vertical line down from parent */}
          <div className="h-8 w-full relative">
             <BranchConnector type="vertical" />
          </div>

          <div className="flex relative pt-4">
             {/* Horizontal connector bar logic handled by children wrappers */}
             {children.map((child, index) => {
               const isFirst = index === 0;
               const isLast = index === children.length - 1;
               const isOnly = children.length === 1;

               return (
                 <div key={child.id} className="flex flex-col items-center relative px-2 md:px-4">
                   {/* Top connectors for children */}
                   <div className="absolute top-0 left-0 right-0 h-4">
                     {!isOnly && (
                       <>
                         {/* Left line (if not first) */}
                         {!isFirst && <div className="absolute top-0 left-0 w-1/2 h-full border-t-2 border-zinc-700/50 rounded-tl-xl" />}
                         {/* Right line (if not last) */}
                         {!isLast && <div className="absolute top-0 right-0 w-1/2 h-full border-t-2 border-zinc-700/50 rounded-tr-xl" />}
                       </>
                     )}
                     {/* Vertical connection to node */}
                     <div className="absolute top-0 left-1/2 -ml-px w-0.5 h-4 bg-zinc-700/50">
                        {/* Data Flow Animation down to child */}
                        <motion.div 
                           className="absolute top-0 left-0 w-full h-[50%] bg-blue-400"
                           animate={{ top: ['-100%', '200%'], opacity: [0, 1, 0] }}
                           transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
                        />
                     </div>
                   </div>
                   
                   <TreeNode 
                     node={child} 
                     allAgents={allAgents} 
                     onSelect={onSelect}
                     depth={depth + 1}
                   />
                 </div>
               );
             })}
          </div>
        </>
      )}
    </div>
  );
};