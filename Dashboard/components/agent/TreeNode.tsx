import React from "react";
import { motion } from "framer-motion";
import { AgentNode, AgentStatus } from "../../types";
import { AgentCard } from "./AgentCard";
import { BranchConnector } from "./TreeConnector";

interface TreeNodeProps {
  node: AgentNode;
  allAgents: AgentNode[];
  onSelect: (agent: AgentNode) => void;
  depth?: number;
}

export const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  allAgents,
  onSelect,
  depth = 0,
}) => {
  // Find children
  const children = allAgents.filter((a) => a.parentId === node.id);
  const hasChildren = children.length > 0;

  // Check if any children are active
  const hasActiveChildren = children.some(
    (c) => c.status === AgentStatus.PROCESSING
  );

  return (
    <div className="flex flex-col items-center relative">
      <AgentCard agent={node} onClick={onSelect} />

      {hasChildren && (
        <>
          {/* Vertical line down from parent */}
          <motion.div
            initial={{ scaleY: 0 }}
            animate={{ scaleY: 1 }}
            transition={{ duration: 0.3 }}
            className={`w-0.5 h-8 origin-top rounded-full ${
              hasActiveChildren
                ? "bg-gradient-to-b from-blue-500/60 to-blue-500/30"
                : "bg-zinc-700/40"
            }`}
          />

          {/* Children row */}
          <div className="flex relative">
            {/* Children container */}
            {children.map((child, index) => {
              const isChildActive = child.status === AgentStatus.PROCESSING;
              const isFirst = index === 0;
              const isLast = index === children.length - 1;
              const isOnly = children.length === 1;

              return (
                <div
                  key={child.id}
                  className="flex flex-col items-center relative px-4 md:px-6"
                >
                  {/* Horizontal connectors at top */}
                  {!isOnly && (
                    <>
                      {/* Left half of horizontal line */}
                      {!isFirst && (
                        <div
                          className={`absolute top-0 right-1/2 h-0.5 w-1/2 ${
                            hasActiveChildren
                              ? "bg-blue-500/50"
                              : "bg-zinc-700/40"
                          }`}
                        />
                      )}
                      {/* Right half of horizontal line */}
                      {!isLast && (
                        <div
                          className={`absolute top-0 left-1/2 h-0.5 w-1/2 ${
                            hasActiveChildren
                              ? "bg-blue-500/50"
                              : "bg-zinc-700/40"
                          }`}
                        />
                      )}
                    </>
                  )}

                  {/* Vertical drop to child */}
                  <motion.div
                    initial={{ scaleY: 0 }}
                    animate={{ scaleY: 1 }}
                    transition={{ duration: 0.3, delay: 0.1 + index * 0.05 }}
                    className={`w-0.5 h-6 origin-top rounded-full relative ${
                      isChildActive
                        ? "bg-gradient-to-b from-blue-500/60 to-blue-500/30"
                        : "bg-zinc-700/40"
                    }`}
                  >
                    {/* Animated data flow pulse */}
                    <motion.div
                      className={`absolute left-0 w-full h-2 rounded-full ${
                        isChildActive ? "bg-blue-400" : "bg-primary/50"
                      }`}
                      initial={{ top: "-50%", opacity: 0 }}
                      animate={{ top: ["0%", "100%"], opacity: [0, 0.8, 0] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: index * 0.3,
                        ease: "easeInOut",
                      }}
                    />
                  </motion.div>

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
