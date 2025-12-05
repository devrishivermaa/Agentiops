import React from "react";
import { motion } from "framer-motion";
import { AgentNode, AgentStatus } from "../../types";
import { AgentCard } from "./AgentCard";
import { lineGrowVariants, horizontalLineVariants } from "./animations";

interface TreeNodeProps {
  node: AgentNode;
  allAgents: AgentNode[];
  onSelect: (agent: AgentNode) => void;
  depth?: number;
  index?: number;
}

export const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  allAgents,
  onSelect,
  depth = 0,
  index = 0,
}) => {
  // Find children
  const children = allAgents.filter((a) => a.parentId === node.id);
  const hasChildren = children.length > 0;

  // Check if any children are active
  const hasActiveChildren = children.some(
    (c) => c.status === AgentStatus.PROCESSING
  );

  // Calculate stagger delay based on depth and index
  const baseDelay = depth * 0.15 + index * 0.1;

  return (
    <motion.div
      className="flex flex-col items-center relative"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.5,
        delay: baseDelay,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <AgentCard agent={node} onClick={onSelect} />

      {hasChildren && (
        <>
          {/* Vertical line down from parent */}
          <motion.div
            initial={{ scaleY: 0, opacity: 0 }}
            animate={{ scaleY: 1, opacity: 1 }}
            transition={{
              duration: 0.4,
              delay: baseDelay + 0.2,
              ease: [0.22, 1, 0.36, 1],
            }}
            className={`w-0.5 h-8 origin-top rounded-full ${
              hasActiveChildren
                ? "bg-gradient-to-b from-blue-500/60 to-blue-500/30"
                : "bg-zinc-700/40"
            }`}
          />

          {/* Children row */}
          <motion.div
            className="flex relative"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: baseDelay + 0.25, duration: 0.3 }}
          >
            {/* Children container */}
            {children.map((child, childIndex) => {
              const isChildActive = child.status === AgentStatus.PROCESSING;
              const isFirst = childIndex === 0;
              const isLast = childIndex === children.length - 1;
              const isOnly = children.length === 1;
              const childDelay = baseDelay + 0.3 + childIndex * 0.12;

              return (
                <motion.div
                  key={child.id}
                  className="flex flex-col items-center relative px-4 md:px-6"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: childDelay, duration: 0.3 }}
                >
                  {/* Horizontal connectors at top */}
                  {!isOnly && (
                    <>
                      {/* Left half of horizontal line */}
                      {!isFirst && (
                        <motion.div
                          initial={{ scaleX: 0, opacity: 0 }}
                          animate={{ scaleX: 1, opacity: 1 }}
                          transition={{
                            duration: 0.35,
                            delay: childDelay,
                            ease: [0.22, 1, 0.36, 1],
                          }}
                          className={`absolute top-0 right-1/2 h-0.5 w-1/2 origin-right ${
                            hasActiveChildren
                              ? "bg-blue-500/50"
                              : "bg-zinc-700/40"
                          }`}
                        />
                      )}
                      {/* Right half of horizontal line */}
                      {!isLast && (
                        <motion.div
                          initial={{ scaleX: 0, opacity: 0 }}
                          animate={{ scaleX: 1, opacity: 1 }}
                          transition={{
                            duration: 0.35,
                            delay: childDelay,
                            ease: [0.22, 1, 0.36, 1],
                          }}
                          className={`absolute top-0 left-1/2 h-0.5 w-1/2 origin-left ${
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
                    initial={{ scaleY: 0, opacity: 0 }}
                    animate={{ scaleY: 1, opacity: 1 }}
                    transition={{
                      duration: 0.4,
                      delay: childDelay + 0.1,
                      ease: [0.22, 1, 0.36, 1],
                    }}
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
                        duration: 1.8,
                        repeat: Infinity,
                        delay: childIndex * 0.4,
                        ease: "easeInOut",
                      }}
                    />
                  </motion.div>

                  <TreeNode
                    node={child}
                    allAgents={allAgents}
                    onSelect={onSelect}
                    depth={depth + 1}
                    index={childIndex}
                  />
                </motion.div>
              );
            })}
          </motion.div>
        </>
      )}
    </motion.div>
  );
};
