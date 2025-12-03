import React from "react";
import { motion } from "framer-motion";
import { verticalLineVariants } from "./animations";

export const BranchConnector: React.FC<{
  type: "vertical" | "horizontal-left" | "horizontal-right" | "horizontal-full";
  className?: string;
  isActive?: boolean;
}> = ({ type, className = "", isActive = false }) => {
  if (type === "vertical") {
    return (
      <motion.div
        variants={verticalLineVariants}
        initial="hidden"
        animate="visible"
        className={`absolute w-0.5 h-full left-1/2 -ml-px origin-top rounded-full ${
          isActive
            ? "bg-gradient-to-b from-blue-500/60 to-blue-500/30"
            : "bg-zinc-700/40"
        } ${className}`}
      >
        {/* Data flow animation */}
        <motion.div
          className={`absolute left-0 w-full h-4 rounded-full ${
            isActive ? "bg-blue-400" : "bg-primary/40"
          }`}
          initial={{ top: "-20%", opacity: 0 }}
          animate={{ top: ["-20%", "100%"], opacity: [0, 0.7, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Glow effect when active */}
        {isActive && (
          <div className="absolute inset-0 w-1 -ml-px bg-blue-500/20 blur-sm" />
        )}
      </motion.div>
    );
  }

  // Horizontal connectors
  return (
    <div
      className={`absolute top-0 h-0.5 w-1/2 rounded-full ${
        isActive
          ? "bg-gradient-to-r from-blue-500/60 to-zinc-700/40"
          : "bg-zinc-700/40"
      } ${
        type.includes("left")
          ? "right-1/2"
          : type.includes("right")
          ? "left-1/2"
          : ""
      } ${className}`}
    />
  );
};
