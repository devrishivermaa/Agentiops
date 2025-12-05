import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Sparkles,
  MessageSquare,
  FileText,
  ArrowRight,
  Loader2,
  Lightbulb,
} from "lucide-react";

interface IntentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (intent: string, context?: string) => void;
  fileName: string;
  isLoading?: boolean;
}

const intentSuggestions = [
  "Summarize for a presentation",
  "Extract key findings and methodology",
  "Create a detailed analysis report",
  "Identify main arguments and conclusions",
  "Extract data and statistics",
];

export const IntentModal: React.FC<IntentModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  fileName,
  isLoading = false,
}) => {
  const [intent, setIntent] = useState("");
  const [context, setContext] = useState("");

  const handleSubmit = () => {
    if (intent.trim()) {
      onSubmit(intent.trim(), context.trim() || undefined);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setIntent(suggestion);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={(e) => e.target === e.currentTarget && !isLoading && onClose()}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: "spring", duration: 0.5 }}
          className="bg-zinc-900 border border-zinc-700/50 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Sparkles className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-zinc-100">
                  What would you like to do?
                </h2>
                <p className="text-sm text-zinc-500">{fileName}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              disabled={isLoading}
              className="p-2 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5 text-zinc-400" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Intent Input */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                <MessageSquare size={16} />
                High-Level Intent
                <span className="text-red-400">*</span>
              </label>
              <textarea
                value={intent}
                onChange={(e) => setIntent(e.target.value)}
                placeholder="Describe what you want to achieve with this document..."
                disabled={isLoading}
                className="w-full h-24 px-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 resize-none disabled:opacity-50"
              />
            </div>

            {/* Quick Suggestions */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-400">
                <Lightbulb size={16} />
                Quick suggestions
              </label>
              <div className="flex flex-wrap gap-2">
                {intentSuggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    disabled={isLoading}
                    className="px-3 py-1.5 text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg border border-zinc-700 hover:border-zinc-600 transition-all disabled:opacity-50"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>

            {/* Context Input */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                <FileText size={16} />
                Document Context
                <span className="text-zinc-500 font-normal">(optional)</span>
              </label>
              <textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Any additional context about the document that might help with processing..."
                disabled={isLoading}
                className="w-full h-20 px-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 resize-none disabled:opacity-50"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-zinc-800 bg-zinc-900/50">
            <button
              onClick={onClose}
              disabled={isLoading}
              className="px-4 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSubmit}
              disabled={!intent.trim() || isLoading}
              className={`
                flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all
                ${
                  intent.trim() && !isLoading
                    ? "bg-primary text-white shadow-lg shadow-primary/25 hover:shadow-primary/40"
                    : "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                }
              `}
            >
              {isLoading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Continue
                  <ArrowRight size={16} />
                </>
              )}
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
