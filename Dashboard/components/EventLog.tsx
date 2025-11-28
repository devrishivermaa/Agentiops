import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../store";
import {
  Terminal,
  Clock,
  AlertCircle,
  CheckCircle,
  Info,
  ChevronDown,
  Sparkles,
  Filter,
} from "lucide-react";

export const EventLog: React.FC = () => {
  const events = useStore((state) => state.events);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<"all" | "success" | "error" | "info">(
    "all"
  );
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (scrollRef.current && autoScroll) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  const filteredEvents = events.filter(
    (e) => filter === "all" || e.severity === filter
  );

  const getIcon = (severity: string) => {
    switch (severity) {
      case "success":
        return <CheckCircle className="w-3.5 h-3.5 text-green-400" />;
      case "error":
        return <AlertCircle className="w-3.5 h-3.5 text-red-400" />;
      case "warning":
        return <AlertCircle className="w-3.5 h-3.5 text-yellow-400" />;
      default:
        return <Info className="w-3.5 h-3.5 text-blue-400" />;
    }
  };

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case "success":
        return "bg-green-500/5 border-green-500/20 hover:bg-green-500/10";
      case "error":
        return "bg-red-500/5 border-red-500/20 hover:bg-red-500/10";
      case "warning":
        return "bg-yellow-500/5 border-yellow-500/20 hover:bg-yellow-500/10";
      default:
        return "bg-zinc-800/30 border-zinc-700/30 hover:bg-zinc-800/50";
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-900/50 border-t border-zinc-800 md:border-t-0 md:border-l md:border-zinc-800 w-full md:w-96 shrink-0 backdrop-blur-sm">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 bg-zinc-900/80">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-zinc-800 border border-zinc-700">
              <Terminal className="w-4 h-4 text-primary" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-zinc-200">Event Log</h3>
              <p className="text-[10px] text-zinc-500">
                Real-time system events
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {events.length > 0 && (
              <motion.div
                className="flex items-center gap-1 px-2 py-1 bg-primary/10 rounded-full border border-primary/20"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
              >
                <Sparkles size={10} className="text-primary" />
                <span className="text-[10px] font-medium text-primary">
                  {events.length}
                </span>
              </motion.div>
            )}
          </div>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <button
              onClick={() => setIsFilterOpen(!isFilterOpen)}
              className="w-full flex items-center justify-between gap-2 px-3 py-1.5 bg-zinc-800/50 rounded-lg border border-zinc-700/50 text-xs text-zinc-400 hover:bg-zinc-800 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Filter size={12} />
                <span className="capitalize">{filter}</span>
              </div>
              <ChevronDown
                size={12}
                className={`transition-transform ${
                  isFilterOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            <AnimatePresence>
              {isFilterOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 rounded-lg border border-zinc-700 overflow-hidden z-10 shadow-xl"
                >
                  {(["all", "success", "error", "info"] as const).map((f) => (
                    <button
                      key={f}
                      onClick={() => {
                        setFilter(f);
                        setIsFilterOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-xs text-left capitalize hover:bg-zinc-700/50 transition-colors ${
                        filter === f
                          ? "text-primary bg-primary/10"
                          : "text-zinc-400"
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1.5 rounded-lg border text-xs transition-colors ${
              autoScroll
                ? "bg-primary/10 border-primary/30 text-primary"
                : "bg-zinc-800/50 border-zinc-700/50 text-zinc-500"
            }`}
          >
            Auto-scroll
          </button>
        </div>
      </div>

      {/* Events List */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredEvents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="p-4 rounded-full bg-zinc-800/50 border border-zinc-700/50 mb-4">
              <Terminal size={24} className="text-zinc-600" />
            </div>
            <p className="text-sm text-zinc-500">No events yet</p>
            <p className="text-xs text-zinc-600 mt-1">
              Events will appear here as the pipeline runs
            </p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {filteredEvents.map((event, index) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.02 }}
                className={`p-3 rounded-xl border transition-all duration-200 ${getSeverityStyles(
                  event.severity
                )}`}
              >
                <div className="flex gap-3">
                  <div className="mt-0.5 shrink-0 p-1 rounded-md bg-zinc-900/50">
                    {getIcon(event.severity)}
                  </div>
                  <div className="flex-1 min-w-0 space-y-1.5">
                    <div className="flex items-center flex-wrap gap-2">
                      <span className="text-[10px] font-mono text-zinc-500 flex items-center gap-1">
                        <Clock size={10} />
                        {event.timestamp.toLocaleTimeString([], {
                          hour12: false,
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </span>
                      {event.agentId && (
                        <span className="px-1.5 py-0.5 rounded-md bg-zinc-800 text-zinc-400 text-[10px] font-mono border border-zinc-700/50">
                          {event.agentId}
                        </span>
                      )}
                    </div>
                    <p
                      className={`text-xs leading-relaxed break-words ${
                        event.severity === "error"
                          ? "text-red-300"
                          : event.severity === "success"
                          ? "text-green-300"
                          : "text-zinc-300"
                      }`}
                    >
                      {event.message}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};
