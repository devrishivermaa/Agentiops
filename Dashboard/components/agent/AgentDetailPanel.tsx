import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AgentNode, AgentStatus, AgentType, AgentEventLog } from "../../types";
import { useStore } from "../../store";
import {
  BrainCircuit,
  BookOpen,
  FileText,
  X,
  Terminal,
  Activity,
  Download,
  FileJson,
  Eye,
  Copy,
  Check,
  Clock,
  Zap,
  BarChart3,
  Layers,
  Globe,
  GitMerge,
  FileOutput,
  Brain,
} from "lucide-react";

interface AgentDetailProps {
  agent: AgentNode;
  onClose: () => void;
}

export const AgentDetailPanel: React.FC<AgentDetailProps> = ({
  agent,
  onClose,
}) => {
  const [viewingJson, setViewingJson] = useState<any | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "logs" | "output">(
    "overview"
  );
  const { downloadReport, downloadJson, pipeline, outputPaths } = useStore();

  // Use real event logs from agent or fallback to generated logs
  const logs: Array<{ time: Date; msg: string; type: string }> =
    agent.events?.length > 0
      ? agent.events.map((e: AgentEventLog) => ({
          time:
            e.timestamp instanceof Date ? e.timestamp : new Date(e.timestamp),
          msg: e.message,
          type:
            e.eventType.includes("completed") || e.eventType.includes("success")
              ? "success"
              : e.eventType.includes("processing") ||
                e.eventType.includes("started")
              ? "processing"
              : e.eventType.includes("failed") || e.eventType.includes("error")
              ? "error"
              : "info",
        }))
      : ([
          {
            time: new Date(agent.startTime || Date.now() - 12000),
            msg: "Agent spawned successfully",
            type: "info",
          },
          {
            time: new Date(
              agent.startTime ? agent.startTime + 2000 : Date.now() - 10000
            ),
            msg: "Agent initialized",
            type: "info",
          },
          {
            time: new Date(
              agent.startTime ? agent.startTime + 4000 : Date.now() - 8000
            ),
            msg: `Assigned role: ${agent.metadata?.role || "Worker"}`,
            type: "success",
          },
          agent.status === AgentStatus.PROCESSING
            ? {
                time: new Date(),
                msg: "Processing data chunks...",
                type: "processing",
              }
            : null,
          agent.status === AgentStatus.COMPLETED
            ? {
                time: new Date(agent.endTime || Date.now()),
                msg: "Task completed successfully.",
                type: "success",
              }
            : null,
        ].filter(Boolean) as Array<{ time: Date; msg: string; type: string }>);

  // Dynamic Output Generation
  const getOutputs = () => {
    if (agent.status !== AgentStatus.COMPLETED) return [];

    const outputs = [];

    if (agent.type === AgentType.MASTER) {
      outputs.push({
        type: "summary",
        content:
          "Pipeline execution completed successfully. All sub-tasks aggregated.",
      });
      if (pipeline.status === "completed") {
        outputs.push({
          type: "download",
          label: "Download Analysis Report (PDF)",
          size: outputPaths.reportPath ? "Available" : "Generating...",
          icon: <FileText size={18} />,
          action: downloadReport,
        });
        outputs.push({
          type: "download",
          label: "Download Results (JSON)",
          size: outputPaths.jsonPath ? "Available" : "Generating...",
          icon: <FileJson size={18} />,
          action: downloadJson,
        });
      }
      outputs.push({
        type: "json",
        label: "analysis_data.json",
        size: "Preview",
        icon: <FileJson size={18} />,
        jsonContent: {
          pipeline_id: agent.id,
          status: "success",
          timestamp: new Date().toISOString(),
          metrics: {
            total_processing_time_ms:
              agent.endTime && agent.startTime
                ? agent.endTime - agent.startTime
                : 0,
            submaster_count: agent.metadata?.submasterCount || "N/A",
          },
        },
      });
    } else if (agent.type === AgentType.RESIDUAL) {
      outputs.push({
        type: "summary",
        content:
          "Global context coordination completed. Context broadcast to all submasters.",
      });
      if (agent.metadata?.context) {
        outputs.push({
          type: "json",
          label: "global_context.json",
          size: "Context Data",
          icon: <FileJson size={18} />,
          jsonContent: {
            agent_id: agent.id,
            role: agent.metadata?.role,
            context_preview: agent.metadata.context,
            broadcast_time: agent.endTime
              ? new Date(agent.endTime).toISOString()
              : null,
          },
        });
      }
    } else if (agent.type === AgentType.SUBMASTER) {
      const pageRange = Array.isArray(agent.metadata?.pages)
        ? `${agent.metadata.pages[0]}-${agent.metadata.pages[1]}`
        : "assigned";

      outputs.push({
        type: "summary",
        content:
          agent.metadata?.summary ||
          `Section analysis complete for pages ${pageRange}.`,
      });
      outputs.push({
        type: "json",
        label: "section_metrics.json",
        size: "Section Data",
        icon: <FileJson size={18} />,
        jsonContent: {
          agent_id: agent.id,
          role: agent.metadata?.role,
          pages_covered: agent.metadata?.pages,
          processing_time_ms:
            agent.endTime && agent.startTime
              ? agent.endTime - agent.startTime
              : null,
        },
      });
    } else {
      // Worker or Reducer Pipeline Agents
      if (agent.type === AgentType.REDUCER_SUBMASTER) {
        outputs.push({
          type: "summary",
          content:
            "Reducer SubMasters completed processing mapper results. Enhanced summaries generated.",
        });
        outputs.push({
          type: "json",
          label: "reducer_submaster_output.json",
          size: "Aggregated Data",
          icon: <FileJson size={18} />,
          jsonContent: {
            agent_id: agent.id,
            role: agent.metadata?.role,
            num_results: agent.metadata?.numResults || "N/A",
            elapsed_time: agent.metadata?.elapsedTime
              ? `${agent.metadata.elapsedTime.toFixed(2)}s`
              : "N/A",
            status: "completed",
          },
        });
      } else if (agent.type === AgentType.REDUCER_RESIDUAL) {
        outputs.push({
          type: "summary",
          content:
            "Global context built from reducer results. Processing plan created for Master Merger.",
        });
        outputs.push({
          type: "json",
          label: "global_context.json",
          size: "Context Data",
          icon: <FileJson size={18} />,
          jsonContent: {
            agent_id: agent.id,
            role: agent.metadata?.role,
            context_size: agent.metadata?.contextSize || "N/A",
            plan_size: agent.metadata?.planSize || "N/A",
            elapsed_time: agent.metadata?.elapsedTime
              ? `${agent.metadata.elapsedTime.toFixed(2)}s`
              : "N/A",
          },
        });
      } else if (agent.type === AgentType.MASTER_MERGER) {
        outputs.push({
          type: "summary",
          content:
            "Master Merger synthesized final comprehensive document with executive summary, insights, and conclusions.",
        });
        outputs.push({
          type: "json",
          label: "final_synthesis.json",
          size: "Synthesis Data",
          icon: <FileJson size={18} />,
          jsonContent: {
            agent_id: agent.id,
            role: agent.metadata?.role,
            result_size: agent.metadata?.resultSize || "N/A",
            elapsed_time: agent.metadata?.elapsedTime
              ? `${agent.metadata.elapsedTime.toFixed(2)}s`
              : "N/A",
            sections: [
              "Executive Summary",
              "Detailed Synthesis",
              "Metadata",
              "Insights & Conclusions",
            ],
          },
        });
        if (pipeline.status === "completed") {
          outputs.push({
            type: "download",
            label: "Download Final Summary (PDF)",
            size: outputPaths.reportPath ? "Available" : "Generating...",
            icon: <FileText size={18} />,
            action: downloadReport,
          });
        }
      } else if (agent.type === AgentType.PDF_GENERATOR) {
        outputs.push({
          type: "summary",
          content: agent.metadata?.pdfPath
            ? `PDF report generated successfully at: ${agent.metadata.pdfPath}`
            : "PDF report generation completed.",
        });
        if (agent.metadata?.pdfPath || pipeline.status === "completed") {
          outputs.push({
            type: "download",
            label: "Download PDF Report",
            size: "Available",
            icon: <FileText size={18} />,
            action: downloadReport,
          });
        }
      } else {
        // Regular Worker
        outputs.push({
          type: "summary",
          content:
            agent.metadata?.summary ||
            `Data extraction finished for Page ${
              agent.metadata?.pages || "?"
            }.`,
        });
        if (
          agent.metadata?.entities?.length ||
          agent.metadata?.keywords?.length
        ) {
          outputs.push({
            type: "json",
            label: "page_analysis.json",
            size: "Page Data",
            icon: <FileJson size={18} />,
            jsonContent: {
              agent_id: agent.id,
              page: agent.metadata?.pages,
              entities: agent.metadata?.entities || [],
              keywords: agent.metadata?.keywords || [],
            },
          });
        }
      }
    }

    return outputs;
  };

  const outputs = getOutputs();

  const handleCopyJson = () => {
    if (viewingJson) {
      navigator.clipboard.writeText(JSON.stringify(viewingJson, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getStatusColor = () => {
    switch (agent.status) {
      case AgentStatus.PROCESSING:
        return "text-blue-400 bg-blue-500/10 border-blue-500/30";
      case AgentStatus.COMPLETED:
        return "text-green-400 bg-green-500/10 border-green-500/30";
      case AgentStatus.FAILED:
        return "text-red-400 bg-red-500/10 border-red-500/30";
      default:
        return "text-zinc-400 bg-zinc-500/10 border-zinc-500/30";
    }
  };

  const getAgentTypeColor = () => {
    switch (agent.type) {
      case AgentType.MASTER:
        return "from-purple-500/20 to-purple-600/5 text-purple-400 border-purple-500/30";
      case AgentType.SUBMASTER:
        return "from-amber-500/20 to-amber-600/5 text-amber-400 border-amber-500/30";
      case AgentType.RESIDUAL:
        return "from-cyan-500/20 to-cyan-600/5 text-cyan-400 border-cyan-500/30";
      case AgentType.REDUCER:
        return "from-emerald-500/20 to-emerald-600/5 text-emerald-400 border-emerald-500/30";
      case AgentType.REDUCER_SUBMASTER:
        return "from-purple-500/20 to-purple-600/5 text-purple-400 border-purple-500/30";
      case AgentType.REDUCER_RESIDUAL:
        return "from-pink-500/20 to-pink-600/5 text-pink-400 border-pink-500/30";
      case AgentType.MASTER_MERGER:
        return "from-orange-500/20 to-orange-600/5 text-orange-400 border-orange-500/30";
      case AgentType.PDF_GENERATOR:
        return "from-rose-500/20 to-rose-600/5 text-rose-400 border-rose-500/30";
      default:
        return "from-blue-500/20 to-blue-600/5 text-blue-400 border-blue-500/30";
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-6 bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, y: 20, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.95, y: 20, opacity: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="bg-zinc-900 border border-zinc-700/50 w-full max-w-3xl max-h-[90vh] rounded-2xl shadow-2xl overflow-hidden relative ring-1 ring-white/5 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className={`relative p-6 border-b border-zinc-800 bg-gradient-to-br ${getAgentTypeColor()}`}
        >
          {/* Background pattern */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.02)_1px,transparent_1px)] bg-[size:32px_32px] opacity-50" />

          <div className="relative flex justify-between items-start">
            <div className="flex items-center gap-4">
              <div
                className={`p-3.5 rounded-xl bg-gradient-to-br ${getAgentTypeColor()} border shadow-lg`}
              >
                {agent.type === AgentType.MASTER ? (
                  <BrainCircuit size={28} />
                ) : agent.type === AgentType.SUBMASTER ? (
                  <BookOpen size={28} />
                ) : agent.type === AgentType.RESIDUAL ? (
                  <Globe size={28} />
                ) : agent.type === AgentType.REDUCER ? (
                  <Layers size={28} />
                ) : agent.type === AgentType.REDUCER_SUBMASTER ? (
                  <Layers size={28} />
                ) : agent.type === AgentType.REDUCER_RESIDUAL ? (
                  <Brain size={28} />
                ) : agent.type === AgentType.MASTER_MERGER ? (
                  <GitMerge size={28} />
                ) : agent.type === AgentType.PDF_GENERATOR ? (
                  <FileOutput size={28} />
                ) : (
                  <FileText size={28} />
                )}
              </div>
              <div>
                <h2 className="text-xl font-bold text-zinc-100">
                  {agent.label}
                </h2>
                <div className="flex items-center gap-3 mt-2">
                  <span className="uppercase tracking-wider text-[10px] border border-zinc-700 px-1.5 py-0.5 rounded">
                    {agent.id}
                  </span>
                  <span>•</span>
                  <span
                    className={`capitalize ${
                      agent.status === "processing"
                        ? "text-blue-400"
                        : agent.status === "completed"
                        ? "text-green-400"
                        : agent.status === "failed"
                        ? "text-red-400"
                        : "text-zinc-500"
                    }`}
                  >
                    {agent.status}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 grid gap-6 grid-cols-1 md:grid-cols-2">
          {/* Stats Column */}
          <div className="space-y-6">
            <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
              <h3 className="text-sm font-semibold text-zinc-400 mb-3 flex items-center gap-2">
                <Activity size={14} /> Agent Metrics
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Progress</span>
                  <span className="text-zinc-200">
                    {agent.metadata?.progress || 0}%
                  </span>
                </div>
                <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${agent.metadata?.progress || 0}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm pt-2 border-t border-zinc-800">
                  <span className="text-zinc-500">Page Range</span>
                  <span className="text-zinc-200 font-mono">
                    {Array.isArray(agent.metadata?.pages)
                      ? `${agent.metadata?.pages[0]} - ${agent.metadata?.pages[1]}`
                      : agent.metadata?.pages || "N/A"}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
              <h3 className="text-sm font-semibold text-zinc-400 mb-3 flex items-center gap-2">
                <Terminal size={14} /> System Output
              </h3>
              {outputs.length > 0 ? (
                <div className="space-y-3">
                  {outputs.map((out, i) => (
                    <div key={i}>
                      {out.type === "json" ? (
                        <div className="flex gap-2 w-full">
                          <button className="flex-1 flex items-center gap-3 p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-colors group text-left">
                            <div className="p-2 rounded bg-zinc-900 text-zinc-400 group-hover:text-primary transition-colors">
                              {out.icon}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-zinc-200 truncate">
                                {out.label}
                              </div>
                              <div className="text-xs text-zinc-500">
                                {out.size}
                              </div>
                            </div>
                            <Download
                              size={16}
                              className="text-zinc-500 group-hover:text-primary"
                            />
                          </button>
                          <button
                            onClick={() => setViewingJson(out.jsonContent)}
                            className="p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-400 hover:text-white transition-colors flex items-center justify-center tooltip-trigger"
                            title="View JSON"
                          >
                            <Eye size={18} />
                          </button>
                        </div>
                      ) : out.type === "download" ? (
                        <button
                          onClick={() => out.action && out.action()}
                          className="flex items-center gap-3 w-full p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-colors group text-left"
                        >
                          <div className="p-2 rounded bg-zinc-900 text-zinc-400 group-hover:text-primary transition-colors">
                            {out.icon}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-zinc-200 truncate">
                              {out.label}
                            </div>
                            <div className="text-xs text-zinc-500">
                              {out.size}
                            </div>
                          </div>
                          <Download
                            size={16}
                            className="text-zinc-500 group-hover:text-primary"
                          />
                        </button>
                      ) : (
                        <div className="text-sm text-green-400 font-mono bg-black/40 p-3 rounded border border-zinc-800/50 break-words">
                          {out.content}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-zinc-600 italic">
                  No final output generated yet...
                </div>
              )}
            </div>
          </div>

          {/* Logs Column */}
          <div className="bg-black/20 rounded-lg border border-zinc-800 flex flex-col min-h-[200px] max-h-[400px]">
            <div className="p-3 border-b border-zinc-800 bg-zinc-900/30 flex items-center justify-between flex-shrink-0">
              <span className="text-xs font-mono text-zinc-400">
                Activity Log
              </span>
              {agent.status === AgentStatus.PROCESSING && (
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-2 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
              {logs.map((log, i) => (
                <div key={i} className="text-zinc-300 break-words">
                  <span className="text-zinc-600 mr-2">
                    [{log.time.toLocaleTimeString()}]
                  </span>
                  <span
                    className={
                      log.type === "success"
                        ? "text-green-400"
                        : log.type === "processing"
                        ? "text-blue-400"
                        : log.type === "error"
                        ? "text-red-400"
                        : "text-zinc-300"
                    }
                  >
                    {log.msg}
                  </span>
                </div>
              ))}
              {agent.status === AgentStatus.PROCESSING && (
                <div className="w-2 h-4 bg-zinc-600 animate-pulse inline-block align-middle ml-2" />
              )}
            </div>
          </div>
        </div>

        {/* JSON Viewer Modal Overlay */}
        <AnimatePresence>
          {viewingJson && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-50 bg-zinc-900/95 backdrop-blur-sm flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-zinc-700 bg-zinc-900">
                <div className="flex items-center gap-2">
                  <FileJson className="text-primary" size={18} />
                  <span className="font-semibold text-zinc-200 text-sm">
                    Raw JSON Output
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopyJson}
                    className="p-2 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors flex items-center gap-2 text-xs"
                  >
                    {copied ? (
                      <Check size={14} className="text-green-500" />
                    ) : (
                      <Copy size={14} />
                    )}
                    {copied ? "Copied" : "Copy"}
                  </button>
                  <button
                    onClick={() => setViewingJson(null)}
                    className="p-2 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors"
                  >
                    <X size={18} />
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-auto p-4 bg-[#0d0d0d]">
                <pre className="text-xs font-mono text-green-400 whitespace-pre-wrap break-all leading-relaxed">
                  {JSON.stringify(viewingJson, null, 2)}
                </pre>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};
