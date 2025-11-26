import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AgentNode, AgentStatus, AgentType } from '../../types';
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
  Check
} from 'lucide-react';

interface AgentDetailProps {
  agent: AgentNode;
  onClose: () => void;
}

export const AgentDetailPanel: React.FC<AgentDetailProps> = ({ agent, onClose }) => {
  const [viewingJson, setViewingJson] = useState<any | null>(null);
  const [copied, setCopied] = useState(false);

  // Mock logs
  const logs = [
    `[${new Date(Date.now() - 10000).toLocaleTimeString()}] Agent initialized`,
    `[${new Date(Date.now() - 8000).toLocaleTimeString()}] Assigned role: ${agent.metadata?.role || 'Worker'}`,
    agent.status === AgentStatus.PROCESSING ? `[${new Date().toLocaleTimeString()}] Processing data chunks...` : '',
    agent.status === AgentStatus.COMPLETED ? `[${new Date().toLocaleTimeString()}] Task completed successfully.` : '',
  ].filter(Boolean);

  // Dynamic Output Generation
  const getOutputs = () => {
    if (agent.status !== AgentStatus.COMPLETED) return [];
    
    const outputs = [];
    
    if (agent.type === AgentType.MASTER) {
      outputs.push({
        type: 'summary',
        content: 'Pipeline execution completed successfully. All sub-tasks aggregated.'
      });
      outputs.push({
        type: 'download',
        label: 'Final_Report_2024.pdf',
        size: '2.4 MB',
        icon: <FileText size={18} />
      });
      outputs.push({
        type: 'json',
        label: 'analysis_data.json',
        size: '156 KB',
        icon: <FileJson size={18} />,
        jsonContent: {
          pipeline_id: agent.id,
          status: "success",
          timestamp: new Date().toISOString(),
          metrics: {
            total_processing_time_ms: 4520,
            pages_analyzed: 45,
            agents_spawned: 12
          },
          risk_assessment: {
            score: 8.5,
            high_priority_flags: ["compliance_warning_pg4", "missing_date_pg12"],
            entities_detected: 142
          }
        }
      });
    } else if (agent.type === AgentType.SUBMASTER) {
      const pageRange = Array.isArray(agent.metadata?.pages) 
        ? `${agent.metadata.pages[0]}-${agent.metadata.pages[1]}` 
        : 'assigned';
      
      outputs.push({
        type: 'summary',
        content: `Section analysis complete for pages ${pageRange}.`
      });
      outputs.push({
        type: 'json',
        label: 'section_metrics.json',
        size: '42 KB',
        icon: <FileJson size={18} />,
        jsonContent: {
           agent_id: agent.id,
           role: agent.metadata?.role,
           pages_covered: agent.metadata?.pages,
           findings_count: 15,
           confidence_score: 0.94,
           extracted_terms: ["Revenue", "Liability", "FY2024"]
        }
      });
    } else {
      // Worker
      outputs.push({
        type: 'summary',
        content: `Data extraction finished for Page ${agent.metadata?.pages || '?'}.`
      });
      outputs.push({
        type: 'data',
        content: 'Confidence Score: 98.5% | Entities Extracted: 42'
      });
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

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div 
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        className="bg-surface border border-zinc-700 w-full max-w-2xl rounded-xl shadow-2xl overflow-hidden relative"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-zinc-800 bg-zinc-900/50">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-lg ${
              agent.type === AgentType.MASTER ? 'bg-purple-500/20 text-purple-400' :
              agent.type === AgentType.SUBMASTER ? 'bg-amber-500/20 text-amber-400' :
              'bg-blue-500/20 text-blue-400'
            }`}>
              {agent.type === AgentType.MASTER ? <BrainCircuit size={24} /> :
               agent.type === AgentType.SUBMASTER ? <BookOpen size={24} /> :
               <FileText size={24} />}
            </div>
            <div>
              <h2 className="text-xl font-bold text-zinc-100">{agent.label}</h2>
              <div className="flex items-center gap-2 text-sm text-zinc-400 mt-1">
                <span className="uppercase tracking-wider text-[10px] border border-zinc-700 px-1.5 py-0.5 rounded">
                  {agent.id}
                </span>
                <span>•</span>
                <span className={`capitalize ${
                    agent.status === 'processing' ? 'text-blue-400' :
                    agent.status === 'completed' ? 'text-green-400' :
                    agent.status === 'failed' ? 'text-red-400' : 'text-zinc-500'
                }`}>
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

        {/* Content */}
        <div className="p-6 grid gap-6 grid-cols-1 md:grid-cols-2">
          {/* Stats Column */}
          <div className="space-y-6">
            <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
              <h3 className="text-sm font-semibold text-zinc-400 mb-3 flex items-center gap-2">
                <Activity size={14} /> Agent Metrics
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Progress</span>
                  <span className="text-zinc-200">{agent.metadata?.progress || 0}%</span>
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
                      : agent.metadata?.pages || 'N/A'}
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
                      {out.type === 'json' ? (
                        <div className="flex gap-2 w-full">
                          <button className="flex-1 flex items-center gap-3 p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-colors group text-left">
                              <div className="p-2 rounded bg-zinc-900 text-zinc-400 group-hover:text-primary transition-colors">
                                {out.icon}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-medium text-zinc-200 truncate">{out.label}</div>
                                <div className="text-xs text-zinc-500">{out.size}</div>
                              </div>
                              <Download size={16} className="text-zinc-500 group-hover:text-primary" />
                          </button>
                          <button 
                            onClick={() => setViewingJson(out.jsonContent)}
                            className="p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-400 hover:text-white transition-colors flex items-center justify-center tooltip-trigger"
                            title="View JSON"
                          >
                            <Eye size={18} />
                          </button>
                        </div>
                      ) : out.type === 'download' ? (
                         <button className="flex items-center gap-3 w-full p-3 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-colors group text-left">
                            <div className="p-2 rounded bg-zinc-900 text-zinc-400 group-hover:text-primary transition-colors">
                              {out.icon}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-zinc-200 truncate">{out.label}</div>
                              <div className="text-xs text-zinc-500">{out.size}</div>
                            </div>
                            <Download size={16} className="text-zinc-500 group-hover:text-primary" />
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
                <div className="text-sm text-zinc-600 italic">No final output generated yet...</div>
              )}
            </div>
          </div>

          {/* Logs Column */}
          <div className="bg-black/20 rounded-lg border border-zinc-800 flex flex-col h-64 md:h-auto">
            <div className="p-3 border-b border-zinc-800 bg-zinc-900/30 flex items-center justify-between">
              <span className="text-xs font-mono text-zinc-400">Activity Log</span>
              {agent.status === AgentStatus.PROCESSING && (
                 <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-2">
              {logs.map((log, i) => (
                <div key={i} className="text-zinc-300 break-words">
                  <span className="text-zinc-600 mr-2">{'>'}</span>
                  {log}
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
              className="absolute inset-0 z-50 bg-surface/95 backdrop-blur-sm flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-zinc-700 bg-zinc-900">
                <div className="flex items-center gap-2">
                  <FileJson className="text-primary" size={18} />
                  <span className="font-semibold text-zinc-200 text-sm">Raw JSON Output</span>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={handleCopyJson}
                    className="p-2 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors flex items-center gap-2 text-xs"
                  >
                    {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                    {copied ? 'Copied' : 'Copy'}
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