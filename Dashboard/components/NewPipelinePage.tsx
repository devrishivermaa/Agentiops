import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  Loader2,
  Sparkles,
  BrainCircuit,
  Users,
  Layers,
  Zap,
  ArrowRight,
  CheckCircle2,
  Clock,
  FileSearch,
  GitBranch,
  Bot,
} from "lucide-react";

interface NewPipelinePageProps {
  onUpload: (file: File) => void;
  onSimulate: () => void;
  isProcessing: boolean;
  isConnected: boolean;
}

export const NewPipelinePage: React.FC<NewPipelinePageProps> = ({
  onUpload,
  onSimulate,
  isProcessing,
  isConnected,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleStartPipeline = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const pipelineSteps = [
    {
      icon: FileSearch,
      title: "Document Extraction",
      description: "Intelligent PDF parsing and content extraction",
      color: "blue",
    },
    {
      icon: BrainCircuit,
      title: "Master Planning",
      description: "AI orchestrator analyzes and creates execution plan",
      color: "purple",
    },
    {
      icon: GitBranch,
      title: "Task Distribution",
      description: "Sub-masters delegate work to specialized workers",
      color: "amber",
    },
    {
      icon: Bot,
      title: "Parallel Processing",
      description: "Worker agents process pages concurrently",
      color: "green",
    },
  ];

  return (
    <div className="relative w-full h-full overflow-auto bg-zinc-950">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950 via-zinc-900/50 to-zinc-950 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/5 via-transparent to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.015)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none" />

      {/* Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-zinc-100 mb-4 tracking-tight">
            Initialize New Pipeline
          </h1>
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto">
            Upload your PDF document and watch our AI agents orchestrate a
            comprehensive analysis in real-time.
          </p>
        </motion.div>

        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-12"
        >
          <div
            className={`
              relative w-full border-2 border-dashed rounded-2xl transition-all duration-300 overflow-hidden
              ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : selectedFile
                  ? "border-green-500/50 bg-green-500/5"
                  : "border-zinc-700/50 hover:border-zinc-600 bg-zinc-900/30"
              }
              ${isProcessing ? "opacity-50 pointer-events-none" : ""}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {/* Animated border glow */}
            {isDragging && (
              <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/10 to-primary/0 animate-pulse" />
            )}

            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".pdf"
              onChange={handleChange}
              disabled={isProcessing}
            />

            <label
              htmlFor="file-upload"
              className="flex flex-col items-center cursor-pointer w-full p-12"
            >
              <AnimatePresence mode="wait">
                {isProcessing ? (
                  <motion.div
                    key="processing"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex flex-col items-center"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-primary/20 blur-2xl rounded-full" />
                      <Loader2 className="w-16 h-16 text-primary animate-spin relative" />
                    </div>
                    <h3 className="text-xl font-semibold text-zinc-100 mt-6 mb-2">
                      Initializing Pipeline...
                    </h3>
                    <p className="text-zinc-500 text-sm">
                      Preparing document for analysis
                    </p>
                  </motion.div>
                ) : selectedFile ? (
                  <motion.div
                    key="selected"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex flex-col items-center"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-green-500/20 blur-2xl rounded-full" />
                      <div className="relative bg-zinc-800 p-5 rounded-2xl border border-green-500/30">
                        <FileText className="w-10 h-10 text-green-400" />
                        <CheckCircle2 className="absolute -bottom-1 -right-1 w-5 h-5 text-green-500 bg-zinc-900 rounded-full" />
                      </div>
                    </div>
                    <h3 className="text-xl font-semibold text-zinc-100 mt-6 mb-1">
                      {selectedFile.name}
                    </h3>
                    <p className="text-zinc-500 text-sm mb-4">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready
                      to process
                    </p>
                    <p className="text-xs text-zinc-600">
                      Click to select a different file
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex flex-col items-center"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-zinc-700/30 blur-2xl rounded-full" />
                      <div className="relative bg-zinc-800 p-5 rounded-2xl border border-zinc-700/50 group-hover:border-zinc-600 transition-colors">
                        <Upload className="w-10 h-10 text-zinc-400" />
                      </div>
                    </div>
                    <h3 className="text-xl font-semibold text-zinc-100 mt-6 mb-2">
                      Drop your PDF here
                    </h3>
                    <p className="text-zinc-500 text-sm text-center max-w-sm">
                      Drag and drop your document, or click to browse.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </label>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-6">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleStartPipeline}
              disabled={!selectedFile || isProcessing}
              className={`
                flex items-center gap-3 px-8 py-3 rounded-xl font-semibold text-base transition-all
                ${
                  selectedFile && !isProcessing
                    ? "bg-primary text-white shadow-lg shadow-primary/25 hover:shadow-primary/40"
                    : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                }
              `}
            >
              <Zap size={18} />
              Start Pipeline
              <ArrowRight size={18} />
            </motion.button>

            {!isConnected && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onSimulate}
                disabled={isProcessing}
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-sm bg-zinc-800/80 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 transition-all"
              >
                <Sparkles size={16} />
                Run Demo Simulation
              </motion.button>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};
