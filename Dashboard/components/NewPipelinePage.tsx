import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  Loader2,
  Sparkles,
  BrainCircuit,
  Zap,
  ArrowRight,
  CheckCircle2,
  FileSearch,
  GitBranch,
  Bot,
  MessageSquare,
  Settings,
  AlertCircle,
} from "lucide-react";
import { useStore } from "../store";
import { SessionStatus } from "../types";
import { IntentModal } from "./IntentModal";
import { MetadataEditor } from "./MetadataEditor";

interface NewPipelinePageProps {
  onUpload: (file: File) => void;
  onSimulate: () => void;
  isProcessing: boolean;
  isConnected: boolean;
  onPipelineStarted?: () => void;
}

export const NewPipelinePage: React.FC<NewPipelinePageProps> = ({
  onUpload,
  onSimulate,
  isProcessing,
  isConnected,
  onPipelineStarted,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showIntentModal, setShowIntentModal] = useState(false);
  const [showMetadataEditor, setShowMetadataEditor] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmittingIntent, setIsSubmittingIntent] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { session, uploadFile, submitIntent, approveAndProcess, resetSession } =
    useStore();

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
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith(".pdf")) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Only PDF files are allowed");
      }
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.toLowerCase().endsWith(".pdf")) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Only PDF files are allowed");
      }
    }
  };

  const handleStartPipeline = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      await uploadFile(selectedFile);
      setShowIntentModal(true);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleIntentSubmit = async (intent: string, context?: string) => {
    if (!session.sessionId) return;

    setIsSubmittingIntent(true);

    try {
      await submitIntent(session.sessionId, intent, context);
      setShowIntentModal(false);
      setShowMetadataEditor(true);
    } catch (err: any) {
      setError(err.message || "Failed to submit intent");
    } finally {
      setIsSubmittingIntent(false);
    }
  };

  const handleApprove = async (
    approved: boolean,
    modifiedMetadata?: Record<string, any>
  ) => {
    if (!session.sessionId) return;

    setIsApproving(true);

    try {
      await approveAndProcess(session.sessionId, approved, modifiedMetadata);
      setShowMetadataEditor(false);
      // Trigger navigation to dashboard
      if (onPipelineStarted) {
        onPipelineStarted();
      }
    } catch (err: any) {
      setError(err.message || "Failed to start processing");
    } finally {
      setIsApproving(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setError(null);
    resetSession();
  };

  // Workflow steps indicator
  const workflowSteps = [
    {
      id: 1,
      label: "Upload",
      icon: Upload,
      status: selectedFile ? "completed" : "current",
    },
    {
      id: 2,
      label: "Intent",
      icon: MessageSquare,
      status:
        session.status === SessionStatus.AWAITING_APPROVAL
          ? "completed"
          : session.status === SessionStatus.GENERATING_METADATA
          ? "current"
          : "pending",
    },
    {
      id: 3,
      label: "Review",
      icon: Settings,
      status:
        session.status === SessionStatus.PROCESSING
          ? "completed"
          : session.status === SessionStatus.AWAITING_APPROVAL
          ? "current"
          : "pending",
    },
    {
      id: 4,
      label: "Process",
      icon: Zap,
      status:
        session.status === SessionStatus.PROCESSING ? "current" : "pending",
    },
  ];

  const isAnyLoading =
    isUploading || isSubmittingIntent || isApproving || isProcessing;

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
          className="text-center mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-zinc-100 mb-4 tracking-tight">
            Initialize New Pipeline
          </h1>
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto">
            Upload your PDF document and watch our AI agents orchestrate a
            comprehensive analysis in real-time.
          </p>
        </motion.div>

        {/* Workflow Steps Indicator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="flex items-center justify-center gap-2 mb-8"
        >
          {workflowSteps.map((step, index) => (
            <React.Fragment key={step.id}>
              <div
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                  step.status === "completed"
                    ? "bg-green-500/10 text-green-400 border border-green-500/30"
                    : step.status === "current"
                    ? "bg-primary/10 text-primary border border-primary/30"
                    : "bg-zinc-800/50 text-zinc-500 border border-zinc-700/50"
                }`}
              >
                <step.icon size={14} />
                <span className="hidden sm:inline">{step.label}</span>
              </div>
              {index < workflowSteps.length - 1 && (
                <div
                  className={`w-8 h-0.5 ${
                    step.status === "completed"
                      ? "bg-green-500/50"
                      : "bg-zinc-700"
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </motion.div>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 flex items-center gap-3 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400"
          >
            <AlertCircle size={18} />
            <span className="text-sm">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              ×
            </button>
          </motion.div>
        )}

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
              ${isAnyLoading ? "opacity-50 pointer-events-none" : ""}
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
              disabled={isAnyLoading}
            />

            <label
              htmlFor="file-upload"
              className="flex flex-col items-center cursor-pointer w-full p-12"
            >
              <AnimatePresence mode="wait">
                {isUploading ? (
                  <motion.div
                    key="uploading"
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
                      Uploading...
                    </h3>
                    <p className="text-zinc-500 text-sm">
                      Preparing your document
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
              disabled={!selectedFile || isAnyLoading}
              className={`
                flex items-center gap-3 px-8 py-3 rounded-xl font-semibold text-base transition-all
                ${
                  selectedFile && !isAnyLoading
                    ? "bg-primary text-white shadow-lg shadow-primary/25 hover:shadow-primary/40"
                    : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                }
              `}
            >
              {isUploading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Zap size={18} />
                  Start Pipeline
                  <ArrowRight size={18} />
                </>
              )}
            </motion.button>

            {selectedFile && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleReset}
                disabled={isAnyLoading}
                className="flex items-center gap-2 px-4 py-3 rounded-xl font-medium text-sm text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-all"
              >
                Reset
              </motion.button>
            )}

            {!isConnected && !selectedFile && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onSimulate}
                disabled={isAnyLoading}
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-sm bg-zinc-800/80 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 transition-all"
              >
                <Sparkles size={16} />
                Run Demo Simulation
              </motion.button>
            )}
          </div>
        </motion.div>
      </div>

      {/* Intent Modal */}
      <IntentModal
        isOpen={showIntentModal}
        onClose={() => setShowIntentModal(false)}
        onSubmit={handleIntentSubmit}
        fileName={selectedFile?.name || session.fileName || ""}
        isLoading={isSubmittingIntent}
      />

      {/* Metadata Editor */}
      <MetadataEditor
        isOpen={showMetadataEditor}
        onClose={() => setShowMetadataEditor(false)}
        onApprove={handleApprove}
        metadata={session.metadata}
        fileName={selectedFile?.name || session.fileName || ""}
        isLoading={isApproving}
      />
    </div>
  );
};
