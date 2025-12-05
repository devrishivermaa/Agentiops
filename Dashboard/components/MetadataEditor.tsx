import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  CheckCircle2,
  Edit3,
  Save,
  ArrowRight,
  FileText,
  Loader2,
  AlertCircle,
  Settings,
  Layers,
  Clock,
  BookOpen,
  ChevronDown,
  ChevronRight,
  RotateCcw,
} from "lucide-react";

interface MetadataEditorProps {
  isOpen: boolean;
  onClose: () => void;
  onApprove: (
    approved: boolean,
    modifiedMetadata?: Record<string, any>
  ) => void;
  metadata: Record<string, any> | null;
  fileName: string;
  isLoading?: boolean;
}

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon,
  children,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-zinc-700/50 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-primary">{icon}</span>
          <span className="font-medium text-zinc-200">{title}</span>
        </div>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-zinc-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-zinc-400" />
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 space-y-3 bg-zinc-900/50">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

interface EditableFieldProps {
  label: string;
  value: any;
  onChange: (value: any) => void;
  type?: "text" | "number" | "textarea" | "select";
  options?: { value: string; label: string }[];
  disabled?: boolean;
}

const EditableField: React.FC<EditableFieldProps> = ({
  label,
  value,
  onChange,
  type = "text",
  options,
  disabled = false,
}) => {
  const inputClass =
    "w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 disabled:opacity-50";

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
        {label}
      </label>
      {type === "textarea" ? (
        <textarea
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          className={`${inputClass} h-20 resize-none`}
          disabled={disabled}
        />
      ) : type === "select" && options ? (
        <select
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          className={inputClass}
          disabled={disabled}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : type === "number" ? (
        <input
          type="number"
          value={value || 0}
          onChange={(e) => onChange(parseInt(e.target.value) || 0)}
          className={inputClass}
          disabled={disabled}
        />
      ) : (
        <input
          type="text"
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          className={inputClass}
          disabled={disabled}
        />
      )}
    </div>
  );
};

export const MetadataEditor: React.FC<MetadataEditorProps> = ({
  isOpen,
  onClose,
  onApprove,
  metadata: initialMetadata,
  fileName,
  isLoading = false,
}) => {
  const [metadata, setMetadata] = useState<Record<string, any>>({});
  const [isEditing, setIsEditing] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (initialMetadata) {
      setMetadata({ ...initialMetadata });
      setHasChanges(false);
    }
  }, [initialMetadata]);

  const updateField = (path: string, value: any) => {
    setMetadata((prev) => {
      const keys = path.split(".");
      const newMetadata = { ...prev };
      let current: any = newMetadata;

      for (let i = 0; i < keys.length - 1; i++) {
        current[keys[i]] = { ...current[keys[i]] };
        current = current[keys[i]];
      }

      current[keys[keys.length - 1]] = value;
      return newMetadata;
    });
    setHasChanges(true);
  };

  const handleReset = () => {
    if (initialMetadata) {
      setMetadata({ ...initialMetadata });
      setHasChanges(false);
    }
  };

  const handleApprove = () => {
    if (hasChanges) {
      onApprove(false, metadata);
    } else {
      onApprove(true);
    }
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
          className="bg-zinc-900 border border-zinc-700/50 rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 shrink-0">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-500/10 rounded-lg">
                <FileText className="w-5 h-5 text-amber-500" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-zinc-100">
                  Review & Approve Metadata
                </h2>
                <p className="text-sm text-zinc-500">{fileName}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsEditing(!isEditing)}
                disabled={isLoading}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  isEditing
                    ? "bg-primary/10 text-primary border border-primary/30"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
              >
                <Edit3 size={14} />
                {isEditing ? "Editing" : "Edit"}
              </button>
              <button
                onClick={onClose}
                disabled={isLoading}
                className="p-2 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50"
              >
                <X className="w-5 h-5 text-zinc-400" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {/* Document Info */}
            <CollapsibleSection
              title="Document Information"
              icon={<BookOpen size={18} />}
              defaultOpen={true}
            >
              <div className="grid grid-cols-2 gap-4">
                <EditableField
                  label="File Name"
                  value={metadata.file_name}
                  onChange={(v) => updateField("file_name", v)}
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Document Type"
                  value={metadata.document_type}
                  onChange={(v) => updateField("document_type", v)}
                  type="select"
                  options={[
                    { value: "research_paper", label: "Research Paper" },
                    { value: "report", label: "Report" },
                    { value: "article", label: "Article" },
                    { value: "book", label: "Book" },
                    { value: "other", label: "Other" },
                  ]}
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Number of Pages"
                  value={metadata.num_pages}
                  onChange={(v) => updateField("num_pages", v)}
                  type="number"
                  disabled={true}
                />
                <EditableField
                  label="Complexity Level"
                  value={metadata.complexity_level}
                  onChange={(v) => updateField("complexity_level", v)}
                  type="select"
                  options={[
                    { value: "low", label: "Low" },
                    { value: "medium", label: "Medium" },
                    { value: "high", label: "High" },
                  ]}
                  disabled={!isEditing || isLoading}
                />
              </div>
            </CollapsibleSection>

            {/* Processing Intent */}
            <CollapsibleSection
              title="Processing Intent"
              icon={<Settings size={18} />}
              defaultOpen={true}
            >
              <div className="space-y-4">
                <EditableField
                  label="High-Level Intent"
                  value={metadata.high_level_intent || metadata.user_notes}
                  onChange={(v) => updateField("high_level_intent", v)}
                  type="textarea"
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Document Context"
                  value={metadata.user_document_context || ""}
                  onChange={(v) => updateField("user_document_context", v)}
                  type="textarea"
                  disabled={!isEditing || isLoading}
                />
              </div>
            </CollapsibleSection>

            {/* Processing Configuration */}
            <CollapsibleSection
              title="Processing Configuration"
              icon={<Layers size={18} />}
              defaultOpen={false}
            >
              <div className="grid grid-cols-2 gap-4">
                <EditableField
                  label="Max Parallel SubMasters"
                  value={metadata.max_parallel_submasters}
                  onChange={(v) => updateField("max_parallel_submasters", v)}
                  type="number"
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Workers per SubMaster"
                  value={metadata.num_workers_per_submaster}
                  onChange={(v) => updateField("num_workers_per_submaster", v)}
                  type="number"
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Preferred Model"
                  value={metadata.preferred_model}
                  onChange={(v) => updateField("preferred_model", v)}
                  type="select"
                  options={[
                    { value: "mistral-small-latest", label: "Mistral Small" },
                    { value: "mistral-medium-latest", label: "Mistral Medium" },
                    { value: "mistral-large-latest", label: "Mistral Large" },
                  ]}
                  disabled={!isEditing || isLoading}
                />
                <EditableField
                  label="Priority"
                  value={metadata.priority}
                  onChange={(v) => updateField("priority", v)}
                  type="select"
                  options={[
                    { value: "low", label: "Low" },
                    { value: "medium", label: "Medium" },
                    { value: "high", label: "High" },
                  ]}
                  disabled={!isEditing || isLoading}
                />
              </div>
            </CollapsibleSection>

            {/* Sections */}
            {metadata.sections && (
              <CollapsibleSection
                title={`Document Sections (${
                  Object.keys(metadata.sections).length
                })`}
                icon={<Clock size={18} />}
                defaultOpen={false}
              >
                <div className="space-y-2">
                  {Object.entries(metadata.sections).map(
                    ([name, section]: [string, any]) => (
                      <div
                        key={name}
                        className="flex items-center justify-between px-3 py-2 bg-zinc-800/50 rounded-lg"
                      >
                        <span className="text-sm font-medium text-zinc-200">
                          {name}
                        </span>
                        <span className="text-xs text-zinc-500">
                          Pages {section.page_start} - {section.page_end}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </CollapsibleSection>
            )}

            {/* Raw JSON View */}
            <CollapsibleSection
              title="Raw Metadata (JSON)"
              icon={<FileText size={18} />}
              defaultOpen={false}
            >
              <pre className="text-xs text-zinc-400 bg-zinc-950 p-4 rounded-lg overflow-x-auto max-h-60">
                {JSON.stringify(metadata, null, 2)}
              </pre>
            </CollapsibleSection>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-6 py-4 border-t border-zinc-800 bg-zinc-900/50 shrink-0">
            <div className="flex items-center gap-2">
              {hasChanges && (
                <button
                  onClick={handleReset}
                  disabled={isLoading}
                  className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50"
                >
                  <RotateCcw size={14} />
                  Reset Changes
                </button>
              )}
              {hasChanges && (
                <span className="text-xs text-amber-500 flex items-center gap-1">
                  <AlertCircle size={12} />
                  Unsaved changes
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
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
                onClick={handleApprove}
                disabled={isLoading}
                className={`
                  flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all
                  ${
                    !isLoading
                      ? "bg-green-600 text-white shadow-lg shadow-green-600/25 hover:shadow-green-600/40 hover:bg-green-500"
                      : "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={16} />
                    {hasChanges ? "Save & Approve" : "Approve & Start"}
                    <ArrowRight size={16} />
                  </>
                )}
              </motion.button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
