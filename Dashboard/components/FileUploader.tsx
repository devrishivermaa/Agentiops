import React, { useState, useCallback } from "react";
import { Upload, FileText, Loader2 } from "lucide-react";

interface FileUploaderProps {
  onUpload: (file: File) => void;
  isProcessing: boolean;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onUpload,
  isProcessing,
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onUpload(e.dataTransfer.files[0]);
      }
    },
    [onUpload]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto p-6">
      <div
        className={`
          w-full h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-300
          ${
            isDragging
              ? "border-primary bg-primary/10"
              : "border-zinc-700 hover:border-zinc-500 bg-zinc-800/30"
          }
          ${isProcessing ? "opacity-50 pointer-events-none" : "cursor-pointer"}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
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
          className="flex flex-col items-center cursor-pointer w-full h-full justify-center"
        >
          {isProcessing ? (
            <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
          ) : (
            <div className="bg-zinc-800 p-4 rounded-full mb-4">
              <Upload className="w-8 h-8 text-zinc-400" />
            </div>
          )}

          <h3 className="text-xl font-bold text-zinc-200 mb-2">
            {isProcessing ? "Initializing Pipeline..." : "Upload PDF Document"}
          </h3>
          <p className="text-zinc-500 text-sm max-w-xs text-center">
            Drag and drop your file here, or click to select. Supported formats:
            PDF
          </p>
        </label>
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
        <div className="bg-zinc-800 p-4 rounded-lg border border-zinc-800">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="font-semibold text-zinc-300">Extract</span>
          </div>
          <p className="text-xs text-zinc-500">
            Intelligent PDF parsing and sectioning
          </p>
        </div>
        <div className="bg-zinc-800 p-4 rounded-lg border border-zinc-800">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-yellow-500" />
            <span className="font-semibold text-zinc-300">Plan</span>
          </div>
          <p className="text-xs text-zinc-500">
            Master agent distributes workload strategy
          </p>
        </div>
        <div className="bg-zinc-800 p-4 rounded-lg border border-zinc-800">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span className="font-semibold text-zinc-300">Synthesize</span>
          </div>
          <p className="text-xs text-zinc-500">
            Workers analyze pages and compile results
          </p>
        </div>
      </div>
    </div>
  );
};
