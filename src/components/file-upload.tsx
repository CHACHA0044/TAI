"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, File, CheckCircle } from "lucide-react";

interface FileUploadProps {
  accept: string;
  onFileSelect: (file: File) => void;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  maxSizeMB?: number;
}

export function FileUpload({
  accept,
  onFileSelect,
  label,
  description,
  icon,
  maxSizeMB = 50,
}: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setError(null);
      if (file.size > maxSizeMB * 1024 * 1024) {
        setError(`File size must be under ${maxSizeMB}MB`);
        return;
      }
      setSelectedFile(file);
      onFileSelect(file);
    },
    [maxSizeMB, onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-2">
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={onInputChange}
        className="sr-only"
        id="file-upload-input"
      />

      <AnimatePresence mode="wait">
        {selectedFile ? (
          <motion.div
            key="selected"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="glass rounded-2xl border border-emerald-500/30 p-5 flex items-center gap-4"
          >
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-white truncate">
                {selectedFile.name}
              </p>
              <p className="text-xs text-white/40 mt-0.5">
                {formatSize(selectedFile.size)}
              </p>
            </div>
            <button
              onClick={clearFile}
              className="w-8 h-8 flex items-center justify-center rounded-lg text-white/40 hover:text-white hover:bg-white/10 transition-all"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        ) : (
          <motion.label
            key="dropzone"
            htmlFor="file-upload-input"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`flex flex-col items-center justify-center gap-4 p-10 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300 ${
              dragOver
                ? "border-purple-500 bg-purple-500/10 scale-[1.01]"
                : "border-white/10 hover:border-white/20 hover:bg-white/2"
            }`}
          >
            <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
              {icon || <Upload className="w-7 h-7 text-white/40" />}
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-white/80">{label}</p>
              {description && (
                <p className="text-xs text-white/40 mt-1">{description}</p>
              )}
              <p className="text-xs text-white/30 mt-2">
                Max file size: {maxSizeMB}MB
              </p>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10">
              <File className="w-3.5 h-3.5 text-white/50" />
              <span className="text-xs text-white/50 font-medium">
                Browse files
              </span>
            </div>
          </motion.label>
        )}
      </AnimatePresence>

      {error && (
        <p className="text-xs text-red-400 px-1">{error}</p>
      )}
    </div>
  );
}
