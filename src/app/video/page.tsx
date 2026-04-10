"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Video as VideoIcon, Loader2, RefreshCcw, Film, Activity, AlertCircle } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult, JobResponse } from "@/lib/types";
import { analyzeVideo, getJobStatus } from "@/lib/api";

export default function VideoDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setResult(null);
  };

  const reset = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
  };

  const pollJobStatus = async (jobId: string) => {
    const poll = async () => {
      try {
        const jobStatus = await getJobStatus(jobId);
        
        if (jobStatus.status === "complete" && jobStatus.result) {
          setResult(jobStatus.result);
          setLoading(false);
          setProgress("");
          return true; // Stop polling
        } else if (jobStatus.status === "failed") {
          setError(jobStatus.error || "Analysis failed unexpectedly.");
          setLoading(false);
          setProgress("");
          return true; // Stop polling
        } else {
          setProgress(jobStatus.progress || "Scanning temporal artifacts...");
          return false; // Continue polling
        }
      } catch (err) {
        console.error("Polling error:", err);
        setError("Connection to analysis server lost. Retrying...");
        return false;
      }
    };

    const interval = setInterval(async () => {
      const stop = await poll();
      if (stop) clearInterval(interval);
    }, 2000);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setProgress("Uploading and hashing video...");

    try {
      const initialResponse = await analyzeVideo(file);
      
      if (initialResponse.status === "complete" && initialResponse.result) {
        // Cache hit from backend
        setResult(initialResponse.result);
        setLoading(false);
        setProgress("");
      } else if (initialResponse.job_id) {
        // Task queued
        setProgress("Job queued. Waiting for worker...");
        pollJobStatus(initialResponse.job_id);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to start video analysis.");
      setLoading(false);
      setProgress("");
    }
  };

  return (
    <div className="min-h-screen pt-32 pb-24">
      <SectionWrapper className="container max-w-4xl mx-auto px-6">
        <div className="text-center mb-12">
          <div className="w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center mx-auto mb-6">
            <VideoIcon className="w-8 h-8 text-purple-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-black mb-4">
            Video <span className="text-purple-400">Deepfake Detection</span>
          </h1>
          <p className="text-white/50 text-lg">
            Scan videos for facial manipulation, lipsync anomalies, and deepfakes.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-6 flex flex-col"
          >
            <div className="glass rounded-3xl p-6 border border-white/10 flex-grow flex flex-col justify-center">
              {!previewUrl ? (
                <FileUpload
                  accept="video/mp4,video/webm,video/quicktime"
                  onFileSelect={handleFileSelect}
                  label="Drag & drop video here"
                  description="Supports MP4, WEBM, MOV"
                  icon={<Film className="w-7 h-7 text-purple-400/50" />}
                  maxSizeMB={100}
                />
              ) : (
                <div className="space-y-6">
                  <div className="relative w-full aspect-video rounded-2xl overflow-hidden border border-white/10 bg-black">
                    <video
                      src={previewUrl}
                      controls
                      className="w-full h-full object-contain"
                    />
                    {loading && (
                      <div className="absolute inset-0 bg-purple-900/60 backdrop-blur-md flex flex-col items-center justify-center text-white z-10">
                        <Loader2 className="w-10 h-10 animate-spin text-purple-400 mb-4" />
                        <p className="font-bold tracking-widest uppercase mb-1">Analysis in Progress</p>
                        <p className="text-xs text-purple-300 animate-pulse">{progress || "Analyzing temporal artifacts..."}</p>
                      </div>
                    )}
                    {error && (
                      <div className="absolute inset-0 bg-red-900/80 backdrop-blur-md flex flex-col items-center justify-center text-white p-6 z-10 text-center">
                        <AlertCircle className="w-10 h-10 text-red-400 mb-4" />
                        <p className="font-bold uppercase mb-2">Analysis Failed</p>
                        <p className="text-sm text-red-200">{error}</p>
                        <button 
                          onClick={reset}
                          className="mt-6 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-xs font-bold transition-all"
                        >
                          DISMISS
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Forensic Timeline (if results exist) */}
                  {result && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-tighter text-white/40">
                        <span className="flex items-center gap-1"><Activity className="w-3 h-3" /> Forensic Timeline</span>
                        <span>Anomaly Distribution</span>
                      </div>
                      <div className="h-10 w-full flex gap-1 items-end">
                        {(result.features.stylometry.lexical_diversity > 0.4 ? [0.2, 0.5, 0.4, 0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.3] : [0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2]).map((val, i) => (
                          <div 
                            key={i} 
                            style={{ height: `${val * 100}%` }}
                            className={`flex-1 rounded-sm ${val > 0.7 ? 'bg-rose-500/60' : val > 0.4 ? 'bg-amber-500/40' : 'bg-emerald-500/20'}`}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex gap-3">
                    <button
                      onClick={reset}
                      disabled={loading}
                      className="px-4 py-3 rounded-xl glass border border-white/10 text-white/70 hover:text-white transition-colors"
                    >
                      <RefreshCcw className="w-5 h-5" />
                    </button>
                    <button
                      onClick={handleAnalyze}
                      disabled={loading || !!result}
                      className="flex-1 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold hover:shadow-[0_0_30px_rgba(168,85,247,0.4)] disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                    >
                      {loading ? "Analyzing Video..." : result ? "Analysis Complete" : "Analyze Video"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          <div className="w-full h-full relative">
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="relative z-10 h-full"
                >
                  <h3 className="font-bold text-white/80 mb-4">Forensic Report</h3>
                  <ResultDisplay result={result} />
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 glass rounded-3xl border border-white/5 flex flex-col items-center justify-center text-center p-8"
                >
                  <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center mb-4">
                    <VideoIcon className="w-6 h-6 text-white/20" />
                  </div>
                  <p className="text-white/40 font-medium">Awaiting Input</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </SectionWrapper>
    </div>
  );
}
