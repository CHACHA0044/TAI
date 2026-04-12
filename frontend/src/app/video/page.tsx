"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Video as VideoIcon, RefreshCcw, Film, AlertCircle, Brain, RotateCcw, Expand, Play } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeVideo, getJobStatus } from "@/lib/api";
import { Button } from "@/components/ui/button";

export default function VideoDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const reset = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setProgress("");
  };

  const pollJobStatus = async (jobId: string) => {
    const poll = async () => {
      try {
        const jobStatus = await getJobStatus(jobId);
        if (jobStatus.status === "complete" && jobStatus.result) {
          setResult(jobStatus.result);
          setLoading(false);
          setProgress("");
          return true;
        }
        if (jobStatus.status === "failed") {
          setError(jobStatus.error || "Analysis failed unexpectedly.");
          setLoading(false);
          setProgress("");
          return true;
        }
        setProgress(jobStatus.progress || "Scanning temporal artifacts...");
        return false;
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
        setResult(initialResponse.result);
        setLoading(false);
        setProgress("");
      } else if (initialResponse.job_id) {
        setProgress("Job queued. Waiting for worker...");
        pollJobStatus(initialResponse.job_id);
      }
    } catch (err: unknown) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to start video analysis.");
      setLoading(false);
      setProgress("");
    }
  };

  const replayVideo = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play().catch(() => undefined);
  };

  const openFullscreen = async () => {
    if (!videoRef.current?.requestFullscreen) return;
    await videoRef.current.requestFullscreen();
  };

  const sceneTags = result?.intelligence?.tags || [];
  const stripValues = result?.frame_scores?.length ? result.frame_scores : result?.signals?.map((s: any) => s.confidence) || [];

  return (
    <div className="min-h-screen pt-32 pb-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-emerald-500/5 blur-[120px] rounded-full -z-10 opacity-60" />

      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-20 h-20 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-emerald-500/10"
          >
            <VideoIcon className="w-10 h-10 text-emerald-400" />
          </motion.div>
          <h1 className="text-5xl md:text-7xl font-black mb-8 tracking-tighter">
            Video <span className="text-emerald-400">Forensics</span>
          </h1>
          <p className="text-white/40 text-lg md:text-xl max-w-3xl mx-auto font-medium">
            Analyze temporal consistency, deepfake risks, lip-sync integrity, and 
            compression behavior across high-frequency sampled frames.
          </p>
        </div>

        <div className="flex flex-col gap-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/[0.02] rounded-[2.5rem] p-10 border border-white/5 shadow-2xl relative backdrop-blur-md"
          >
            {!previewUrl ? (
              <FileUpload
                accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,video/avi"
                onFileSelect={handleFileSelect}
                label="Initialize Video Intake"
                description="Supports MP4, WEBM, MOV, AVI (Max 100MB)"
                icon={<Film className="w-8 h-8 text-emerald-500/30" />}
                maxSizeMB={100}
              />
            ) : (
              <div className="space-y-8">
                <div className="relative w-full aspect-video rounded-3xl overflow-hidden border border-white/5 bg-black/60 shadow-inner">
                  <video ref={videoRef} src={previewUrl} controls preload="metadata" playsInline className="w-full h-full object-contain" />
                  
                  {loading && (
                    <div className="absolute inset-0 bg-black/80 backdrop-blur-xl flex flex-col items-center justify-center z-50">
                      <div className="w-64 h-1 bg-white/5 rounded-full mb-8 overflow-hidden">
                        <motion.div
                          className="h-full bg-emerald-500"
                          initial={{ x: "-100%" }}
                          animate={{ x: "100%" }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                        />
                      </div>
                      <p className="font-black tracking-[0.4em] uppercase text-xs text-emerald-500 animate-pulse mb-3">Analyzing Intelligence</p>
                      <p className="text-[10px] font-bold text-white/30 uppercase tracking-widest">{progress || "Scanning samples..."}</p>
                    </div>
                  )}

                  {error && (
                    <div className="absolute inset-0 bg-rose-950/90 backdrop-blur-xl flex flex-col items-center justify-center p-8 z-50 text-center">
                      <AlertCircle className="w-12 h-12 text-rose-500 mb-4" />
                      <p className="font-black uppercase tracking-widest text-rose-200 mb-2">Analysis Failed</p>
                      <p className="text-sm text-rose-300/60 font-medium">{error}</p>
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button variant="ghost" onClick={replayVideo} disabled={loading} className="h-12 px-6">
                    <Play className="w-4 h-4 mr-2" /> REPLAY
                  </Button>
                  <Button variant="ghost" onClick={openFullscreen} disabled={loading} className="h-12 px-6">
                    <Expand className="w-4 h-4 mr-2" /> FULLSCREEN
                  </Button>
                  <Button variant="ghost" onClick={reset} disabled={loading} className="h-12 px-6">
                    <RefreshCcw className="w-4 h-4 mr-2" /> FLUSH
                  </Button>
                  
                  <Button
                    variant={loading ? "loading" : "primary"}
                    onClick={handleAnalyze}
                    disabled={loading || !!result}
                    className="flex-1 h-12 text-lg"
                  >
                    {loading ? "PROCESSING..." : result ? "SCAN COMPLETE" : "DEPLOY ANALYSIS"}
                  </Button>
                  
                  {result && (
                    <Button variant="ghost" onClick={handleAnalyze} disabled={loading} className="h-12 px-6">
                      <RotateCcw className="w-4 h-4 mr-2" /> RETRY
                    </Button>
                  )}
                </div>

                {result && (
                  <div className="space-y-6 pt-4 border-t border-white/5">
                    {result?.intelligence && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }} 
                        animate={{ opacity: 1, scale: 1 }} 
                        className="p-6 rounded-2xl bg-white/[0.02] border border-white/5"
                      >
                        <div className="flex items-center gap-2 mb-4">
                          <Brain className="w-4 h-4 text-emerald-500/50" />
                          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-500/50">Tactical Context</span>
                        </div>
                        <p className="text-white/60 text-sm font-medium leading-relaxed">{result.intelligence.summary}</p>
                        {sceneTags.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-4">
                            {sceneTags.map((tag: string) => (
                              <span key={tag} className="px-3 py-1 rounded-lg bg-black/40 border border-white/5 text-[10px] text-white/40 font-bold uppercase tracking-widest">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </motion.div>
                    )}

                    {stripValues.length > 0 && (
                      <div className="p-6 rounded-2xl border border-white/5 bg-black/40">
                        <div className="flex items-center justify-between mb-6">
                          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/20">Temporal Artifact Strip</span>
                          <span className="text-[10px] font-black text-white/20">{stripValues.length} QUADRANTS</span>
                        </div>
                        <div className="h-16 w-full flex gap-1 items-end">
                          {stripValues.map((val: number, i: number) => (
                            <motion.div
                              key={i}
                              initial={{ height: 0 }}
                              animate={{ height: `${Math.max(val * 100, 10)}%` }}
                              className={`flex-1 rounded-sm ${
                                val > 0.7 ? "bg-rose-500/60" : val > 0.4 ? "bg-emerald-500/60" : "bg-emerald-500/20"
                              }`}
                              title={`Unit ${i + 1}: ${Math.round(val * 100)}%`}
                            />
                          ))}
                        </div>
                        <p className="text-[10px] font-black text-white/10 mt-4 uppercase tracking-[0.2em]">High frequency AI indication detected in upper spectrum</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </motion.div>

          <AnimatePresence mode="wait">
            {result && (
              <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <div className="flex items-center gap-6">
                  <h3 className="font-black text-[10px] uppercase tracking-[0.4em] text-emerald-500/50">Comprehensive Forensic Export</h3>
                  <div className="h-px flex-1 bg-white/5" />
                </div>
                <ResultDisplay result={result} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </SectionWrapper>
    </div>
  );
}
