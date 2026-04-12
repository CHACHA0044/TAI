"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Video as VideoIcon, Loader2, RefreshCcw, Film, AlertCircle, Brain, RotateCcw, Expand, Play } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeVideo, getJobStatus } from "@/lib/api";

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
  const stripValues = result?.frame_scores?.length ? result.frame_scores : result?.signals?.map((s) => s.confidence) || [];

  return (
    <div className="min-h-screen pt-32 pb-24">
      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-20 h-20 rounded-3xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mx-auto mb-8 relative"
          >
            <div className="absolute inset-0 bg-purple-500/20 blur-2xl rounded-full" />
            <VideoIcon className="w-10 h-10 text-purple-400 relative z-10" />
          </motion.div>
          <h1 className="text-4xl md:text-6xl font-black mb-6 tracking-tight">
            Video <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-fuchsia-400">Forensics</span>
          </h1>
          <p className="text-white/60 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
            Analyze temporal consistency, deepfake risk, lip-sync integrity, and compression behavior across sampled frames.
          </p>
        </div>

        <div className="flex flex-col gap-12">
          <div className="glass rounded-3xl p-6 border border-white/10">
            {!previewUrl ? (
              <FileUpload
                accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,video/avi"
                onFileSelect={handleFileSelect}
                label="Drag & drop video here"
                description="Supports MP4, WEBM, MOV, AVI"
                icon={<Film className="w-7 h-7 text-purple-400/50" />}
                maxSizeMB={100}
              />
            ) : (
              <div className="space-y-5">
                <div className="relative w-full aspect-video rounded-3xl overflow-hidden border border-white/10 bg-black/80">
                  <video ref={videoRef} src={previewUrl} controls preload="metadata" playsInline className="w-full h-full object-contain" />
                  {loading && (
                    <div className="absolute inset-0 bg-purple-900/65 backdrop-blur-md flex flex-col items-center justify-center text-white z-20">
                      <Loader2 className="w-12 h-12 animate-spin text-purple-300 mb-4" />
                      <p className="font-black uppercase tracking-[0.25em] text-xs mb-2">Analyzing Video</p>
                      <p className="text-xs text-purple-100 animate-pulse">{progress || "Running temporal and facial forensics..."}</p>
                    </div>
                  )}
                  {error && (
                    <div className="absolute inset-0 bg-red-900/80 backdrop-blur-md flex flex-col items-center justify-center text-white p-6 z-20 text-center">
                      <AlertCircle className="w-10 h-10 text-red-300 mb-4" />
                      <p className="font-bold uppercase mb-2">Analysis Failed</p>
                      <p className="text-sm text-red-100">{error}</p>
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap gap-2">
                  <button onClick={replayVideo} disabled={loading} className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/75 hover:text-white transition-all text-xs font-semibold disabled:opacity-40">
                    <Play className="w-3.5 h-3.5" />
                    Replay
                  </button>
                  <button onClick={openFullscreen} disabled={loading} className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/75 hover:text-white transition-all text-xs font-semibold disabled:opacity-40">
                    <Expand className="w-3.5 h-3.5" />
                    Fullscreen
                  </button>
                  <button onClick={reset} disabled={loading} className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/75 hover:text-white transition-all text-xs font-semibold disabled:opacity-40">
                    <RefreshCcw className="w-3.5 h-3.5" />
                    Re-upload
                  </button>
                  {result && (
                    <button onClick={handleAnalyze} disabled={loading} className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/75 hover:text-white transition-all text-xs font-semibold disabled:opacity-40">
                      <RotateCcw className="w-3.5 h-3.5" />
                      Retry analysis
                    </button>
                  )}
                  <button onClick={handleAnalyze} disabled={loading || !!result} className="flex-1 min-w-[180px] py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-fuchsia-600 text-white font-bold hover:shadow-[0_0_30px_rgba(168,85,247,0.4)] disabled:opacity-50 transition-all text-sm">
                    {loading ? "Analyzing…" : result ? "Analysis Complete ✓" : "Analyze Video"}
                  </button>
                </div>

                {(result?.intelligence || stripValues.length > 0) && (
                  <div className="space-y-4">
                    {result?.intelligence && (
                      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="p-4 rounded-2xl bg-purple-500/5 border border-purple-500/20">
                        <div className="flex items-center gap-2 mb-2 text-[10px] font-black uppercase tracking-widest text-purple-300">
                          <Brain className="w-3.5 h-3.5" />
                          <span>Scene Intelligence</span>
                        </div>
                        <p className="text-white/80 text-xs leading-relaxed">{result.intelligence.summary}</p>
                        {sceneTags.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-3">
                            {sceneTags.map((tag) => (
                              <span key={tag} className="px-2 py-0.5 rounded-md bg-white/5 border border-white/10 text-[9px] text-white/45 uppercase tracking-tighter">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </motion.div>
                    )}
                    {stripValues.length > 0 && (
                      <div className="rounded-2xl border border-white/10 bg-black/30 p-4 space-y-3">
                        <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-tighter text-white/45">
                          <span>Preview frame strip</span>
                          <span>{stripValues.length} samples</span>
                        </div>
                        <div className="h-12 w-full flex gap-1 items-end">
                          {stripValues.map((val, i) => (
                            <div
                              key={i}
                              style={{ height: `${Math.max(val * 100, 8)}%` }}
                              className={`flex-1 rounded-sm transition-all duration-500 ${val > 0.7 ? "bg-rose-500/70" : val > 0.4 ? "bg-amber-500/60" : "bg-emerald-500/45"}`}
                              title={`Sample ${i + 1}: ${Math.round(val * 100)}%`}
                            />
                          ))}
                        </div>
                        <p className="text-[10px] text-white/30">Higher bars indicate stronger synthetic signal in sampled frames.</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          <AnimatePresence mode="wait">
            {result ? (
              <motion.div key="result" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <div className="flex items-center gap-4 mb-2">
                  <div className="h-px flex-1 bg-white/5" />
                  <h3 className="font-black text-[10px] uppercase tracking-[0.3em] text-white/30">Video Forensic Intelligence Report</h3>
                  <div className="h-px flex-1 bg-white/5" />
                </div>
                <ResultDisplay result={result} />
              </motion.div>
            ) : (
              !loading && !previewUrl && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass rounded-[2rem] border border-white/5 flex flex-col items-center justify-center text-center p-16">
                  <div className="w-20 h-20 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-6">
                    <VideoIcon className="w-8 h-8 text-white/10" />
                  </div>
                  <h4 className="text-white/60 font-bold text-lg mb-2">No Video Loaded</h4>
                  <p className="text-white/30 text-sm max-w-xs">Upload a clip to begin temporal and deepfake forensics.</p>
                </motion.div>
              )
            )}
          </AnimatePresence>
        </div>
      </SectionWrapper>
    </div>
  );
}
