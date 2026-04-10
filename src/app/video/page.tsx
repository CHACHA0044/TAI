"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Video as VideoIcon, Loader2, RefreshCcw, Film } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeVideo } from "@/lib/api";

export default function VideoDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

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

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await analyzeVideo(file);
      setResult(res);
    } catch (error) {
      console.error(error);
      setResult({
        truth_score: 0.08,
        ai_generated_score: 0.92,
        bias_score: 0.1,
        credibility_score: 0.15,
        confidence_score: 0.96,
        explanation: "API call failed. Mock result: Audio-visual desynchronization detected. Lip-sync temporal anomalies indicate a generic facial re-enactment model.",
        features: {
          perplexity: 0,
          stylometry: { sentence_length_variance: 0, repetition_score: 0, lexical_diversity: 0 },
        },
        signals: [
          { source: "Lip Sync Analysis", verified: false, confidence: 0.08 },
          { source: "Frame Consistency Check", verified: false, confidence: 0.12 },
        ],
        metadata: { model: "mock-fallback", latency_ms: 0, timestamp: new Date().toISOString() },
      });
    } finally {
      setLoading(false);
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
                        <p className="font-bold tracking-widest uppercase mb-1">Frame-by-Frame Processing</p>
                        <p className="text-xs text-purple-300 animate-pulse">Analyzing micro-expressions...</p>
                      </div>
                    )}
                  </div>
                  
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
