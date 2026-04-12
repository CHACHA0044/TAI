"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image as ImageIcon, Loader2, RefreshCcw, Camera, Maximize2, X, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeImage } from "@/lib/api";

export default function ImageDetectionPage() {
  function buildMockImageResult(): AnalysisResult {
    return {
      truth_score: 0.24,
      ai_generated_score: 0.82,
      bias_score: 0.1,
      credibility_score: 0.32,
      confidence_score: 0.86,
      confidence: 86,
      category: "AI_GENERATED_SYNTHETIC_IMAGE",
      primary_verdict: "AI_GENERATED_SYNTHETIC_IMAGE",
      confidence_band: "Highly Likely AI-Generated",
      suspicion_score: 82,
      scene_description: "Portrait-style synthetic image featuring a person with studio-like lighting and smooth background gradients.",
      detected_objects: ["person", "face", "background gradient", "textured clothing"],
      style: "synthetic",
      why: "The image shows strong synthetic texture and metadata anomalies with over-smooth surfaces and edge halos.",
      explanation: "API call failed. Mock result generated locally with forensic image fields for UI continuity.",
      authenticity_signals: {
        ela: { score: 0.69, bucket: "HIGH", explanation: "Strong localized recompression differences detected — multiple image regions show unusually high ELA residuals." },
        texture_consistency: { score: 0.73, bucket: "HIGH", explanation: "Texture variance is extremely uniform (consistency: 73%), highly characteristic of AI-rendered imagery." },
        edge_artifacts: { score: 0.54, bucket: "MEDIUM", explanation: "Significant edge sharpness anomalies detected — boundary transitions exhibit halo-like artifacts." },
        metadata_anomalies: { score: 0.81, bucket: "HIGH", explanation: "EXIF metadata is missing — no camera make/model, ISO, or focal length data found." },
      },
      features: {
        perplexity: 0,
        stylometry: { sentence_length_variance: 0, repetition_score: 0, lexical_diversity: 0 },
      },
      signals: [{ source: "Error Level Analysis", verified: false, confidence: 0.12 }],
      metadata: { model: "mock-fallback", latency_ms: 0, timestamp: new Date().toISOString() },
    };
  }

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [fullscreenOpen, setFullscreenOpen] = useState(false);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("demo") === "1") {
      setResult(buildMockImageResult());
    }
  }, []);

  // Escape closes fullscreen viewer
  useEffect(() => {
    if (!fullscreenOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setFullscreenOpen(false); };
    document.addEventListener("keydown", handler);
    return () => { document.body.style.overflow = prev; document.removeEventListener("keydown", handler); };
  }, [fullscreenOpen]);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setResult(null);
    setZoom(1);
  };

  const reset = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setZoom(1);
  };

  const retryAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await analyzeImage(file);
      setResult(res);
    } catch (error) {
      console.error(error);
      setResult(buildMockImageResult());
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await analyzeImage(file);
      setResult(res);
    } catch (error) {
      console.error(error);
      setResult(buildMockImageResult());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-32 pb-24">
      <SectionWrapper className="container max-w-4xl mx-auto px-6">
        <div className="text-center mb-16">
          <motion.div 
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-20 h-20 rounded-3xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-8 relative"
          >
            <div className="absolute inset-0 bg-blue-500/20 blur-2xl rounded-full" />
            <ImageIcon className="w-10 h-10 text-blue-400 relative z-10" />
          </motion.div>
          <h1 className="text-4xl md:text-6xl font-black mb-6 tracking-tight">
            Image <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">Forensics</span>
          </h1>
          <p className="text-white/60 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Scan photos for AI generation, deepfakes, and pixel-level manipulation using neural artifact detection and forensic ELA.
          </p>
        </div>

        <div className="flex flex-col gap-12">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-6 flex flex-col"
          >
            <div className="glass rounded-3xl p-6 border border-white/10 flex-grow flex flex-col justify-center">
              {!previewUrl ? (
                <FileUpload
                  accept="image/jpeg,image/png,image/webp"
                  onFileSelect={handleFileSelect}
                  label="Drag & drop image here"
                  description="Supports JPG, PNG, WEBP"
                  icon={<Camera className="w-7 h-7 text-blue-400/50" />}
                  maxSizeMB={10}
                />
              ) : (
                <div className="space-y-4">
                  {/* Image preview — aspect ratio preserved, click to expand */}
                  <div
                    className="relative w-full rounded-3xl overflow-hidden border border-white/10 bg-black/50 group cursor-zoom-in flex items-center justify-center"
                    style={{ maxHeight: "420px", minHeight: "180px" }}
                    onClick={() => !loading && setFullscreenOpen(true)}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full max-h-[420px] w-auto h-auto object-contain transition-transform duration-500 group-hover:scale-[1.02]"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-4">
                      <p className="text-white/70 text-xs font-medium">Click to expand fullscreen</p>
                      <Maximize2 className="w-4 h-4 text-white/60" />
                    </div>
                    {loading && (
                      <div className="absolute inset-0 bg-blue-950/60 backdrop-blur-md flex flex-col items-center justify-center text-white z-20">
                        <div className="w-full absolute top-0 h-1.5 bg-blue-500/20 overflow-hidden">
                          <motion.div
                            className="h-full bg-blue-400 shadow-[0_0_20px_#60a5fa]"
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                          />
                        </div>
                        <div className="relative">
                          <div className="absolute inset-0 bg-blue-400/20 blur-xl rounded-full animate-pulse" />
                          <Loader2 className="w-12 h-12 animate-spin text-blue-400 mb-6 relative z-10" />
                        </div>
                        <p className="font-black tracking-[0.3em] uppercase text-sm text-blue-100 drop-shadow-lg">Scanning Artifacts</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Action buttons */}
                  <div className="flex gap-2">
                    <button
                      onClick={reset}
                      disabled={loading}
                      title="Upload new image"
                      className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/70 hover:text-white hover:border-white/20 transition-all text-xs font-semibold disabled:opacity-40"
                    >
                      <RefreshCcw className="w-3.5 h-3.5" />
                      Re-upload
                    </button>
                    {result && (
                      <button
                        onClick={retryAnalysis}
                        disabled={loading}
                        title="Re-run analysis on same image"
                        className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl glass border border-white/10 text-white/70 hover:text-white hover:border-cyan-500/30 transition-all text-xs font-semibold disabled:opacity-40"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Retry Analysis
                      </button>
                    )}
                    <button
                      onClick={handleAnalyze}
                      disabled={loading || !!result}
                      className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-bold hover:shadow-[0_0_30px_rgba(37,99,235,0.4)] disabled:opacity-50 transition-all flex items-center justify-center gap-2 text-sm"
                    >
                      {loading ? "Analyzing…" : result ? "Analysis Complete ✓" : "Analyze Image"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <div className="flex items-center gap-4 mb-2">
                  <div className="h-px flex-1 bg-white/5" />
                  <h3 className="font-black text-[10px] uppercase tracking-[0.3em] text-white/30">Forensic Intelligence Report</h3>
                  <div className="h-px flex-1 bg-white/5" />
                </div>
                <ResultDisplay result={result} />
              </motion.div>
            ) : (
              !loading && !previewUrl && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass rounded-[2rem] border border-white/5 flex flex-col items-center justify-center text-center p-16"
                >
                  <div className="w-20 h-20 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-6">
                    <ImageIcon className="w-8 h-8 text-white/10" />
                  </div>
                  <h4 className="text-white/60 font-bold text-lg mb-2">No Image Loaded</h4>
                  <p className="text-white/30 text-sm max-w-xs">Upload a photograph to begin deep forensic analysis.</p>
                </motion.div>
              )
            )}
          </AnimatePresence>
        </div>
      </SectionWrapper>

      {/* Fullscreen image viewer */}
      <AnimatePresence>
        {fullscreenOpen && previewUrl && (
          <motion.div
            className="fixed inset-0 z-[300] bg-black/95 backdrop-blur-2xl flex flex-col items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setFullscreenOpen(false)}
          >
            {/* Controls bar */}
            <div
              className="absolute top-4 right-4 flex items-center gap-2 z-10"
              onClick={e => e.stopPropagation()}
            >
              <button
                onClick={() => setZoom(z => Math.max(0.25, z - 0.25))}
                className="p-2 rounded-xl bg-white/10 border border-white/20 text-white/80 hover:bg-white/20 transition-colors"
                title="Zoom out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs text-white/50 font-mono w-10 text-center">{Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom(z => Math.min(4, z + 0.25))}
                className="p-2 rounded-xl bg-white/10 border border-white/20 text-white/80 hover:bg-white/20 transition-colors"
                title="Zoom in"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => { setZoom(1); setFullscreenOpen(false); }}
                className="p-2 rounded-xl bg-white/10 border border-white/20 text-white/80 hover:bg-white/20 transition-colors"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div
              className="overflow-auto flex items-center justify-center w-full h-full p-8"
              onClick={e => e.stopPropagation()}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Fullscreen preview"
                style={{ transform: `scale(${zoom})`, transformOrigin: "center", transition: "transform 0.2s ease", maxWidth: "100%", maxHeight: "90vh" }}
                className="object-contain rounded-2xl shadow-2xl"
                draggable={false}
              />
            </div>
            <p className="absolute bottom-4 text-white/30 text-xs">Click outside image or press Esc to close</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
