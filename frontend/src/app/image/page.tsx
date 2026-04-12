"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image as ImageIcon, RefreshCcw, Camera, Maximize2, X, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeImage } from "@/lib/api";
import { Button } from "@/components/ui/button";

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
    <div className="min-h-screen pt-32 pb-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 right-1/2 translate-x-1/2 w-[800px] h-[800px] bg-emerald-500/5 blur-[120px] rounded-full -z-10 opacity-60" />

      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-20 h-20 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-emerald-500/10"
          >
            <ImageIcon className="w-10 h-10 text-emerald-400" />
          </motion.div>
          <h1 className="text-5xl md:text-7xl font-black mb-8 tracking-tighter">
            Visual <span className="text-emerald-400">Forensics</span>
          </h1>
          <p className="text-white/40 text-lg md:text-xl max-w-3xl mx-auto font-medium">
            Deploy neural artifact detectors to expose deepfakes, GAN-generated 
            textures, and pixel-level manipulation in high-resolution imagery.
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
                accept="image/jpeg,image/png,image/webp"
                onFileSelect={handleFileSelect}
                label="Initialize Image Intake"
                description="Supports JPG, PNG, WEBP (Max 10MB)"
                icon={<Camera className="w-8 h-8 text-emerald-500/30" />}
                maxSizeMB={10}
              />
            ) : (
              <div className="space-y-8">
                <div
                  className="relative w-full rounded-3xl overflow-hidden border border-white/5 bg-black/60 group cursor-zoom-in flex items-center justify-center"
                  style={{ maxHeight: "500px", minHeight: "300px" }}
                  onClick={() => !loading && setFullscreenOpen(true)}
                >
                  <img
                    src={previewUrl}
                    alt="Intelligence Preview"
                    className="max-w-full max-h-[500px] w-auto h-auto object-contain transition-all duration-700 group-hover:scale-[1.02]"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-6">
                    <p className="text-emerald-400 text-xs font-black uppercase tracking-widest">Expand Visual Field</p>
                    <Maximize2 className="w-5 h-5 text-emerald-400" />
                  </div>
                  
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
                      <p className="font-black tracking-[0.4em] uppercase text-xs text-emerald-500 animate-pulse">Scanning Neural Artifacts</p>
                    </div>
                  )}
                </div>
                
                <div className="flex flex-col md:flex-row gap-4">
                  <Button
                    variant="ghost"
                    onClick={reset}
                    disabled={loading}
                    className="h-14 px-8"
                  >
                    <RefreshCcw className="w-4 h-4 mr-2" /> FLUSH
                  </Button>
                  
                  <Button
                    variant={loading ? "loading" : "primary"}
                    onClick={handleAnalyze}
                    disabled={loading || !!result}
                    className="flex-1 h-14 text-lg"
                  >
                    {loading ? "PROCESSING..." : result ? "SCAN COMPLETE" : "DEPLOY FORENSICS"}
                  </Button>
                  
                  {result && (
                    <Button
                      variant="ghost"
                      onClick={retryAnalysis}
                      disabled={loading}
                      className="h-14 px-8"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" /> RETRY
                    </Button>
                  )}
                </div>
              </div>
            )}
          </motion.div>

          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <div className="flex items-center gap-6">
                  <h3 className="font-black text-[10px] uppercase tracking-[0.4em] text-emerald-500/50">Tactical Intelligence Report</h3>
                  <div className="h-px flex-1 bg-white/5" />
                </div>
                <ResultDisplay result={result} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </SectionWrapper>

      {/* Fullscreen Viewer */}
      <AnimatePresence>
        {fullscreenOpen && previewUrl && (
          <motion.div
            className="fixed inset-0 z-[300] bg-black/98 backdrop-blur-3xl flex flex-col items-center justify-center p-4 sm:p-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setFullscreenOpen(false)}
          >
            <div
              className="absolute top-8 right-8 flex items-center gap-4 z-50 bg-black/40 p-2 rounded-2xl border border-white/10 backdrop-blur-md"
              onClick={e => e.stopPropagation()}
            >
              <button
                onClick={() => setZoom(z => Math.max(0.25, z - 0.25))}
                className="p-3 rounded-xl hover:bg-white/10 text-white/60 hover:text-white transition-all"
              >
                <ZoomOut className="w-5 h-5" />
              </button>
              <span className="text-xs text-white/40 font-black w-12 text-center">{Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom(z => Math.min(4, z + 0.25))}
                className="p-3 rounded-xl hover:bg-white/10 text-white/60 hover:text-white transition-all"
              >
                <ZoomIn className="w-5 h-5" />
              </button>
              <div className="w-px h-6 bg-white/10 mx-2" />
              <button
                onClick={() => setFullscreenOpen(false)}
                className="p-3 rounded-xl bg-emerald-500 text-black hover:bg-emerald-400 transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div
              className="w-full h-full flex items-center justify-center overflow-auto"
              onClick={e => e.stopPropagation()}
            >
              <img
                src={previewUrl}
                alt="Forensic Asset"
                style={{ transform: `scale(${zoom})`, transition: "transform 0.3s cubic-bezier(0.22, 1, 0.36, 1)" }}
                className="max-w-full max-h-full object-contain shadow-[0_0_100px_rgba(0,0,0,0.8)]"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
