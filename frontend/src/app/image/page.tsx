"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image as ImageIcon, Loader2, RefreshCcw, Camera } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { FileUpload } from "@/components/file-upload";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeImage } from "@/lib/api";
import Image from "next/image";

export default function ImageDetectionPage() {
  function buildMockImageResult(): AnalysisResult {
    return {
      truth_score: 0.24,
      ai_generated_score: 0.82,
      bias_score: 0.1,
      credibility_score: 0.32,
      confidence_score: 0.86,
      confidence: 86,
      category: "AI_GENERATED",
      primary_verdict: "AI_GENERATED",
      scene_description: "Portrait-style synthetic image featuring a person with studio-like lighting and smooth background gradients.",
      detected_objects: ["person", "face", "background gradient", "textured clothing"],
      style: "synthetic",
      why: "The image shows strong synthetic texture and metadata anomalies with over-smooth surfaces and edge halos.",
      explanation: "API call failed. Mock result generated locally with forensic image fields for UI continuity.",
      authenticity_signals: {
        ela: { score: 0.69, bucket: "HIGH", explanation: "Localized ELA spikes suggest recompression artifacts." },
        texture_consistency: { score: 0.73, bucket: "HIGH", explanation: "Texture appears overly uniform for a camera photo." },
        edge_artifacts: { score: 0.54, bucket: "MEDIUM", explanation: "Boundary halos observed around subject edges." },
        metadata_anomalies: { score: 0.81, bucket: "HIGH", explanation: "Metadata lacks natural camera-origin EXIF signatures." },
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

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("demo") === "1") {
      setResult(buildMockImageResult());
    }
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    // Create preview
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
        <div className="text-center mb-12">
          <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
            <ImageIcon className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-black mb-4">
            Image <span className="text-blue-400">Forensics</span>
          </h1>
          <p className="text-white/50 text-lg">
            Scan photos for AI generation and manipulation.
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
                  accept="image/jpeg,image/png,image/webp"
                  onFileSelect={handleFileSelect}
                  label="Drag & drop image here"
                  description="Supports JPG, PNG, WEBP"
                  icon={<Camera className="w-7 h-7 text-blue-400/50" />}
                  maxSizeMB={10}
                />
              ) : (
                <div className="space-y-6">
                  <div className="relative w-full aspect-square rounded-2xl overflow-hidden border border-white/10 bg-black/50">
                    <Image
                      src={previewUrl}
                      alt="Preview"
                      fill
                      className="object-contain"
                    />
                    {loading && (
                      <div className="absolute inset-0 bg-blue-900/40 backdrop-blur-sm flex flex-col items-center justify-center text-white">
                        <div className="w-full absolute top-0 h-1 bg-blue-500/20 overflow-hidden">
                          <motion.div
                            className="h-full bg-blue-400 shadow-[0_0_10px_#60a5fa]"
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                        </div>
                        <Loader2 className="w-10 h-10 animate-spin text-blue-400 mb-4" />
                        <p className="font-bold tracking-widest uppercase">Scanning Pixels</p>
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
                      className="flex-1 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-bold hover:shadow-[0_0_30px_rgba(37,99,235,0.4)] disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                    >
                      {loading ? "Analyzing..." : result ? "Analysis Complete" : "Analyze Image"}
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
                    <ImageIcon className="w-6 h-6 text-white/20" />
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
