"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Loader2, Link as LinkIcon, RefreshCcw, Send, AlertCircle } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeText, analyzeURL } from "@/lib/api";

type InputMode = "text" | "url";

export default function TextDetectionPage() {
  const [mode, setMode] = useState<InputMode>("text");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res;
      if (mode === "url") {
        res = await analyzeURL(input);
      } else {
        res = await analyzeText(input);
      }
      setResult(res);
    } catch (err: any) {
      console.error(err);
      if (mode === "url") {
        setError("Unable to extract content. Please paste text manually.");
      } else {
        setError(err.message || "Analysis failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-32 pb-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-emerald-500/5 blur-[120px] rounded-full -z-10" />

      <SectionWrapper className="container max-w-6xl mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-16">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-20 h-20 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-emerald-500/10"
          >
            <FileText className="w-10 h-10 text-emerald-400" />
          </motion.div>
          <motion.h1 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="text-5xl md:text-6xl font-black mb-6 tracking-tight"
          >
            Truth<span className="text-emerald-400">Guard</span> Engine
          </motion.h1>
          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-white/50 text-xl max-w-2xl mx-auto font-medium"
          >
            Multi-modal verification for the modern information landscape. 
            Detect AI, verify claims, and uncover bias in seconds.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Input Side */}
          <div className="lg:col-span-12 xl:col-span-5 space-y-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass rounded-[2rem] p-8 border border-white/10 shadow-2xl relative"
            >
              {/* Mode Toggle */}
              <div className="flex bg-white/5 rounded-2xl p-1.5 mb-8 border border-white/5">
                <button 
                  onClick={() => { setMode("text"); setInput(""); setError(null); }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-black rounded-xl transition-all ${
                    mode === "text" ? "bg-emerald-500 text-white shadow-lg" : "text-white/40 hover:text-white/60"
                  }`}
                >
                  <FileText className="w-4 h-4" /> TEXT
                </button>
                <button 
                  onClick={() => { setMode("url"); setInput(""); setError(null); }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-black rounded-xl transition-all ${
                    mode === "url" ? "bg-emerald-500 text-white shadow-lg" : "text-white/40 hover:text-white/60"
                  }`}
                >
                  <LinkIcon className="w-4 h-4" /> URL
                </button>
              </div>

              {/* Input Field */}
              <div className="relative">
                {mode === "text" ? (
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Enter article text or claim to verify..."
                    className="w-full h-64 bg-black/40 border border-white/10 rounded-2xl p-6 text-white placeholder:text-white/20 resize-none focus:outline-none focus:border-emerald-500/50 transition-all font-medium text-lg leading-relaxed shadow-inner"
                  />
                ) : (
                  <div className="relative">
                    <input
                      type="url"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="https://example.com/article"
                      className="w-full bg-black/40 border border-white/10 rounded-2xl p-6 text-white placeholder:text-white/20 focus:outline-none focus:border-emerald-500/50 transition-all font-medium text-lg shadow-inner"
                    />
                  </div>
                )}
                
                {mode === "text" && (
                  <div className="absolute bottom-4 right-6 flex items-center gap-4">
                    <span className={`text-[10px] font-black uppercase tracking-widest ${input.length > 4500 ? 'text-rose-400' : 'text-white/20'}`}>
                      {input.length} / 5000
                    </span>
                  </div>
                )}
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {error && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center gap-3 text-rose-400 text-sm font-bold"
                  >
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit Button */}
              <button
                onClick={handleAnalyze}
                disabled={loading || !input.trim()}
                className="w-full mt-8 py-5 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white font-black text-xl hover:shadow-[0_0_40px_rgba(16,185,129,0.3)] hover:-translate-y-1 active:translate-y-0 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3 group"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" /> 
                    <span>VERIFYING SIGNALS...</span>
                  </>
                ) : (
                  <>
                    <span>INVOKE GUARDIAN</span>
                    <Send className="w-5 h-5 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                  </>
                )}
              </button>
            </motion.div>

            <div className="flex items-center gap-6 px-4">
              <div className="flex -space-x-3">
                {[1,2,3,4].map(i => (
                  <div key={i} className="w-8 h-8 rounded-full border-2 border-black bg-white/10" />
                ))}
              </div>
              <p className="text-xs text-white/30 font-bold">10k+ analysts using TruthGuard</p>
            </div>
          </div>

          {/* Results Side */}
          <div className="lg:col-span-12 xl:col-span-7 min-h-[600px]">
            {result ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between px-2">
                  <h3 className="text-sm font-black text-white/40 uppercase tracking-[0.3em]">Analysis Complete</h3>
                  <button 
                    onClick={() => { setResult(null); setInput(""); setError(null); }}
                    className="flex items-center gap-2 text-[10px] font-black text-emerald-400 hover:text-emerald-300 transition-colors uppercase tracking-widest"
                  >
                    <RefreshCcw className="w-3 h-3" /> New Session
                  </button>
                </div>
                <ResultDisplay result={result} />
              </div>
            ) : (
              <div className="h-full glass rounded-[2rem] border border-white/5 flex flex-col items-center justify-center text-center p-12 relative group">
                <div className="absolute inset-0 bg-emerald-500/[0.02] opacity-0 group-hover:opacity-100 transition-opacity rounded-[2rem]" />
                <div className="w-24 h-24 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-8 bg-white/[0.02]">
                  <RefreshCcw className="w-8 h-8 text-white/10 animate-pulse" />
                </div>
                <h3 className="text-2xl font-black text-white/60 mb-4 tracking-tight">System Idle</h3>
                <p className="text-white/30 font-medium max-w-sm">
                  The Text Verification engine is ready. Input raw content or a link on the left to begin deep-signal analysis.
                </p>
                <div className="mt-12 grid grid-cols-2 gap-4 w-full max-w-sm">
                  <div className="p-4 rounded-2xl border border-white/5 bg-white/[0.01] text-xs font-bold text-white/20 uppercase tracking-widest">
                    AI Detection
                  </div>
                  <div className="p-4 rounded-2xl border border-white/5 bg-white/[0.01] text-xs font-bold text-white/20 uppercase tracking-widest">
                    Bias Scoring
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </SectionWrapper>
    </div>
  );
}
