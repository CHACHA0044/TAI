"use client";

import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
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
  const prefersReducedMotion = useReducedMotion();

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
    } catch (err: unknown) {
      console.error(err);
      if (mode === "url") {
        setError("Unable to extract content. Please paste text manually.");
      } else {
        const msg = err instanceof Error ? err.message : "Analysis failed. Please try again.";
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-24 sm:pt-32 pb-16 sm:pb-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[400px] h-[400px] sm:w-[800px] sm:h-[800px] bg-emerald-500/5 blur-[80px] sm:blur-[120px] rounded-full -z-10 opacity-50 md:opacity-100" />

      <SectionWrapper className="container w-full max-w-6xl mx-auto px-4 sm:px-6">
        {/* Header */}
        <div className="text-center mb-10 sm:mb-16">
          <motion.div 
            initial={prefersReducedMotion ? { opacity: 1 } : { scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl sm:rounded-3xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-6 sm:mb-8 shadow-2xl shadow-emerald-500/10"
          >
            <FileText className="w-8 h-8 sm:w-10 sm:h-10 text-emerald-400" />
          </motion.div>
          <motion.h1 
            initial={prefersReducedMotion ? { opacity: 1 } : { y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="text-4xl sm:text-5xl md:text-6xl font-black mb-4 sm:mb-6 tracking-tight flex flex-col sm:block"
          >
            <span>AI-Powered <span className="text-emerald-400">Fact</span></span>
            <span className="ml-0 sm:ml-3">Verification</span>
          </motion.h1>
          <motion.p 
            initial={prefersReducedMotion ? { opacity: 1 } : { y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: prefersReducedMotion ? 0 : 0.1 }}
            className="text-white/50 text-base sm:text-lg md:text-xl max-w-2xl mx-auto font-medium px-2"
          >
            Paste any claim, article, or URL. TruthGuard analyses it for factual accuracy, bias, manipulation, sarcasm, and AI generation — in seconds.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-12 gap-8 sm:gap-10">
          {/* Input Side */}
          <div className="lg:col-span-12 xl:col-span-5 space-y-6 sm:space-y-8">
            <motion.div
              initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass rounded-[1.5rem] sm:rounded-[2rem] p-5 sm:p-8 border border-white/10 shadow-2xl relative"
            >
              {/* Panel heading */}
              <div className="mb-5 sm:mb-6">
                <h2 className="text-base sm:text-lg font-black text-white/80 tracking-tight">
                  What would you like to verify?
                </h2>
                <p className="text-[11px] sm:text-xs text-white/30 mt-1 font-medium">
                  Enter a claim, paste an article, or drop a URL — we&apos;ll do the rest.
                </p>
              </div>

              {/* Mode Toggle */}
              <div className="mb-2">
                <p className="text-[10px] font-black uppercase tracking-[0.18em] text-white/30 mb-2">Input type</p>
                <div className="flex bg-white/5 rounded-2xl p-1.5 border border-white/5">
                <button 
                  onClick={() => { setMode("text"); setInput(""); setError(null); }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-black rounded-xl transition-all min-h-[44px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                    mode === "text" ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white/60 active:bg-white/5"
                  }`}
                >
                  <FileText className="w-4 h-4" /> TEXT
                </button>
                <button 
                  onClick={() => { setMode("url"); setInput(""); setError(null); }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-black rounded-xl transition-all min-h-[44px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                    mode === "url" ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white/60 active:bg-white/5"
                  }`}
                >
                  <LinkIcon className="w-4 h-4" /> URL
                </button>
              </div>
              </div>

              {/* Input Field */}
              <div className="mt-5 sm:mt-6">
                <p className="text-[10px] font-black uppercase tracking-[0.18em] text-white/30 mb-2">
                  {mode === "text" ? "Your text or claim" : "Article URL to verify"}
                </p>
                <div className="relative">
                {mode === "text" ? (
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Paste a claim, headline, or full article here to analyse…"
                    className="w-full h-48 sm:h-64 bg-black/40 border border-white/10 rounded-2xl p-4 sm:p-6 text-white placeholder:text-white/20 resize-none outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50 transition-all font-medium text-base sm:text-lg leading-relaxed shadow-inner"
                  />
                ) : (
                  <div className="relative">
                    <input
                      type="url"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="https://example.com/article-to-verify"
                      className="w-full h-14 sm:h-auto bg-black/40 border border-white/10 rounded-2xl p-4 sm:p-6 text-white placeholder:text-white/20 outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50 transition-all font-medium text-base sm:text-lg shadow-inner"
                    />
                  </div>
                )}
                
                {mode === "text" && (
                  <div className="absolute bottom-4 right-4 sm:right-6 flex items-center gap-4 bg-black/40 px-2 py-1 rounded-md backdrop-blur-md">
                    <span className={`text-[10px] font-black uppercase tracking-widest ${input.length > 4500 ? 'text-rose-400' : 'text-emerald-400/60'}`}>
                      {input.length} / 5000
                    </span>
                  </div>
                )}
                </div>
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {error && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-4 sm:mt-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-start sm:items-center gap-3 text-rose-400 text-sm font-bold"
                  >
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 sm:mt-0" />
                    <span className="leading-tight">{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit Button */}
              <button
                onClick={handleAnalyze}
                disabled={loading || !input.trim()}
                className="w-full mt-6 sm:mt-8 py-4 sm:py-5 min-h-[56px] rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white font-black text-lg sm:text-xl hover:shadow-[0_0_40px_rgba(16,185,129,0.3)] hover:-translate-y-1 active:translate-y-0 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex flex-row items-center justify-center gap-2 sm:gap-3 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 sm:w-6 sm:h-6 animate-spin shrink-0" /> 
                    <span className="whitespace-nowrap">VERIFYING SIGNALS...</span>
                  </>
                ) : (
                  <>
                    <span className="whitespace-nowrap">ANALYSE CONTENT</span>
                    <Send className="w-4 h-4 sm:w-5 sm:h-5 shrink-0 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                  </>
                )}
              </button>
            </motion.div>

            <div className="flex items-center gap-4 sm:gap-6 px-2 sm:px-4">
              <div className="flex -space-x-3">
                {[1,2,3,4].map(i => (
                  <div key={i} className="w-6 h-6 sm:w-8 sm:h-8 rounded-full border-2 border-black bg-white/10" />
                ))}
              </div>
              <p className="text-[10px] sm:text-xs text-white/30 font-bold uppercase tracking-wide">10k+ verified sessions</p>
            </div>
          </div>

          {/* Results Side */}
          <div className="lg:col-span-12 xl:col-span-7 min-h-[400px] sm:min-h-[600px] flex flex-col">
            {result ? (
              <div className="space-y-4 sm:space-y-6 flex-1">
                <div className="flex items-center justify-between px-2">
                  <h3 className="text-xs sm:text-sm font-black text-emerald-500/50 uppercase tracking-[0.2em] sm:tracking-[0.3em]">Analysis Complete</h3>
                  <button 
                    onClick={() => { setResult(null); setInput(""); setError(null); }}
                    className="flex items-center gap-2 text-[10px] font-black text-emerald-400 hover:text-emerald-300 transition-colors uppercase tracking-widest min-h-[44px] px-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 rounded-lg"
                  >
                    <RefreshCcw className="w-3 h-3" /> New Session
                  </button>
                </div>
                <ResultDisplay result={result} />
              </div>
            ) : (
              <div className="flex-1 glass rounded-[1.5rem] sm:rounded-[2rem] border border-white/5 flex items-center justify-center p-6 sm:p-12 relative group min-h-[400px]">
                <div className="absolute inset-0 bg-emerald-500/[0.02] opacity-0 group-hover:opacity-100 transition-opacity rounded-[1.5rem] sm:rounded-[2rem]" />
                <div className="flex flex-col items-center justify-center text-center w-full max-w-[280px] sm:max-w-sm">
                  <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-6 sm:mb-8 bg-white/[0.02]">
                    <RefreshCcw className="w-6 h-6 sm:w-8 sm:h-8 text-white/10 animate-spin-slow" />
                  </div>
                  <h3 className="text-xl sm:text-2xl font-black text-white/60 mb-3 sm:mb-4 tracking-tight">System Idle</h3>
                  <p className="text-sm border-t border-b border-white/5 py-4 sm:py-0 sm:border-none sm:text-base text-white/30 font-medium">
                    The Text Verification engine is ready. Input raw content or a link on the left to begin deep-signal analysis.
                  </p>
                  <div className="mt-8 sm:mt-12 grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 w-full">
                    <div className="p-3 sm:p-4 rounded-xl sm:rounded-2xl border border-white/5 bg-white/[0.01] text-[10px] sm:text-xs font-bold text-white/20 uppercase tracking-widest flex items-center justify-center text-center whitespace-nowrap">
                      AI Detection
                    </div>
                    <div className="p-3 sm:p-4 rounded-xl sm:rounded-2xl border border-white/5 bg-white/[0.01] text-[10px] sm:text-xs font-bold text-white/20 uppercase tracking-widest flex items-center justify-center text-center whitespace-nowrap">
                      Bias Scoring
                    </div>
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
