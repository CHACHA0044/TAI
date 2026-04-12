"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Link as LinkIcon, RefreshCcw, Send, AlertCircle } from "lucide-react";
import { SectionWrapper } from "@/components/section-wrapper";
import { ResultDisplay } from "@/components/result-display";
import { AnalysisResult } from "@/lib/types";
import { analyzeText, analyzeURL } from "@/lib/api";
import { Button } from "@/components/ui/button";

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
    <div className="min-h-screen pt-32 pb-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-emerald-500/5 blur-[120px] rounded-full -z-10 opacity-60" />

      <SectionWrapper className="container w-full max-w-7xl mx-auto px-6">
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
            className="text-5xl md:text-7xl font-black mb-8 tracking-tighter"
          >
            Tactical <span className="text-emerald-400">Text</span> Analysis
          </motion.h1>
          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-white/40 text-lg md:text-xl max-w-3xl mx-auto font-medium"
          >
            Expose misinformation hubs, detect AI-generated propaganda, and verify 
            source credibility using multi-layered NLP forensics.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Input Side */}
          <div className="lg:col-span-12 xl:col-span-5 space-y-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white/[0.02] rounded-[2.5rem] p-10 border border-white/5 shadow-2xl relative backdrop-blur-md"
            >
              {/* Panel heading */}
              <div className="mb-8">
                <h2 className="text-lg font-black text-white/80 tracking-tight">Intelligence Intake</h2>
                <p className="text-sm text-white/30 mt-2 font-medium">Select your input vectors and provide the raw intelligence.</p>
              </div>

              {/* Mode Toggle */}
              <div className="mb-6">
                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-500/50 mb-3">Vector Mode</p>
                <div className="flex bg-black/40 rounded-2xl p-1.5 border border-white/5">
                  <button 
                    onClick={() => { setMode("text"); setInput(""); setError(null); }}
                    className={`flex-1 flex items-center justify-center gap-2 py-3 text-xs font-black rounded-xl transition-all ${
                      mode === "text" ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white/60"
                    }`}
                  >
                    <FileText className="w-4 h-4" /> RAW TEXT
                  </button>
                  <button 
                    onClick={() => { setMode("url"); setInput(""); setError(null); }}
                    className={`flex-1 flex items-center justify-center gap-2 py-3 text-xs font-black rounded-xl transition-all ${
                      mode === "url" ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white/60"
                    }`}
                  >
                    <LinkIcon className="w-4 h-4" /> SOURCE URL
                  </button>
                </div>
              </div>

              {/* Input Field */}
              <div className="space-y-4">
                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-500/50">
                  {mode === "text" ? "Payload Content" : "Target URL"}
                </p>
                <div className="relative">
                  {mode === "text" ? (
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Paste claim or article for deep-signal analysis..."
                      className="w-full h-64 bg-black/60 border border-white/5 rounded-3xl p-6 text-white placeholder:text-white/10 resize-none outline-none focus:border-emerald-500/30 transition-all font-medium text-lg leading-relaxed"
                    />
                  ) : (
                    <input
                      type="url"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="https://news-source.com/article-path"
                      className="w-full bg-black/60 border border-white/5 rounded-2xl p-6 text-white placeholder:text-white/10 outline-none focus:border-emerald-500/30 transition-all font-medium text-lg"
                    />
                  )}
                  
                  {mode === "text" && (
                    <div className="absolute bottom-6 right-6 px-3 py-1 bg-black/80 rounded-full border border-white/5">
                      <span className={`text-[10px] font-black uppercase tracking-widest ${input.length > 4500 ? 'text-rose-400' : 'text-emerald-500/40'}`}>
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
                    className="mt-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center gap-3 text-rose-400 text-sm font-bold"
                  >
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit Button */}
              <div className="mt-10">
                <Button
                  variant={loading ? "loading" : "primary"}
                  onClick={handleAnalyze}
                  className="w-full h-16 text-lg"
                  disabled={!input.trim()}
                >
                  {loading ? "INITIALIZING FORENSICS" : "DEPLOY ANALYSIS"}
                  {!loading && <Send className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />}
                </Button>
              </div>
            </motion.div>
          </div>

          {/* Results Side */}
          <div className="lg:col-span-12 xl:col-span-7 flex flex-col">
            {result ? (
              <div className="space-y-6 flex-1">
                <div className="flex items-center justify-between px-4">
                  <h3 className="text-[10px] font-black text-emerald-500/50 uppercase tracking-[0.4em]">Forensic Output Complete</h3>
                  <Button 
                    variant="ghost" 
                    onClick={() => { setResult(null); setInput(""); setError(null); }}
                    className="h-10 text-[10px] font-black gap-2"
                  >
                    <RefreshCcw className="w-3 h-3" /> FLUSH SESSION
                  </Button>
                </div>
                <ResultDisplay result={result} />
              </div>
            ) : (
              <div className="flex-1 rounded-[2.5rem] border border-white/5 bg-white/[0.01] backdrop-blur-sm flex items-center justify-center p-12 min-h-[500px]">
                <div className="flex flex-col items-center justify-center text-center max-w-sm">
                  <div className="w-24 h-24 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-8 bg-white/[0.02]">
                    <RefreshCcw className="w-8 h-8 text-white/10 animate-spin-slow" />
                  </div>
                  <h3 className="text-2xl font-black text-white/60 mb-4 tracking-tight">Kernel Status: IDLE</h3>
                  <p className="text-base text-white/30 font-medium leading-relaxed">
                    Awaiting intelligence intake. Input raw text or a validated URL 
                    on the left to initiate multi-modal signal processing.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </SectionWrapper>
    </div>
  );
}
