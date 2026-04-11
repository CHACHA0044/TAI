"use client";

import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  Info,
  ExternalLink,
  Target,
  BrainCircuit,
  Scale,
  Globe,
  Cpu,
  Mic,
  Newspaper,
  FileText,
} from "lucide-react";
import { AnalysisResult } from "@/lib/types";
import { ScoreBar } from "./score-bar";
import { DebugPanel } from "./debug-panel";

interface ResultDisplayProps {
  result: AnalysisResult;
}

export function ResultDisplay({ result }: ResultDisplayProps) {
  // Determine verdict based on truth score
  const getVerdict = () => {
    // If we have a specific category, use that first for clearer results
    if (result.category) {
      switch (result.category) {
        case "AI_GENERATED":
          return { label: "AI Generated", color: "text-rose-400", icon: ShieldAlert, bg: "bg-rose-500/10", border: "border-rose-500/30" };
        case "DEEPFAKE":
          return { label: "Deepfake Found", color: "text-amber-400", icon: ShieldQuestion, bg: "bg-amber-500/10", border: "border-amber-500/30" };
        case "REAL":
          return { label: "Organic Photo", color: "text-emerald-400", icon: ShieldCheck, bg: "bg-emerald-500/10", border: "border-emerald-500/30" };
      }
    }

    if (result.truth_score > 0.7) return { label: "Authentic", color: "text-emerald-400", icon: ShieldCheck, bg: "bg-emerald-500/10", border: "border-emerald-500/30" };
    if (result.truth_score > 0.4) return { label: "Unverified", color: "text-amber-400", icon: ShieldQuestion, bg: "bg-amber-500/10", border: "border-amber-500/30" };
    return { label: "Evidence of Fake", color: "text-rose-400", icon: ShieldAlert, bg: "bg-rose-500/10", border: "border-rose-500/30" };
  };

  const verdict = getVerdict();
  const Icon = verdict.icon;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-6"
    >
      {/* Top Verdict Card */}
      <div className={`p-6 rounded-3xl ${verdict.bg} border ${verdict.border} backdrop-blur-xl relative overflow-hidden group`}>
        <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
          <Icon className="w-24 h-24" />
        </div>
        
        <div className="relative z-10 flex items-center gap-5">
          <div className={`p-4 rounded-2xl ${verdict.bg} border ${verdict.border} shadow-lg`}>
            <Icon className={`w-8 h-8 ${verdict.color}`} />
          </div>
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40 mb-1">Final Engine Verdict</p>
            <h2 className={`text-3xl font-black ${verdict.color} tracking-tight`}>{verdict.label}</h2>
          </div>
          <div className="ml-auto text-right">
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40 mb-1">Confidence</p>
            <p className="text-xl font-mono font-bold text-white">{Math.round(result.confidence_score * 100)}%</p>
          </div>
        </div>
      </div>

      {/* Source URL banner — shown when result originated from URL extraction */}
      {result.source && (result.source.startsWith("http://") || result.source.startsWith("https://")) && (
        <div className="flex items-center gap-3 px-5 py-3 rounded-2xl bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-bold">
          <Globe className="w-4 h-4 flex-shrink-0" />
          <span className="text-white/50 uppercase tracking-widest text-[10px] flex-shrink-0">Analyzed URL</span>
          <a
            href={result.source}
            target="_blank"
            rel="noopener noreferrer"
            className="truncate hover:text-sky-300 transition-colors flex items-center gap-1"
            title={result.source}
          >
            {result.source}
            <ExternalLink className="w-3 h-3 flex-shrink-0 ml-1" />
          </a>
        </div>
      )}

      {/* Core Score Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {result.category ? (
          <div className="glass rounded-2xl border border-white/10 p-5 group hover:border-white/20 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 rounded-lg bg-white/5">
                <Globe className="w-4 h-4 text-emerald-400" />
              </div>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${result.source && result.source !== "Unknown" ? "bg-rose-500/20 text-rose-400" : "bg-white/5 text-white/30"}`}>
                {result.source && result.source !== "Unknown" ? "IDENTIFIED" : "UNKNOWN"}
              </span>
            </div>
            <p className="text-xs font-bold text-white/80 mb-1">Origin / Source</p>
            <p className="text-lg font-black text-white/90 truncate">{result.source && result.source !== "Unknown" ? result.source : "No match found"}</p>
            <p className="text-[10px] text-white/30 leading-tight mt-1">
              {result.source && result.source !== "Unknown"
                ? "Source identified via filename markers or embedded metadata."
                : "No known AI generator signature found in file metadata."}
            </p>
          </div>
        ) : (
          <ScoreCard 
            label="Truth Score" 
            score={result.truth_score} 
            icon={<Target className="w-4 h-4 text-emerald-400" />}
            description="Factual accuracy vs. verified sources."
          />
        )}
        
        <ScoreCard 
          label={result.category ? "AI Artifact Score" : "AI Likelihood"}
          score={result.ai_generated_score} 
          icon={<BrainCircuit className="w-4 h-4 text-rose-400" />}
          description={result.category ? "Neural network probability that content is synthetic." : "Likelihood content was machine-generated."}
          invertColor
        />

        {!result.category && (
          <ScoreCard 
            label="Bias Level" 
            score={result.bias_score} 
            icon={<Scale className="w-4 h-4 text-amber-400" />}
            description="Detected slant, propaganda, or loaded framing."
            invertColor
          />
        )}

        {result.category && (
          <ScoreCard 
            label="Pixel Integrity (ELA)" 
            score={result.credibility_score} 
            icon={<Cpu className="w-4 h-4 text-amber-400" />}
            description="How consistent the compression noise is. Low = edited or synthetic."
          />
        )}
      </div>

      {/* Forensic Verdict Summary — only for image/video (category present) */}
      {result.category && result.signals && result.signals.length > 0 && (
        <div className="glass rounded-3xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">Forensic Verdict Summary</h3>
            <span className="ml-auto text-[10px] text-white/30">{result.signals.length} layers checked</span>
          </div>
          <div className="grid gap-2">
            {result.signals.map((signal, i) => {
              const pct = Math.round(signal.confidence * 100);
              return (
                <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border ${signal.verified ? "bg-emerald-500/5 border-emerald-500/20" : "bg-rose-500/5 border-rose-500/20"}`}>
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${signal.verified ? "bg-emerald-500" : "bg-rose-500"}`} />
                  <span className="text-xs font-semibold text-white/80 flex-1">{signal.source}</span>
                  <span className="text-[10px] font-mono text-white/40">{pct}%</span>
                  <span className={`text-[10px] font-black uppercase tracking-wide ${signal.verified ? "text-emerald-400" : "text-rose-400"}`}>
                    {signal.verified ? "PASS" : "FAIL"}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-white/25 mt-3">
            Each layer is an independent forensic check. 2 or more FAILs = high confidence of manipulation or synthesis.
          </p>
        </div>
      )}

      {/* Explanation Section */}
      <div className="glass rounded-3xl border border-white/10 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">
            {result.category ? "Forensic Analysis Report" : "AI Reasoning"}
          </h3>
        </div>
        <p className="text-white/70 text-sm leading-relaxed mb-6 italic">
          "{result.explanation}"
        </p>

        {/* Verification Signals — only shown for text results (no category) */}
        {!result.category && result.signals && result.signals.length > 0 && (
          <div className="space-y-4 pt-4 border-t border-white/5">
            <h4 className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em]">Verification Signals</h4>
            <div className="grid gap-3">
              {result.signals.map((signal, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
                  <span className="text-xs font-medium text-white/70">{signal.source}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16">
                      <ScoreBar label="" score={signal.confidence} showPercentage={false} color={signal.verified ? "bg-emerald-500" : "bg-rose-500"} />
                    </div>
                    <span className="text-[10px] font-mono text-white/40 w-8 text-right">{Math.round(signal.confidence * 100)}%</span>
                    <span className={`text-[10px] font-bold w-16 text-right ${signal.verified ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {signal.verified ? 'VERIFIED' : 'FLAGGED'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Raw Metadata Section */}
        {result.metadata?.raw_metadata && Object.keys(result.metadata.raw_metadata).length > 0 && (
          <div className="mt-6 pt-6 border-t border-white/5">
            <h4 className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em] mb-4 text-center">Extraction: Source Metadata</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {Object.entries(result.metadata.raw_metadata).map(([key, value]) => (
                <div key={key} className="p-3 rounded-xl bg-black/30 border border-white/5 flex flex-col gap-1">
                  <span className="text-[10px] uppercase font-bold text-white/20 tracking-wider">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <span className="text-xs font-mono text-white/80 truncate" title={String(value)}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Audio Forensics Score */}
      {result.audio_score !== undefined && result.audio_score !== null && (
        <div className="glass rounded-3xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Mic className="w-4 h-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">Audio Forensics</h3>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <ScoreBar label="Audio Authenticity" score={result.audio_score} showPercentage />
            </div>
            <div className="text-right">
              <p className="text-2xl font-black font-mono text-white">{Math.round(result.audio_score * 100)}%</p>
              <p className={`text-xs font-bold mt-1 ${result.audio_score >= 0.5 ? "text-emerald-400" : "text-rose-400"}`}>
                {result.audio_score >= 0.7 ? "AUTHENTIC" : result.audio_score >= 0.5 ? "UNCERTAIN" : "SYNTHETIC"}
              </p>
            </div>
          </div>
          <p className="text-[10px] text-white/30 mt-3">
            Spectral and temporal audio features analyzed for synthetic speech indicators (TTS, voice cloning).
          </p>
        </div>
      )}

      {/* News Consistency Score */}
      {result.news_consistency_score !== undefined && result.news_consistency_score !== null && (
        <div className="glass rounded-3xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Newspaper className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">News Consistency</h3>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <ScoreBar
                label="Consistency with Known Reporting"
                score={result.news_consistency_score}
                showPercentage
                color={result.news_consistency_score >= 0.5 ? "bg-sky-500" : "bg-rose-500"}
              />
            </div>
            <div className="text-right">
              <p className="text-2xl font-black font-mono text-white">{Math.round(result.news_consistency_score * 100)}%</p>
              <p className={`text-xs font-bold mt-1 ${result.news_consistency_score >= 0.6 ? "text-emerald-400" : result.news_consistency_score >= 0.4 ? "text-amber-400" : "text-rose-400"}`}>
                {result.news_consistency_score >= 0.6 ? "CONSISTENT" : result.news_consistency_score >= 0.4 ? "UNCERTAIN" : "INCONSISTENT"}
              </p>
            </div>
          </div>
          <p className="text-[10px] text-white/30 mt-3">
            Semantic similarity cross-referenced against real-world news headlines to detect unsupported claims.
          </p>
        </div>
      )}

      {/* OCR Extracted Text */}
      {result.ocr_text && result.ocr_text.trim().length > 0 && (
        <div className="glass rounded-3xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">OCR Extracted Text</h3>
            <span className="ml-auto text-[10px] text-white/30">{result.ocr_text.trim().length} chars</span>
          </div>
          <pre className="text-xs text-white/70 bg-black/40 rounded-xl p-4 border border-white/5 whitespace-pre-wrap break-words max-h-48 overflow-y-auto font-mono leading-relaxed">
            {result.ocr_text.trim()}
          </pre>
          <p className="text-[10px] text-white/30 mt-3">
            Text extracted from the image via Tesseract OCR and fed into the truthfulness pipeline.
          </p>
        </div>
      )}

      {/* Advanced Debug Section */}
      <DebugPanel data={result} />
    </motion.div>
  );
}

function ScoreCard({ label, score, icon, description, invertColor }: { label: string; score: number; icon: React.ReactNode; description: string; invertColor?: boolean }) {
  // For AI/Bias scores, high = bad (red). For Truth/ELA/credibility, high = good (green).
  const getColor = () => {
    const v = invertColor ? 1 - score : score;
    if (v > 0.6) return "bg-emerald-500";
    if (v > 0.35) return "bg-amber-500";
    return "bg-rose-500";
  };
  const getTextColor = () => {
    const v = invertColor ? 1 - score : score;
    if (v > 0.6) return "text-emerald-400";
    if (v > 0.35) return "text-amber-400";
    return "text-rose-400";
  };
  return (
    <div className="glass rounded-2xl border border-white/10 p-5 group hover:border-white/20 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="p-2 rounded-lg bg-white/5">
          {icon}
        </div>
        <span className={`text-2xl font-black font-mono ${getTextColor()}`}>{Math.round(score * 100)}%</span>
      </div>
      <p className="text-xs font-bold text-white/80 mb-1">{label}</p>
      <p className="text-[10px] text-white/30 leading-tight">{description}</p>
      <div className="mt-4">
        <ScoreBar label="" score={score} showPercentage={false} color={getColor()} />
      </div>
    </div>
  );
}
