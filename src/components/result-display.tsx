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
  Scale
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
    if (result.truth_score > 0.7) return { label: "Authentic", color: "text-emerald-400", icon: ShieldCheck, bg: "bg-emerald-500/10", border: "border-emerald-500/30" };
    if (result.truth_score > 0.4) return { label: "Unverified", color: "text-amber-400", icon: ShieldQuestion, bg: "bg-amber-500/10", border: "border-amber-500/30" };
    return { label: "Likely Fake", color: "text-rose-400", icon: ShieldAlert, bg: "bg-rose-500/10", border: "border-rose-500/30" };
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

      {/* Core Score Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ScoreCard 
          label="Truth Score" 
          score={result.truth_score} 
          icon={<Target className="w-4 h-4 text-emerald-400" />}
          description="Factual accuracy probability."
        />
        <ScoreCard 
          label="AI Detection" 
          score={result.ai_generated_score} 
          icon={<BrainCircuit className="w-4 h-4 text-rose-400" />}
          description="Likelihood of machine origin."
        />
        <ScoreCard 
          label="Bias Analysis" 
          score={result.bias_score} 
          icon={<Scale className="w-4 h-4 text-amber-400" />}
          description="Propaganda & slant detection."
        />
      </div>

      {/* Explanation Section */}
      <div className="glass rounded-3xl border border-white/10 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">AI Reasoning</h3>
        </div>
        <p className="text-white/70 text-sm leading-relaxed mb-6 italic">
          "{result.explanation}"
        </p>

        {/* Verification Signals */}
        {result.signals && result.signals.length > 0 && (
          <div className="space-y-4 pt-4 border-t border-white/5">
            <h4 className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em]">External Corroboration</h4>
            <div className="grid gap-3">
              {result.signals.map((signal, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
                  <span className="text-xs font-medium text-white/70">{signal.source}</span>
                  <div className="flex items-center gap-3">
                    <ScoreBar label="" score={signal.confidence} showPercentage={false} color={signal.verified ? "bg-emerald-500" : "bg-rose-500"} />
                    <span className={`text-[10px] font-bold ${signal.verified ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {signal.verified ? 'VERIFIED' : 'REFUTED'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Advanced Debug Section */}
      <DebugPanel data={result} />
    </motion.div>
  );
}

function ScoreCard({ label, score, icon, description }: { label: string; score: number; icon: React.ReactNode; description: string }) {
  return (
    <div className="glass rounded-2xl border border-white/10 p-5 group hover:border-white/20 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="p-2 rounded-lg bg-white/5">
          {icon}
        </div>
        <span className="text-2xl font-black font-mono text-white">{Math.round(score * 100)}%</span>
      </div>
      <p className="text-xs font-bold text-white/80 mb-1">{label}</p>
      <p className="text-[10px] text-white/30 leading-tight">{description}</p>
      <div className="mt-4">
        <ScoreBar label="" score={score} showPercentage={false} />
      </div>
    </div>
  );
}
