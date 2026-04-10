"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp, Cpu, Activity, Zap } from "lucide-react";
import { AnalysisResult } from "@/lib/types";

interface DebugPanelProps {
  data: AnalysisResult;
}

export function DebugPanel({ data }: DebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const stylometry = data.features?.stylometry;
  const perplexity = data.features?.perplexity ?? 0;
  const model = data.metadata?.model ?? "unknown";
  const latency = data.metadata?.latency_ms ?? 0;

  return (
    <div className="mt-8">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors group"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
            <Cpu className="w-4 h-4" />
          </div>
          <span className="text-sm font-bold text-white/80">Advanced Metrics / Debug Panel</span>
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4 text-white/40" /> : <ChevronDown className="w-4 h-4 text-white/40" />}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-6 grid grid-cols-2 gap-4 bg-black/40 border-x border-b border-white/10 rounded-b-2xl">
              {/* Stylometry */}
              <div className="space-y-4">
                <h4 className="flex items-center gap-2 text-xs font-black text-rose-400/80 uppercase tracking-widest">
                  <Activity className="w-3 h-3" /> Stylometry
                </h4>
                <div className="space-y-3">
                  <MetricRow label="Sentence Variance" value={(stylometry?.sentence_length_variance ?? 0).toFixed(2)} />
                  <MetricRow label="Repetition Score" value={(stylometry?.repetition_score ?? 0).toFixed(4)} />
                  <MetricRow label="Lexical Diversity" value={(stylometry?.lexical_diversity ?? 0).toFixed(4)} />
                </div>
              </div>

              {/* System Performance */}
              <div className="space-y-4">
                <h4 className="flex items-center gap-2 text-xs font-black text-emerald-400/80 uppercase tracking-widest">
                  <Zap className="w-3 h-3" /> System Performance
                </h4>
                <div className="space-y-3">
                  <MetricRow label="Perplexity" value={perplexity.toFixed(1)} />
                  <MetricRow label="Model" value={model} />
                  <MetricRow label="Latency" value={`${latency}ms`} />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-white/5 last:border-0">
      <span className="text-[10px] text-white/40">{label}</span>
      <span className="text-[11px] font-mono text-white/80">{value}</span>
    </div>
  );
}
