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
  const rawMeta = (data.metadata?.raw_metadata ?? {}) as Record<string, unknown>;
  const featureMetrics = (rawMeta.feature_metrics ?? {}) as Record<string, unknown>;
  const rawClassifierOutputs = (rawMeta.raw_classifier_outputs ?? {}) as Record<string, unknown>;
  const aggregationRule = String(rawMeta.aggregation_rule ?? data.triggered_rule ?? "N/A");
  const debug = (data.debug ?? {}) as Record<string, unknown>;
  const thresholdValues = (debug.threshold_values_used ?? {}) as Record<string, unknown>;
  const detectorConfidences = (debug.detector_confidences ?? {}) as Record<string, unknown>;
  const sarcasmRuleHits = Array.isArray(debug.sarcasm_rule_hits) ? debug.sarcasm_rule_hits : [];
  const biasRuleHits = Array.isArray(debug.bias_rule_hits) ? debug.bias_rule_hits : [];
  const manipulationRuleHits = Array.isArray(debug.manipulation_rule_hits) ? debug.manipulation_rule_hits : [];

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
            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-black/40 border-x border-b border-white/10 rounded-b-2xl">
              {/* Primary Analysis Metrics */}
              <div className="space-y-4">
                <h4 className="flex items-center gap-2 text-xs font-black text-rose-400/80 uppercase tracking-widest">
                  <Activity className="w-3 h-3" /> {data.category ? "Forensic Layers" : "Stylometry"}
                </h4>
                <div className="space-y-3">
                  {data.category ? (
                    <>
                      <MetricRow label="Neural Signal Strength" value={(stylometry?.sentence_length_variance ?? 0).toFixed(4)} />
                      <MetricRow label="ELA Dispersion" value={perplexity.toFixed(4)} />
                      <MetricRow label="Metadata Count" value={stylometry?.repetition_score ?? 0} />
                      <MetricRow label="Scan Confidence" value={(stylometry?.lexical_diversity ?? 0).toFixed(4)} />
                    </>
                  ) : (
                    <>
                      <MetricRow label="Sentence Variance" value={(stylometry?.sentence_length_variance ?? 0).toFixed(2)} />
                      <MetricRow label="Burstiness" value={Number(stylometry?.burstiness ?? featureMetrics.burstiness ?? 0).toFixed(4)} />
                      <MetricRow label="Repetition Score" value={(stylometry?.repetition_score ?? 0).toFixed(4)} />
                      <MetricRow label="Lexical Diversity" value={(stylometry?.lexical_diversity ?? 0).toFixed(4)} />
                    </>
                  )}
                </div>
              </div>

              {/* System Performance */}
              <div className="space-y-4">
                <h4 className="flex items-center gap-2 text-xs font-black text-emerald-400/80 uppercase tracking-widest">
                  <Zap className="w-3 h-3" /> System Performance
                </h4>
                <div className="space-y-3">
                  {!data.category && <MetricRow label="Perplexity" value={perplexity.toFixed(1)} />}
                  {!data.category && <MetricRow label="Aggregation Rule" value={aggregationRule} />}
                  {!data.category && <MetricRow label="Text Type Detected" value={String(debug.text_type_detected ?? "N/A")} />}
                  {!data.category && <MetricRow label="Verifiability Result" value={String(debug.verifiability_result ?? "N/A")} />}
                  {!data.category && <MetricRow label="Trust Agent Confidence" value={String(debug.trust_agent_confidence ?? "N/A")} />}
                  {!data.category && <MetricRow label="Retrieval Support Score" value={String(debug.retrieval_support_score ?? "N/A")} />}
                  {!data.category && <MetricRow label="Retrieval Contradiction Score" value={String(debug.retrieval_contradiction_score ?? "N/A")} />}
                  {!data.category && <MetricRow label="Thresholds Exposed" value={Object.keys(thresholdValues).length} />}
                  {!data.category && <MetricRow label="Detector Confidences" value={Object.keys(detectorConfidences).length} />}
                  {!data.category && <MetricRow label="Sarcasm Rule Hits" value={sarcasmRuleHits.length} />}
                  {!data.category && <MetricRow label="Bias Rule Hits" value={biasRuleHits.length} />}
                  {!data.category && <MetricRow label="Manipulation Rule Hits" value={manipulationRuleHits.length} />}
                  {!data.category && <MetricRow label="Feature Burstiness" value={String(featureMetrics.burstiness ?? "0")} />}
                  {!data.category && <MetricRow label="Feature Sentence Variance" value={String(featureMetrics.sentence_variance ?? stylometry?.sentence_length_variance ?? 0)} />}
                  {!data.category && <MetricRow label="Feature Lexical Diversity" value={String(featureMetrics.lexical_diversity ?? stylometry?.lexical_diversity ?? 0)} />}
                  {!data.category && <MetricRow label="Raw Truth Output" value={String(rawClassifierOutputs.truth_model_raw ?? data.truth_score)} />}
                  {!data.category && <MetricRow label="Raw AI Output" value={String(rawClassifierOutputs.ai_model_raw ?? data.ai_generated_score)} />}
                  {!data.category && <MetricRow label="Raw Bias Output" value={String(rawClassifierOutputs.bias_model_raw ?? data.bias_score)} />}
                  <MetricRow label="Active Model" value={model} />
                  <MetricRow label="Processing Latency" value={`${latency}ms`} />
                  <MetricRow label="Timestamp" value={new Date(data.metadata.timestamp).toLocaleTimeString()} />
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
