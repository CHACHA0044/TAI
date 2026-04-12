"use client";

import { ReactElement, useMemo, useState } from "react";
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
  Mic,
  Newspaper,
  FileText,
  CircleDot,
} from "lucide-react";
import { AnalysisResult } from "@/lib/types";
import { ScoreBar } from "./score-bar";
import { DebugPanel } from "./debug-panel";
import { composeVerdict } from "@/lib/verdict-composer";
import { MetricCardData, MetricKey } from "./result/metric-types";
import { ScoreCardGrid } from "./result/score-card-grid";
import { MetricModal } from "./result/metric-modal";

const METRIC_VISIBILITY_THRESHOLD = 0.18;

interface ResultDisplayProps {
  result: AnalysisResult;
}

const toneStyles = {
  emerald: {
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    icon: ShieldCheck,
  },
  amber: {
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    icon: ShieldQuestion,
  },
  rose: {
    color: "text-rose-400",
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
    icon: ShieldAlert,
  },
  sky: {
    color: "text-sky-400",
    bg: "bg-sky-500/10",
    border: "border-sky-500/30",
    icon: ShieldQuestion,
  },
  purple: {
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/30",
    icon: ShieldQuestion,
  },
  fuchsia: {
    color: "text-fuchsia-400",
    bg: "bg-fuchsia-500/10",
    border: "border-fuchsia-500/30",
    icon: BrainCircuit,
  },
  slate: {
    color: "text-white",
    bg: "bg-white/5",
    border: "border-white/20",
    icon: CircleDot,
  },
} as const;

function toRatio(value: number | undefined, fallback = 0): number {
  const numeric = Number.isFinite(value) ? Number(value) : fallback;
  if (numeric > 1) return Math.max(0, Math.min(1, numeric / 100));
  return Math.max(0, Math.min(1, numeric));
}

function confidenceLabel(confidence: number) {
  if (confidence >= 0.78) return "High clarity";
  if (confidence >= 0.52) return "Moderate clarity";
  return "Low clarity";
}

function confidenceNarrative(confidence: number) {
  if (confidence >= 0.78) return "Signals are consistent across detectors and evidence checks.";
  if (confidence >= 0.52) return "Some signals align, but there are meaningful competing indicators.";
  return "Evidence and detectors conflict, so interpretation should remain cautious.";
}

function metricBoostByVerdict(verdict: AnalysisResult["primary_verdict"]): Partial<Record<MetricKey, number>> {
  switch (verdict) {
    case "VERIFIED_FACT":
    case "FALSE_FACT":
    case "UNVERIFIED_CLAIM":
      return { truth: 0.6, verifiability: 0.5 };
    case "BIASED_CONTENT":
      return { bias: 0.65, truth: 0.2, verifiability: 0.2 };
    case "MANIPULATIVE_CONTENT":
      return { manipulation: 0.65, bias: 0.2 };
    case "OPINION":
      return { opinion: 0.65, truth: 0.2 };
    case "SATIRE_OR_SARCASM":
      return { sarcasm: 0.65, opinion: 0.2 };
    case "LIKELY_AI_GENERATED":
      return { ai: 0.65 };
    default:
      return {};
  }
}

export function ResultDisplay({ result }: ResultDisplayProps) {
  const [activeMetric, setActiveMetric] = useState<MetricCardData | null>(null);

  const truth = result.dimensions ? toRatio(result.dimensions.truth_score) : toRatio(result.truth_score);
  const verifiability = result.dimensions?.verifiability !== undefined ? toRatio(result.dimensions.verifiability) : 0.5;
  const aiLikelihood = result.dimensions ? toRatio(result.dimensions.ai_likelihood) : toRatio(result.ai_generated_score);
  const bias = result.dimensions ? toRatio(result.dimensions.bias_score) : toRatio(result.bias_score);
  const manipulation = toRatio(result.dimensions?.manipulation_score);
  const opinion = toRatio(result.dimensions?.opinion_score);
  const sarcasm = toRatio(result.dimensions?.sarcasm_score);
  const confidence = toRatio(result.confidence ?? result.confidence_score);

  const composedVerdict = useMemo(
    () =>
      composeVerdict({
        primaryVerdict: result.primary_verdict,
        truth,
        verifiability,
        aiLikelihood,
        bias,
        manipulation,
        opinion,
        sarcasm,
        confidence,
      }),
    [result.primary_verdict, truth, verifiability, aiLikelihood, bias, manipulation, opinion, sarcasm, confidence],
  );

  const verdictStyle = toneStyles[composedVerdict.tone];
  const VerdictIcon = verdictStyle.icon;

  const categoryVerdict = useMemo(() => {
    if (!result.category) return null;

    switch (result.category) {
      case "AI_GENERATED":
        return {
          label: "Likely AI-Generated Image",
          color: "text-rose-400",
          bg: "bg-rose-500/10",
          border: "border-rose-500/30",
          icon: ShieldAlert,
        };
      case "EDITED":
        return {
          label: "Likely Edited / Composited",
          color: "text-amber-400",
          bg: "bg-amber-500/10",
          border: "border-amber-500/30",
          icon: ShieldQuestion,
        };
      case "MIXED":
        return {
          label: "Mixed Authenticity Signals",
          color: "text-purple-400",
          bg: "bg-purple-500/10",
          border: "border-purple-500/30",
          icon: ShieldQuestion,
        };
      case "UNCERTAIN":
        return {
          label: "Uncertain / Needs Manual Review",
          color: "text-sky-400",
          bg: "bg-sky-500/10",
          border: "border-sky-500/30",
          icon: ShieldQuestion,
        };
      case "DEEPFAKE":
        return {
          label: "Deepfake Found",
          color: "text-amber-400",
          bg: "bg-amber-500/10",
          border: "border-amber-500/30",
          icon: ShieldQuestion,
        };
      default:
        return {
          label: "Likely Real Photograph",
          color: "text-emerald-400",
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/30",
          icon: ShieldCheck,
        };
    }
  }, [result.category]);

  const metrics = useMemo(() => {
    const expanded = result.expanded_analysis;
    const debug = result.debug;
    const verdictBoost = metricBoostByVerdict(result.primary_verdict);

    const baseMetrics: MetricCardData[] = [
      {
        key: "truth",
        label: "Truth Score",
        description: "Factual alignment with verifiable evidence.",
        score: truth,
        icon: <Target className="w-4 h-4 text-emerald-400" />,
        relevance: 1.8 + truth + (verdictBoost.truth ?? 0),
        meaning: "Truth score estimates how well core claims align with known facts and evidence.",
        whyAssigned: expanded?.truth_score?.explanation || "Model and retrieval signals were combined to estimate factual correctness.",
        indicators: result.signals?.map((signal) => `${signal.source}: ${Math.round(signal.confidence * 100)}% confidence`) || [],
        interpretation: "Higher values suggest stronger factual grounding. Lower values indicate contradiction or weak support.",
        confidenceContext: confidenceNarrative(confidence),
        evidence: expanded?.truth_score?.evidence,
        sources: expanded?.truth_score?.sources,
      },
      {
        key: "verifiability",
        label: "Verifiability",
        description: "How testable and sourceable the claim is right now.",
        score: verifiability,
        icon: <Info className="w-4 h-4 text-cyan-400" />,
        relevance: 1.7 + verifiability + (verdictBoost.verifiability ?? 0),
        meaning: "Verifiability measures whether enough concrete evidence exists to confidently confirm or refute the claim.",
        whyAssigned: expanded?.verifiability?.explanation || "The engine evaluated whether the claim can be checked against reliable, current sources.",
        indicators: expanded?.verifiability?.indicators || [],
        interpretation: "High values mean the claim is checkable. Low values often indicate missing, ambiguous, or non-testable evidence.",
        confidenceContext: confidenceNarrative(confidence),
      },
      {
        key: "bias",
        label: "Bias Level",
        description: "Loaded framing, ideological slant, or one-sided rhetoric.",
        score: bias,
        invertColor: true,
        icon: <Scale className="w-4 h-4 text-amber-400" />,
        relevance: bias + 0.4 + (verdictBoost.bias ?? 0),
        meaning: "Bias score reflects the intensity of slanted framing and emotionally loaded language.",
        whyAssigned: expanded?.bias_score?.explanation || "The model detected signs of framing that may steer interpretation.",
        indicators: [...(expanded?.bias_score?.indicators || []), ...(debug?.bias_rule_hits || [])],
        interpretation: "Higher bias does not automatically make a claim false, but it can reduce neutrality and trustworthiness.",
        confidenceContext: confidenceNarrative(confidence),
      },
      {
        key: "manipulation",
        label: "Manipulation Score",
        description: "Action pressure, coercion, fear, or urgency cues.",
        score: manipulation,
        invertColor: true,
        icon: <ShieldAlert className="w-4 h-4 text-rose-400" />,
        relevance: manipulation + 0.4 + (verdictBoost.manipulation ?? 0),
        meaning: "Manipulation score indicates persuasive pressure intended to push behavior rather than inform.",
        whyAssigned: expanded?.manipulation_score?.explanation || "The engine found language patterns tied to urgency, coercion, or emotional pressure.",
        indicators: [...(expanded?.manipulation_score?.indicators || []), ...(debug?.manipulation_rule_hits || [])],
        interpretation: "Higher values indicate stronger pressure tactics. It is an influence signal, separate from factuality.",
        confidenceContext: confidenceNarrative(confidence),
      },
      {
        key: "opinion",
        label: "Opinion Score",
        description: "Subjective judgment and preference-oriented language.",
        score: opinion,
        invertColor: true,
        icon: <ShieldQuestion className="w-4 h-4 text-sky-400" />,
        relevance: opinion + 0.35 + (verdictBoost.opinion ?? 0),
        meaning: "Opinion score estimates how much of the text reflects personal judgment rather than objective claims.",
        whyAssigned: expanded?.opinion_score?.explanation || "The classifier detected comparative or subjective wording patterns.",
        indicators: expanded?.opinion_score?.indicators || [],
        interpretation: "High opinion scores suggest value statements that should not be read as strict factual assertions.",
        confidenceContext: confidenceNarrative(confidence),
      },
      {
        key: "sarcasm",
        label: "Sarcasm Score",
        description: "Satirical or ironic framing that can invert literal meaning.",
        score: sarcasm,
        invertColor: true,
        icon: <ShieldQuestion className="w-4 h-4 text-purple-400" />,
        relevance: sarcasm + 0.35 + (verdictBoost.sarcasm ?? 0),
        meaning: "Sarcasm score detects rhetorical irony that can make literal fact-check interpretations unreliable.",
        whyAssigned: result.dimensions?.sarcasm
          ? "Sarcasm cues were detected in sentence framing and wording patterns."
          : "No strong sarcasm cues were dominant in the current text.",
        indicators: result.debug?.sarcasm_rule_hits || [],
        interpretation: "Higher sarcasm means literal reading may be misleading; context and intent matter more.",
        confidenceContext: confidenceNarrative(confidence),
      },
      {
        key: "ai",
        label: "AI Likelihood",
        description: "Style signal for machine-generated writing patterns.",
        score: aiLikelihood,
        invertColor: true,
        icon: <BrainCircuit className="w-4 h-4 text-fuchsia-400" />,
        relevance: aiLikelihood + 0.2 + (verdictBoost.ai ?? 0),
        meaning: "AI likelihood estimates whether writing style resembles machine-generated language patterns.",
        whyAssigned: expanded?.ai_likelihood?.explanation || "Stylometric and classifier signals were combined to estimate synthetic writing probability.",
        indicators: expanded?.ai_likelihood?.indicators || [],
        interpretation: "Use this as provenance context. It should not be treated as direct evidence of factual truth or falsehood.",
        confidenceContext: confidenceNarrative(confidence),
      },
    ];

    const anchors: MetricKey[] = ["truth", "verifiability"];
    const critical = new Set<MetricKey>(anchors);

    switch (result.primary_verdict) {
      case "BIASED_CONTENT":
        critical.add("bias");
        break;
      case "MANIPULATIVE_CONTENT":
        critical.add("manipulation");
        break;
      case "OPINION":
        critical.add("opinion");
        break;
      case "SATIRE_OR_SARCASM":
        critical.add("sarcasm");
        break;
      case "LIKELY_AI_GENERATED":
        critical.add("ai");
        break;
    }

    const visible = baseMetrics.filter((metric) => critical.has(metric.key) || metric.score >= METRIC_VISIBILITY_THRESHOLD);

    const anchorOrder: Record<MetricKey, number> = {
      truth: 0,
      verifiability: 1,
      bias: 2,
      manipulation: 3,
      opinion: 4,
      sarcasm: 5,
      ai: 6,
    };

    return visible.sort((a, b) => {
      const aAnchor = anchors.includes(a.key);
      const bAnchor = anchors.includes(b.key);

      if (aAnchor && bAnchor) return anchorOrder[a.key] - anchorOrder[b.key];
      if (aAnchor) return -1;
      if (bAnchor) return 1;

      if (b.relevance !== a.relevance) return b.relevance - a.relevance;
      return b.score - a.score;
    });
  }, [
    result.expanded_analysis,
    result.debug,
    result.signals,
    result.primary_verdict,
    result.dimensions?.sarcasm,
    truth,
    verifiability,
    bias,
    manipulation,
    opinion,
    sarcasm,
    aiLikelihood,
    confidence,
  ]);

  const imageMetrics = useMemo(() => {
    if (!result.category) return [];
    const authenticitySignals = result.authenticity_signals || {};
    const keyLabelMap: Record<string, { label: string; description: string; icon: ReactElement }> = {
      ela: { label: "ELA Anomaly", description: "Localized recompression/edit signal.", icon: <Target className="w-4 h-4 text-cyan-400" /> },
      texture_consistency: { label: "Texture Irregularity", description: "Over-smooth or repetitive local textures.", icon: <CircleDot className="w-4 h-4 text-amber-400" /> },
      edge_artifacts: { label: "Edge Artifacts", description: "Halos, seams, or irregular edge transitions.", icon: <ShieldAlert className="w-4 h-4 text-rose-400" /> },
      lighting_consistency: { label: "Lighting Inconsistency", description: "Illumination direction mismatch across regions.", icon: <Globe className="w-4 h-4 text-purple-400" /> },
      shadow_mismatch: { label: "Shadow Mismatch", description: "Shadows/luminance do not align naturally.", icon: <ShieldQuestion className="w-4 h-4 text-purple-400" /> },
      noise_pattern_mismatch: { label: "Noise Pattern Mismatch", description: "Sensor noise looks inconsistent.", icon: <Info className="w-4 h-4 text-sky-400" /> },
      compression_anomalies: { label: "Compression Anomaly", description: "Compression profile differs from natural capture.", icon: <ShieldAlert className="w-4 h-4 text-rose-400" /> },
      face_hand_inconsistency: { label: "Face/Hand Anomaly", description: "Potential facial/limb structure artifact.", icon: <ShieldAlert className="w-4 h-4 text-amber-400" /> },
      object_realism: { label: "Object Realism Risk", description: "Potentially implausible object details.", icon: <Scale className="w-4 h-4 text-amber-400" /> },
      metadata_anomalies: { label: "Metadata Anomaly", description: "EXIF/source metadata is missing or suspicious.", icon: <FileText className="w-4 h-4 text-cyan-400" /> },
    };

    return Object.entries(authenticitySignals)
      .map(([key, value]) => {
        const score = toRatio(value?.score);
        const mapped = keyLabelMap[key] || {
          label: key.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase()),
          description: "Forensic authenticity signal.",
          icon: <Info className="w-4 h-4 text-white/70" />,
        };
        return {
          key: key as MetricKey,
          label: mapped.label,
          description: mapped.description,
          score,
          invertColor: true,
          icon: mapped.icon,
          relevance: score,
          meaning: mapped.description,
          whyAssigned: value?.explanation || "Signal contributed to authenticity assessment.",
          indicators: [value?.bucket ? `Severity bucket: ${value.bucket}` : "", `Signal score: ${Math.round(score * 100)}%`].filter(Boolean),
          interpretation: "Higher values indicate stronger forensic concern for this specific signal.",
          confidenceContext: confidenceNarrative(confidence),
        } as MetricCardData;
      })
      .filter((metric) => metric.score >= METRIC_VISIBILITY_THRESHOLD || metric.score >= 0.5)
      .sort((a, b) => b.relevance - a.relevance);
  }, [result.category, result.authenticity_signals, confidence]);

  const primarySignals = metrics.slice(0, 3);
  const secondarySignals = metrics.slice(3);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-6"
    >
      <div className={`p-6 rounded-3xl ${result.category ? categoryVerdict?.bg : verdictStyle.bg} border ${result.category ? categoryVerdict?.border : verdictStyle.border} backdrop-blur-xl relative overflow-hidden group`}>
        <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
          {(() => { const CvIcon = categoryVerdict?.icon; return result.category && CvIcon ? <CvIcon className="w-24 h-24" /> : <VerdictIcon className="w-24 h-24" />; })()}
        </div>

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-start gap-6">
          <div className="flex items-start gap-4 flex-1 min-w-0">
            <div className={`p-4 rounded-2xl ${result.category ? categoryVerdict?.bg : verdictStyle.bg} border ${result.category ? categoryVerdict?.border : verdictStyle.border} shadow-lg`}>
              {(() => { const CvIcon = categoryVerdict?.icon; return result.category && CvIcon ? <CvIcon className={`w-8 h-8 ${categoryVerdict!.color}`} /> : <VerdictIcon className={`w-8 h-8 ${verdictStyle.color}`} />; })()}
            </div>

            <div className="min-w-0">
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40 mb-1">Forensic verdict summary</p>
              <h2 className={`text-3xl font-black tracking-tight ${result.category && categoryVerdict ? categoryVerdict.color : verdictStyle.color}`}>
                {result.category && categoryVerdict ? categoryVerdict.label : composedVerdict.label}
              </h2>
              <p className="text-sm text-white/70 mt-2 max-w-2xl">
                {result.category
                  ? (result.why || result.scene_description || "Layered forensic checks were applied across metadata and model signals.")
                  : composedVerdict.explanation}
              </p>
              <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/15 bg-black/20 text-[11px] text-white/70">
                <span className="uppercase tracking-wider text-white/45">Engine verdict</span>
                <span className="font-mono">{result.primary_verdict || "N/A"}</span>
              </div>
            </div>
          </div>

          <div className="lg:w-[260px] rounded-2xl border border-white/10 bg-black/20 p-4 space-y-3">
            <div className="flex items-baseline justify-between">
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Confidence</p>
              <p className="text-2xl font-black text-white font-mono">{Math.round(confidence * 100)}%</p>
            </div>
            <ScoreBar label="" score={confidence} showPercentage={false} color="bg-cyan-500" />
            <p className="text-[11px] font-bold text-cyan-300">{confidenceLabel(confidence)}</p>
            <p className="text-[10px] text-white/45">Confidence reflects analysis clarity and signal consistency, not absolute truth.</p>
          </div>
        </div>
      </div>

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

      {!result.category && (
        <div className="glass rounded-3xl border border-white/10 p-5 sm:p-6 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white/85 uppercase tracking-wider">Signal cards</h3>
              <p className="text-xs text-white/45">Showing {metrics.length} of 7 metrics based on relevance and transparency rules.</p>
            </div>
          </div>
          <ScoreCardGrid metrics={metrics} onOpenMetric={setActiveMetric} />
        </div>
      )}

      {result.category && (
        <div className="glass rounded-3xl border border-white/10 p-6 space-y-6">
          <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
            <div className="rounded-2xl bg-white/[0.03] border border-white/10 p-5 space-y-3">
              <p className="text-[10px] uppercase tracking-[0.2em] text-white/40 font-black">Scene description</p>
              <p className="text-white/85 text-sm leading-relaxed">{result.scene_description || result.explanation}</p>
              {result.detected_objects && result.detected_objects.length > 0 && (
                <div className="flex flex-wrap gap-2 pt-1">
                  {result.detected_objects.slice(0, 8).map((objectName) => (
                    <span key={objectName} className="text-[10px] px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300">
                      {objectName}
                    </span>
                  ))}
                </div>
              )}
              <div className="text-[11px] text-white/55">
                Style: <span className="text-white/80 font-semibold">{result.style || "unknown"}</span>
              </div>
            </div>
            <div className="rounded-2xl bg-black/20 border border-white/10 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Confidence</p>
                <p className="text-2xl font-black text-white font-mono">{Math.round((result.confidence ?? result.confidence_score * 100))}%</p>
              </div>
              <ScoreBar label="" score={toRatio(result.confidence ?? result.confidence_score)} showPercentage={false} color="bg-cyan-500" />
              <p className="text-[11px] text-cyan-300">{confidenceLabel(toRatio(result.confidence ?? result.confidence_score))}</p>
              <div className="text-[10px] text-white/45">Primary source marker: {result.source && result.source !== "Unknown" ? result.source : "No explicit generator signature found"}</div>
            </div>
          </div>

          {imageMetrics.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-bold text-white/85 uppercase tracking-wider">Key forensic signals</h3>
                <p className="text-[11px] text-white/45">Showing strongest indicators only</p>
              </div>
              <ScoreCardGrid metrics={imageMetrics} onOpenMetric={setActiveMetric} />
            </div>
          )}

          <details className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
            <summary className="cursor-pointer text-xs font-bold text-white/80 uppercase tracking-wider">
              Technical details
            </summary>
            <div className="mt-3 text-xs text-white/70 space-y-2">
              {result.if_uncertain && <p><span className="text-white/45">Uncertainty note:</span> {result.if_uncertain}</p>}
              <p><span className="text-white/45">Why this result:</span> {result.why || result.explanation}</p>
              <p><span className="text-white/45">Engine trigger:</span> {result.triggered_rule || "N/A"}</p>
            </div>
          </details>
        </div>
      )}

      {!result.category && (
        <div className="glass rounded-3xl border border-white/10 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-white/85 uppercase tracking-wider">Signal hierarchy</h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl bg-white/[0.03] border border-white/10 p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-400/90 font-black mb-2">Primary drivers</p>
              <div className="space-y-2">
                {primarySignals.map((metric) => (
                  <div key={metric.key} className="flex items-center justify-between text-xs text-white/75">
                    <span>{metric.label}</span>
                    <span className="font-mono">{Math.round(metric.score * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-2xl bg-white/[0.03] border border-white/10 p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-white/40 font-black mb-2">Secondary context</p>
              <div className="space-y-2">
                {secondarySignals.length > 0 ? (
                  secondarySignals.map((metric) => (
                    <div key={metric.key} className="flex items-center justify-between text-xs text-white/65">
                      <span>{metric.label}</span>
                      <span className="font-mono">{Math.round(metric.score * 100)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-white/45">No additional high-relevance signals were shown.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {result.category && result.signals && result.signals.length > 0 && (
        <div className="glass rounded-3xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">Signal checks</h3>
          </div>
          <div className="grid gap-2">
            {result.signals.map((signal, i) => (
              <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border ${signal.verified ? "bg-emerald-500/5 border-emerald-500/20" : "bg-rose-500/5 border-rose-500/20"}`}>
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${signal.verified ? "bg-emerald-500" : "bg-rose-500"}`} />
                <span className="text-xs font-semibold text-white/80 flex-1">{signal.source}</span>
                <span className="text-[10px] font-mono text-white/40">{Math.round(signal.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="glass rounded-3xl border border-white/10 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">
            {result.category ? "Forensic Analysis Report" : "Analysis Narrative"}
          </h3>
        </div>
        <p className="text-white/70 text-sm leading-relaxed mb-6 italic">{`"${result.explanation}"`}</p>

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
                    <span className={`text-[10px] font-bold w-16 text-right ${signal.verified ? "text-emerald-400" : "text-rose-400"}`}>
                      {signal.verified ? "VERIFIED" : "FLAGGED"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.metadata?.raw_metadata && Object.keys(result.metadata.raw_metadata).length > 0 && (
          <div className="mt-6 pt-6 border-t border-white/5">
            <h4 className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em] mb-4 text-center">Extraction: Source Metadata</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {Object.entries(result.metadata.raw_metadata).map(([key, value]) => (
                <div key={key} className="p-3 rounded-xl bg-black/30 border border-white/5 flex flex-col gap-1">
                  <span className="text-[10px] uppercase font-bold text-white/20 tracking-wider">{key.replace(/_/g, " ")}</span>
                  <span className="text-xs font-mono text-white/80 truncate" title={String(value)}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

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
          <p className="text-[10px] text-white/30 mt-3">Spectral and temporal audio features analyzed for synthetic speech indicators (TTS, voice cloning).</p>
        </div>
      )}

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
          <p className="text-[10px] text-white/30 mt-3">Semantic similarity cross-referenced against real-world news headlines to detect unsupported claims.</p>
        </div>
      )}

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
          <p className="text-[10px] text-white/30 mt-3">Text extracted from the image via OCR and fed into the truthfulness pipeline.</p>
        </div>
      )}

      <MetricModal metric={activeMetric} isOpen={!!activeMetric} onClose={() => setActiveMetric(null)} />

      <DebugPanel data={result} />
    </motion.div>
  );
}
