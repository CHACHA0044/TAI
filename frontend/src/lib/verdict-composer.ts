import { PrimaryVerdict } from "@/lib/types";

export interface VerdictComposeInput {
  primaryVerdict?: PrimaryVerdict;
  truth: number;
  verifiability: number;
  aiLikelihood: number;
  bias: number;
  manipulation: number;
  opinion: number;
  sarcasm: number;
  confidence: number;
}

export interface ComposedVerdict {
  label: string;
  explanation: string;
  tone: "emerald" | "amber" | "rose" | "sky" | "purple" | "fuchsia" | "slate";
}

const clamp = (value: number) => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

const toneByEngineVerdict: Partial<Record<PrimaryVerdict, ComposedVerdict["tone"]>> = {
  VERIFIED_FACT: "emerald",
  FALSE_FACT: "rose",
  UNVERIFIED_CLAIM: "amber",
  OPINION: "sky",
  BIASED_CONTENT: "amber",
  MANIPULATIVE_CONTENT: "rose",
  SATIRE_OR_SARCASM: "purple",
  CONSPIRACY_OR_EXTRAORDINARY_CLAIM: "amber",
  LIKELY_AI_GENERATED: "fuchsia",
  MIXED_ANALYSIS: "slate",
};

export function composeVerdict(input: VerdictComposeInput): ComposedVerdict {
  const truth = clamp(input.truth);
  const verifiability = clamp(input.verifiability);
  const ai = clamp(input.aiLikelihood);
  const bias = clamp(input.bias);
  const manipulation = clamp(input.manipulation);
  const opinion = clamp(input.opinion);
  const sarcasm = clamp(input.sarcasm);
  const confidence = clamp(input.confidence);

  if (ai >= 0.95 && truth < 0.7 && input.primaryVerdict !== "VERIFIED_FACT") {
    return {
      label: "Likely AI-Generated Writing",
      tone: "fuchsia",
      explanation: "The writing style strongly resembles machine-generated patterns. Treat it as synthetic-style text unless independently verified.",
    };
  }

  if (opinion >= 0.6 && truth < 0.75 && verifiability < 0.65) {
    return {
      label: "Opinion / Subjective Statement",
      tone: "sky",
      explanation: "The text leans on subjective judgment and preference language, so it should be read as perspective rather than a strictly checkable fact.",
    };
  }

  if (bias >= 0.6 && manipulation < 0.55) {
    return {
      label: "Biased / Slanted Content",
      tone: "amber",
      explanation: "The analysis detected loaded or one-sided framing that can shape interpretation even when parts of the text may still be factual.",
    };
  }

  if (truth >= 0.75 && verifiability >= 0.58 && bias < 0.45 && manipulation < 0.45 && opinion < 0.55 && sarcasm < 0.5) {
    return {
      label: "Verified Fact",
      tone: "emerald",
      explanation: "Core claims align with reliable evidence and are sufficiently verifiable.",
    };
  }

  if (truth >= 0.64 && verifiability < 0.58 && sarcasm < 0.55) {
    return {
      label: "Likely True but Unverifiable",
      tone: "amber",
      explanation: "The claim appears plausible, but available evidence is incomplete or not concrete enough for full verification.",
    };
  }

  if (truth <= 0.34 && verifiability >= 0.5) {
    return {
      label: "Likely Misleading / False Claim",
      tone: "rose",
      explanation: "Evidence leans against the core claim, suggesting the statement is misleading or factually incorrect.",
    };
  }

  if (verifiability < 0.46 && truth < 0.64) {
    return {
      label: "Unverified Claim",
      tone: "amber",
      explanation: "There is not enough reliable evidence to confidently confirm or reject this claim right now.",
    };
  }

  if ((sarcasm >= 0.58 && truth < 0.7) || (confidence < 0.45 && sarcasm > 0.45)) {
    return {
      label: "Mixed / Ambiguous Analysis",
      tone: "purple",
      explanation: "Sarcasm or rhetorical framing introduces ambiguity, so literal factual interpretation is less reliable.",
    };
  }

  const mixedSignals = [
    truth >= 0.6,
    verifiability >= 0.55,
    bias >= 0.55,
    manipulation >= 0.55,
    opinion >= 0.55,
    sarcasm >= 0.55,
  ].filter(Boolean).length;

  if (mixedSignals >= 3) {
    return {
      label: "Mixed / Ambiguous Analysis",
      tone: "slate",
      explanation: "Multiple competing signals are present, so this text is best interpreted with nuance rather than a single hard category.",
    };
  }

  const fallbackTone = input.primaryVerdict ? toneByEngineVerdict[input.primaryVerdict] ?? "slate" : "slate";
  return {
    label: "Mixed / Ambiguous Analysis",
    tone: fallbackTone,
    explanation: "Signals are not strong enough in one direction to give a definitive classification.",
  };
}
