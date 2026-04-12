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
  const engineVerdict = input.primaryVerdict;

  if (opinion >= 0.62 && verifiability <= 0.62 && sarcasm < 0.58) {
    return {
      label: truth < 0.55 ? "Personal Review / Opinion" : "Subjective Experience",
      tone: "sky",
      explanation: "The text is dominated by personal perspective and subjective experience, so objective fact-verification is limited.",
    };
  }

  if (truth < 0.25 && (bias > 0.8 || manipulation > 0.75)) {
    return {
      label: "Biased / Manipulative Content",
      tone: "rose",
      explanation: "Very low truth alignment combined with extreme framing pressure indicates high-risk misinformation-style content.",
    };
  }

  if (ai >= 0.9 && truth < 0.7 && engineVerdict !== "VERIFIED_FACT") {
    return {
      label: "AI-Generated Formal Writing",
      tone: "fuchsia",
      explanation: "The writing pattern strongly matches machine-generated formal prose. Verify origin and factual claims independently.",
    };
  }

  if (bias >= 0.6 && manipulation < 0.55) {
    return {
      label: "Biased / Manipulative Content",
      tone: "amber",
      explanation: "Loaded framing and one-sided language are dominant, which can distort interpretation even when isolated facts are present.",
    };
  }

  if (truth >= 0.75 && verifiability >= 0.58 && bias < 0.45 && manipulation < 0.45 && opinion < 0.55 && sarcasm < 0.5) {
    return {
      label: "Factually Supported Statement",
      tone: "emerald",
      explanation: "The core claim aligns with strong factual signals and appears coherent with established knowledge.",
    };
  }

  if (truth >= 0.64 && verifiability < 0.58 && sarcasm < 0.55) {
    return {
      label: "Unverifiable Claim",
      tone: "amber",
      explanation: "The statement may be plausible, but current evidence is insufficient to verify it confidently.",
    };
  }

  if (truth <= 0.34 && verifiability >= 0.5) {
    return {
      label: "Likely False / Unsupported Claim",
      tone: "rose",
      explanation: "Evidence leans against the core claim, suggesting the statement is misleading or factually incorrect.",
    };
  }

  if (verifiability < 0.46 && truth < 0.64) {
    return {
      label: "Unverifiable Claim",
      tone: "amber",
      explanation: "There is not enough reliable evidence to confidently confirm or reject this claim right now.",
    };
  }

  if ((sarcasm >= 0.58 && truth < 0.7) || (confidence < 0.45 && sarcasm > 0.45)) {
    return {
      label: "Satirical / Non-Literal Framing",
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
      label: "Mixed Fact and Opinion",
      tone: "slate",
      explanation: "Multiple competing signals are present, so this text is best interpreted with nuance rather than a single hard category.",
    };
  }

  if (engineVerdict === "FALSE_FACT" || engineVerdict === "CONSPIRACY_OR_EXTRAORDINARY_CLAIM") {
    return {
      label: "Likely False / Unsupported Claim",
      tone: "rose",
      explanation: "The engine detected contradiction or extraordinary-claim patterns that are not credibly supported.",
    };
  }

  if (engineVerdict === "UNVERIFIED_CLAIM") {
    return {
      label: "Unverifiable Claim",
      tone: "amber",
      explanation: "The claim lacks enough trustworthy evidence to support a definitive factual decision.",
    };
  }

  const fallbackTone = input.primaryVerdict ? toneByEngineVerdict[input.primaryVerdict] ?? "slate" : "slate";
  return {
    label: "Context-Dependent Analysis",
    tone: fallbackTone,
    explanation: "No single signal fully dominates, so this result reflects a balanced interpretation with residual uncertainty.",
  };
}
