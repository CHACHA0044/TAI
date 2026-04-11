// Unified response type matching both FastAPI and mock backends
export interface AnalysisResult {
  truth_score: number;
  ai_generated_score: number;
  bias_score: number;
  credibility_score: number;
  confidence_score: number;
  primary_verdict?: PrimaryVerdict;
  confidence?: number;
  dimensions?: Dimensions;
  expanded_analysis?: ExpandedAnalysis;
  triggered_rule?: string;
  claim_type?: string;
  dimension_buckets?: Record<string, "LOW" | "MEDIUM" | "HIGH">;
  debug?: {
    text_type_detected?: string;
    verifiability_result?: string;
    trust_agent_confidence?: "support" | "contradiction" | "inconclusive" | string;
    retrieval_support_score?: number;
    retrieval_contradiction_score?: number;
    fusion_weights?: Record<string, number>;
    triggered_rule?: string;
    detector_fired_first?: string;
    why_verdict_chosen?: string;
    final_rule_triggered?: string;
    raw_intermediate_scores?: Record<string, unknown>;
  };
  manipulation_score?: number;
  conspiracy_flag?: boolean;
  sarcasm?: boolean;
  features: {
    perplexity: number;
    stylometry: {
      sentence_length_variance: number;
      repetition_score: number;
      lexical_diversity: number;
      burstiness?: number;
    };
  };
  signals: {
    source: string;
    verified: boolean;
    confidence: number;
  }[];
  intelligence?: {
    summary: string;
    tags: string[];
  };
  explanation: string;
  category?: "REAL" | "AI_GENERATED" | "DEEPFAKE";
  source?: string;
  metadata: {
    model: string;
    latency_ms: number;
    timestamp: string;
    raw_metadata?: Record<string, unknown>;
  };
  // Extended forensics fields
  audio_score?: number;
  news_consistency_score?: number;
  ocr_text?: string;
  frame_scores?: number[];
}

export type PrimaryVerdict =
  | "VERIFIED_FACT"
  | "FALSE_FACT"
  | "UNVERIFIED_CLAIM"
  | "OPINION"
  | "BIASED_CONTENT"
  | "MANIPULATIVE_CONTENT"
  | "SATIRE_OR_SARCASM"
  | "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"
  | "LIKELY_AI_GENERATED"
  | "MIXED_ANALYSIS";

export interface Dimensions {
  truth_score: number;
  verifiability?: number;
  ai_likelihood: number;
  bias_score: number;
  manipulation_score: number;
  sarcasm_score?: number;
  opinion_score?: number;
  sarcasm: boolean;
  conspiracy_flag: boolean;
}

export interface ExpandedAnalysis {
  truth_score?: {
    explanation?: string;
    evidence?: string;
    sources?: string[];
  };
  ai_likelihood?: {
    explanation?: string;
    indicators?: string[];
  };
  bias_score?: {
    explanation?: string;
    indicators?: string[];
  };
  manipulation_score?: {
    explanation?: string;
    indicators?: string[];
  };
  opinion_score?: {
    explanation?: string;
    indicators?: string[];
  };
  verifiability?: {
    explanation?: string;
    indicators?: string[];
  };
}

export type DetectionMode = "text" | "image" | "video";

export interface JobResponse {
  status: "processing" | "complete" | "failed";
  job_id?: string;
  result?: AnalysisResult;
  progress?: string;
  error?: string;
}

export interface NavItem {
  label: string;
  href: string;
  icon?: string;
}
