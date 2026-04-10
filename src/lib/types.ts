// Unified response type matching both FastAPI and mock backends
export interface AnalysisResult {
  truth_score: number;
  ai_generated_score: number;
  bias_score: number;
  credibility_score: number;
  confidence_score: number;
  features: {
    perplexity: number;
    stylometry: {
      sentence_length_variance: number;
      repetition_score: number;
      lexical_diversity: number;
    };
  };
  signals: {
    source: string;
    verified: boolean;
    confidence: number;
  }[];
  explanation: string;
  metadata: {
    model: string;
    latency_ms: number;
    timestamp: string;
  };
}

export type DetectionMode = "text" | "image" | "video";

export interface NavItem {
  label: string;
  href: string;
  icon?: string;
}
