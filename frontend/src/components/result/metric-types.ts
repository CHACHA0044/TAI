import { ReactNode } from "react";

export type MetricKey = "truth" | "verifiability" | "ai" | "bias" | "manipulation" | "opinion" | "sarcasm";

export interface MetricCardData {
  key: MetricKey;
  label: string;
  description: string;
  score: number;
  invertColor?: boolean;
  icon: ReactNode;
  relevance: number;
  meaning: string;
  whyAssigned: string;
  indicators: string[];
  interpretation: string;
  confidenceContext: string;
  evidence?: string;
  sources?: string[];
}
