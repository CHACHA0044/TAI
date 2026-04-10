import { AnalysisResult } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export async function analyzeText(text: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: text }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to analyze text');
  }

  return response.json();
}

export async function analyzeURL(url: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: url }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to analyze URL');
  }

  return response.json();
}

// Placeholders for image and video (mock mode)
export async function analyzeImage(_image: File): Promise<AnalysisResult> {
  console.warn("analyzeImage not yet implemented — returning mock");
  return {
    truth_score: 0.5,
    ai_generated_score: 0.72,
    bias_score: 0.1,
    credibility_score: 0.5,
    confidence_score: 0.84,
    explanation: "Image analysis is currently in mock mode.",
    features: {
      perplexity: 0,
      stylometry: { sentence_length_variance: 0, repetition_score: 0, lexical_diversity: 0 },
    },
    signals: [],
    metadata: { model: "mock", latency_ms: 0, timestamp: new Date().toISOString() },
  };
}

export async function analyzeVideo(_video: File): Promise<AnalysisResult> {
  console.warn("analyzeVideo not yet implemented — returning mock");
  return {
    truth_score: 0.5,
    ai_generated_score: 0.92,
    bias_score: 0.1,
    credibility_score: 0.5,
    confidence_score: 0.96,
    explanation: "Video analysis is currently in mock mode.",
    features: {
      perplexity: 0,
      stylometry: { sentence_length_variance: 0, repetition_score: 0, lexical_diversity: 0 },
    },
    signals: [],
    metadata: { model: "mock", latency_ms: 0, timestamp: new Date().toISOString() },
  };
}
