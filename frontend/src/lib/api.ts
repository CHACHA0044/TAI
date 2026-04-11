import { AnalysisResult, JobResponse } from './types';

/**
 * Resolve the backend base URL.
 *
 * Priority:
 *  1. NEXT_PUBLIC_API_URL env var  →  set this in Vercel / Netlify / Docker for any deployment
 *  2. Fallback: http://localhost:8000  →  works automatically for local development
 *
 * No code changes or env files are required to run locally.
 * For deployment, set NEXT_PUBLIC_API_URL to your deployed backend URL.
 */
function resolveApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, ''); // strip trailing slash
  }
  // In a browser context during local dev, default to localhost backend
  if (typeof window !== 'undefined') {
    return 'http://localhost:8000';
  }
  // SSR fallback (Next.js server-side calls) — also default to localhost
  return 'http://localhost:8000';
}

const API_BASE_URL = resolveApiBase();

if (process.env.NODE_ENV === 'development') {
  console.info(`[TruthGuard] API base: ${API_BASE_URL}`);
}

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
export async function analyzeImage(image: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', image);

  const response = await fetch(`${API_BASE_URL}/analyze-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to analyze image');
  }

  return response.json();
}

export async function analyzeVideo(video: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', video);

  const response = await fetch(`${API_BASE_URL}/analyze-video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to analyze video');
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
    method: 'GET',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to fetch job status');
  }

  return response.json();
}
