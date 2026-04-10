const express = require("express");
const cors = require("cors");
const multer = require("multer");
const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

// Delay helper to simulate AI processing time
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ------------------------------------------------------------------
// Helper: build a response matching the unified frontend contract
// ------------------------------------------------------------------
function buildAnalysisResponse({
  truthScore,
  aiScore,
  biasScore,
  credibilityScore,
  confidenceScore,
  explanation,
  signals,
  perplexity,
  sentenceVariance,
  repetitionScore,
  lexicalDiversity,
  latencyMs,
}) {
  return {
    truth_score: truthScore,
    ai_generated_score: aiScore,
    bias_score: biasScore,
    credibility_score: credibilityScore,
    confidence_score: confidenceScore,
    explanation,
    features: {
      perplexity,
      stylometry: {
        sentence_length_variance: sentenceVariance,
        repetition_score: repetitionScore,
        lexical_diversity: lexicalDiversity,
      },
    },
    signals,
    metadata: {
      model: "mock-server-v1",
      latency_ms: latencyMs,
      timestamp: new Date().toISOString(),
    },
  };
}

// ------------------------------------------------------------------
// POST /analyze-text
// ------------------------------------------------------------------
app.post("/analyze-text", async (req, res) => {
  const start = Date.now();
  const text = req.body.content || req.body.text || "";

  if (!text.trim()) {
    return res.status(422).json({ detail: "Request body must include 'content' or 'text'." });
  }

  // Simulate processing delay (1-2s)
  await delay(1000 + Math.random() * 1000);

  const isFake =
    text.toLowerCase().includes("urgent") ||
    text.toLowerCase().includes("secret") ||
    text.toLowerCase().includes("shocking");

  const truthScore = isFake ? 0.22 : 0.82;
  const aiScore = isFake ? 0.85 : 0.12;

  res.json(
    buildAnalysisResponse({
      truthScore,
      aiScore,
      biasScore: isFake ? 0.72 : 0.15,
      credibilityScore: isFake ? 0.25 : 0.88,
      confidenceScore: isFake ? 0.90 : 0.85,
      explanation: isFake
        ? "High concentration of emotional triggers and unsubstantiated claims. Syntax matches known AI generation patterns."
        : "Text structure and varied vocabulary indicate human authorship. Claims cross-referenced with reliable databases show no anomalies.",
      signals: [
        { source: "Reuters - headline match", verified: !isFake, confidence: isFake ? 0.3 : 0.92 },
        { source: "AP News - topical check", verified: !isFake, confidence: isFake ? 0.25 : 0.88 },
      ],
      perplexity: isFake ? 18.4 : 62.7,
      sentenceVariance: isFake ? 3.2 : 28.5,
      repetitionScore: isFake ? 0.35 : 0.06,
      lexicalDiversity: isFake ? 0.42 : 0.78,
      latencyMs: Date.now() - start,
    })
  );
});

// ------------------------------------------------------------------
// POST /analyze-url
// ------------------------------------------------------------------
app.post("/analyze-url", async (req, res) => {
  const start = Date.now();
  const url = req.body.content || req.body.url || "";

  if (!url.trim()) {
    return res.status(422).json({ detail: "Request body must include 'content' or 'url'." });
  }

  await delay(1500 + Math.random() * 1500);

  res.json(
    buildAnalysisResponse({
      truthScore: 0.65,
      aiScore: 0.30,
      biasScore: 0.40,
      credibilityScore: 0.70,
      confidenceScore: 0.72,
      explanation:
        "Content extracted from URL shows moderate factual alignment. Some claims require further verification from primary sources.",
      signals: [
        { source: "BBC News - similar report", verified: true, confidence: 0.75 },
        { source: "Snopes - partial match", verified: false, confidence: 0.45 },
      ],
      perplexity: 41.3,
      sentenceVariance: 15.8,
      repetitionScore: 0.12,
      lexicalDiversity: 0.65,
      latencyMs: Date.now() - start,
    })
  );
});

// ------------------------------------------------------------------
// GET /health
// ------------------------------------------------------------------
app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "truthguard-mock-api" });
});

// ------------------------------------------------------------------
// POST /analyze-image (mock)
// ------------------------------------------------------------------
app.post("/analyze-image", upload.single("image"), async (req, res) => {
  await delay(2000 + Math.random() * 2000);
  res.json({
    verdict: "SUSPICIOUS",
    confidence: 0.84,
    fakeProbability: 0.72,
    explanation:
      "Detected anomalous pixel gradients and inconsistent lighting, characteristic of diffusion-based AI generation.",
    details: [
      { label: "Noise Analysis (ELA)", value: "High Variance", score: 0.8 },
      { label: "Lighting Consistency", value: "Suspicious", score: 0.65 },
      { label: "GAN Artifacts", value: "Detected", score: 0.75 },
    ],
    processingTime: Number((2 + Math.random() * 2).toFixed(2)),
  });
});

// ------------------------------------------------------------------
// POST /analyze-video (mock)
// ------------------------------------------------------------------
app.post("/analyze-video", upload.single("video"), async (req, res) => {
  await delay(3000 + Math.random() * 2000);
  res.json({
    verdict: "FAKE",
    confidence: 0.96,
    fakeProbability: 0.92,
    explanation:
      "Deepfake detected. Frame-by-frame temporal analysis identified significant lip-sync deviation.",
    details: [
      { label: "Lip Sync Deviation", value: "High (0.42s)", score: 0.95 },
      { label: "Temporal Smoothing", value: "Detected", score: 0.88 },
      { label: "Pulse/Heartbeat AI", value: "Missing", score: 0.9 },
    ],
    processingTime: Number((3 + Math.random() * 2).toFixed(2)),
  });
});

app.listen(port, () => {
  console.log(`TruthGuard mock API running on http://localhost:${port}`);
});
