# TAI - Trust & Authenticity Intelligence 🛡️

TAI is a state-of-the-art platform designed to combat misinformation and AI-generated deception. By leveraging a multi-task learning architecture based on **RoBERTa**, it provides a "Trust Score" for digital content based on veracity, origin, and bias.

---

## 🚀 The Vision: "The Trust Guard"
In a digital landscape filled with deepfakes and automated narratives, TAI acts as a transparent filter. It doesn't just give a "true" or "false" answer—it provides a breakdown of **why** a piece of content might be untrustworthy using stylometric features and cross-referenced news signals.

> [!TIP]
> **Interested in contributing?** Read our [Collaborators Guide](COLLABORATORS.md) to get started with the team!

---

## 🛠️ Technical Deep Dive

### 🧠 Backend: The Inference Engine
The core of TAI resides in `backend/inference/engine.py`. This engine coordinates several sophisticated subsystems:
- **Multi-Task RoBERTa**: A fine-tuned transformer model that outputs three scores simultaneously (Truth, AI Probability, and Bias).
- **Feature Extraction**: Uses `GPT-2` to calculate text **perplexity** (AI detection) and `Spacy` for lexical diversity and sentence variance (stylometry).
- **Trust Agents**: Located in `backend/services/trust_agents.py`, these agents use sentence-transformers to cross-reference claims against verified factual datasets.
- **Fusion Logic**: The engine "fuses" results from the local model, external agents, and (optionally) GPT-4o to provide a high-confidence final score.

### 🎨 Frontend: Modern Dashboard
Built with **Next.js 15** and **Shadcn UI**, the frontend is designed for high-density information display:
- **Real-time Scoring**: Interactive `ScoreBar` and `ResultDisplay` components.
- **Specialized Analysers**: Dedicated portals for Text, Image, and Video analysis.
- **Debug Panel**: A developer-friendly `DebugPanel.tsx` that exposes model latency, raw confidence scores, and feature breakdowns.

---

## 📂 Project Structure & Key Files

```bash
tai/
├── backend/
│   ├── inference/engine.py     # 🚀 The Heart: Coordinates model and agents
│   ├── training/model.py       # 🧠 The Brain: Multi-task RoBERTa architecture
│   ├── services/trust_agents.py # 🕵️ The Fact-Checker: Handles claim verification
│   └── train_full.py           # ⚗️ The Lab: Main training script for 75k samples
├── src/
│   ├── components/             # 🧱 The Building Blocks: Reusable UI cards/bars
│   ├── lib/api.ts              # 📞 The Bridge: Type-safe API client for backend
│   └── app/                    # 🗺️ The Map: Main Next.js routing structure
└── COLLABORATORS.md            # 🤝 The Community: How to help out
```

---

## 🔧 Installation & Setup

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```
*Note: On first run, it will attempt to load the RoBERTa model. If no local weights exist, it will fallback to a mock mode or download base weights.*

### 2. Frontend Setup
```bash
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to see the dashboard.

---

## 📜 Roadmap
- [ ] **Multimodal Expansion**: Implementing OCR for image analysis.
- [ ] **Browser Extension**: Real-time analysis for web browsing.
- [ ] **API Access**: Public endpoints for third-party verification.

---

## 🤝 Community
We welcome all contributors. Please check out the [COLLABORATORS.md](COLLABORATORS.md) for current priorities and onboarding steps.
