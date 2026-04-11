---
title: TruthGuard AI Backend
emoji: 🛡️
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
---

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

tai/
├── frontend/                   # 🎨 Frontend (Next.js 15 + Shadcn UI)
│   ├── src/                    # 🗺️ Next.js routing & components
│   ├── public/                 # 🖼️ Static assets
│   └── package.json            # 📦 Dependencies & scripts
├── backend/                    # 🧠 Backend (FastAPI + Python)
│   ├── inference/engine.py     # 🚀 Coordination logic
│   ├── training/model.py       # 🧠 RoBERTa architecture
│   ├── main.py                 # 📞 API entry point
│   └── requirements.txt        # 🐍 Python dependencies
├── netlify.toml                # 🌐 Netlify deployment config
└── COLLABORATORS.md            # 🤝 Contributing guide

---

## 🔧 Installation & Setup

### 1. Backend Setup
```bash
# From the root directory
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
*The API will run on `http://localhost:8000` (FastAPI default) or `http://localhost:5000` (Mock/Node default).*

### 2. Frontend Setup
```bash
# From the root directory
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to see the dashboard.

---

## 🌐 Deployment

### Frontend (Netlify)
The repository is prepared for zero-config deployment on **Netlify**.
- **Base directory**: `frontend`
- **Build command**: `npm run build`
- **Publish directory**: `.next`

### Backend (Hugging Face)
The backend is optimized for deployment as a **Hugging Face Space** using Docker or a Python environment.
- Production API URL: `https://prana-v12-truthguard-api.hf.space`

---

## 📜 Roadmap
- [ ] **Multimodal Expansion**: Implementing OCR for image analysis.
- [ ] **Browser Extension**: Real-time analysis for web browsing.
- [ ] **API Access**: Public endpoints for third-party verification.

---

## 🤝 Community
We welcome all contributors. Please check out the [COLLABORATORS.md](COLLABORATORS.md) for current priorities and onboarding steps.
