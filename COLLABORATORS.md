# Collaborator's Onboarding Guide 🤝

Welcome to the TAI development team! This guide provides a deeper level of detail for those looking to contribute to the code, models, or research of the Trust & Authenticity Intelligence platform.

---

## 🛠️ Technology-Specific Focus Areas

### 🎨 Frontend (React/Next.js)
The frontend is the window through which users perceive "Truth." We focus on **Rich Aesthetics** and **Dynamic Design**.
- **Current priorities**:
    - Enhancing the `ResultDisplay.tsx` with more granular "Evidence" cards.
    - Implementing the specialized handlers in `app/image` and `app/video`.
    - Optimizing Framer Motion animations for smoother transitions.

### 🐍 Backend (FastAPI/Python)
The backend is a high-performance pipeline that must handle complex NLP tasks with low latency.
- **Current priorities**:
    - **Inference Optimization**: Reducing the 17s/it training CPU bottleneck.
    - **Scraping & Extraction**: Improving `utils/url_extractor.py` to handle dynamic (JavaScript-rendered) content.
    - **Trust Agent Expansion**: Adding more factual sources to `services/trust_agents.py`.

### 🧠 AI & Machine Learning
We are constantly refining our multi-task RoBERTa model.
- **Current priorities**:
    - **Stylometry Research**: Improving `FeatureExtractor.py` to better detect LLM-specific patterns like frequent word choice or lack of sentence variance.
    - **Dataset Cleaning**: Helping prune the 75k dataset for higher quality samples.
    - **Quantization**: Exporting model weights to more efficient formats for CPU deployment.

---

## 🏗️ Getting Started (Step-by-Step)

1.  **Fork and Clone**: Pull the repo and create a new feature branch.
2.  **Environment Setup**:
    - **Node**: `npm install` (v18+)
    - **Python**: `pip install -r backend/requirements.txt` (3.10+)
3.  **Local Testing**:
    - Start the backend: `python backend/main.py`.
    - Start the frontend: `npm run dev`.
    - Verify they are communicating correctly (the `DebugPanel` in the UI is your friend here).
4.  **Pull Request**: Provide a clear description of what changed and any performance/accuracy benchmarks if applicable.

---

## 📏 Standards & Best Practices

- **TypeScript**: We use strict typing. Avoid `any` at all costs.
- **Python**: Follow PEP 8. All new classes in the `inference` logic should include docstrings explaining their "Trust Signal" contribution.
- **Styling**: Stick to the Tailwind design system defined in `globals.css` and Shadcn variables.
- **Documentation**: If you change an API contract, update `README.md`.

---

## 📬 Communication
We believe in transparent and asynchronous communication. Use GitHub Issues for feature requests and Discussions for architectural brainstorming.

**Back to [README.md](README.md)** | **Let's build something trustworthy.**
