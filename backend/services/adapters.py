from typing import Dict, Any, List


class HuggingFaceBiasClassifierAdapter:
    """Placeholder adapter for future HF bias classification integration."""

    def classify(self, text: str) -> Dict[str, Any]:
        return {"bias_score": 0.0, "indicators": [], "provider": "hf_stub"}


class ExternalFactCheckAdapter:
    """Placeholder adapter for external fact-check/search APIs."""

    def lookup(self, claim: str) -> Dict[str, Any]:
        return {"claim": claim, "matches": [], "provider": "external_factcheck_stub"}


class SearchVerificationPipelineAdapter:
    """Placeholder adapter for full search verification orchestration."""

    def run(self, text: str) -> List[Dict[str, Any]]:
        return []
