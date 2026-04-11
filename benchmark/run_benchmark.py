import json
import csv
import time
import os
import requests
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("benchmark/results/benchmark_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")

# Configuration
DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000/analyze-text",
    "timeout": 120,
    "batch_size": 5,
    "output_dir": "benchmark/results",
    "delay": 1.0,
}

def normalize_score(score: float) -> str:
    """Normalize 0-1.0 or 0-100 score into LOW, MEDIUM, HIGH."""
    # Handle both 0-1.0 and 0-100
    if score <= 1.0:
        val = score * 100
    else:
        val = score
        
    if val <= 33:
        return "LOW"
    elif val <= 66:
        return "MEDIUM"
    else:
        return "HIGH"

def map_api_verdict_to_benchmark(api_verdict: str) -> str:
    """Maps API verdict strings to benchmark category names."""
    v = api_verdict.upper()
    if "VERIFIED AUTHENTIC" in v or "VERIFIED FACT" in v:
        return "VERIFIED_FACT"
    if "FALSE" in v or "MISLEADING" in v:
        return "FALSE_FACT"
    if "UNVERIFIED" in v:
        return "UNVERIFIED_CLAIM"
    if "OPINION" in v:
        return "OPINION"
    if "BIASED" in v:
        return "BIASED_CONTENT"
    if "MANIPULATIVE" in v:
        return "MANIPULATIVE_CONTENT"
    if "SATIRE" in v or "SARCASM" in v:
        return "SATIRE_OR_SARCASM"
    if "CONSPIRACY" in v:
        return "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"
    if "AI" in v:
        return "LIKELY_AI_GENERATED"
    return "MIXED_ANALYSIS"

class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        os.makedirs(self.config["output_dir"], exist_ok=True)
        self.results = []
        self.failures = []

    def run(self, input_file: str, reverse=False):
        logger.info(f"Loading benchmark inputs from {input_file}...")
        with open(input_file, "r") as f:
            inputs = json.load(f)

        if reverse:
            logger.info("Reversing input order (200 -> 0)...")
            inputs = inputs[::-1]

        total = len(inputs)
        logger.info(f"Starting benchmark run for {total} test cases...")

        for i, case in enumerate(inputs, 1):
            logger.info(f"[{i}/{total}] Processing ID {case['id']}: {case['text'][:70]}...")
            
            try:
                actual = self._call_api(case["text"])
                comparison = self._compare(case, actual)
                self.results.append(comparison)
                if not comparison["overall_pass"]:
                    self.failures.append(comparison)
            except Exception as e:
                logger.error(f"  FAILED to process ID {case['id']}: {e}")
                # Log a failure result for consistency
                self.results.append({
                    "id": case["id"],
                    "category": case.get("category", "UNKNOWN"),
                    "input_text": case["text"],
                    "error": str(e),
                    "overall_pass": False,
                    "timestamp": datetime.now().isoformat()
                })

            # Incremental save every 10 cases
            if i % 10 == 0:
                self._export_csv()

        self._export_csv()
        self._generate_confusion_matrix()
        self._print_summary()

    def _call_api(self, text: str) -> Dict[str, Any]:
        try:
            # Respect rate limits
            time.sleep(self.config["delay"])
            
            response = requests.post(
                self.config["api_url"],
                json={"text": text},
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API Call failed: {str(e)}")
            return {
                "verdict": "ERROR",
                "scores": {"truth": 0, "ai": 0, "bias": 0},
                "explanation": f"API Error: {str(e)}"
            }

    def _compare(self, case: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        expected = case["expected"]
        
        # Mapping and Normalization
        actual_verdict = map_api_verdict_to_benchmark(actual.get("verdict", ""))
        actual_truth = normalize_score(actual.get("truth_score", 0))
        actual_ai = normalize_score(actual.get("ai_generated_score", 0))
        actual_bias = normalize_score(actual.get("bias_score", 0))
        # Use bias as proxy for manipulation if not available
        actual_manipulation = normalize_score(actual.get("bias_score", 0)) 

        # Matches
        verdict_match = actual_verdict == expected["primary_verdict"]
        truth_match = actual_truth == expected["truth_score"]
        ai_match = actual_ai == expected["ai_likelihood"]
        bias_match = actual_bias == expected["bias_score"]
        manip_match = actual_manipulation == expected["manipulation_score"]

        overall_pass = all([verdict_match, truth_match, ai_match, bias_match, manip_match])

        res = {
            "id": case["id"],
            "category": case.get("category", ""),
            "input_text": case["text"],
            "expected_primary_verdict": expected["primary_verdict"],
            "actual_primary_verdict": actual_verdict,
            "verdict_match": verdict_match,
            "expected_truth_score": expected["truth_score"],
            "actual_truth_score": actual_truth,
            "truth_match": truth_match,
            "expected_ai_likelihood": expected["ai_likelihood"],
            "actual_ai_likelihood": actual_ai,
            "ai_match": ai_match,
            "expected_bias_score": expected["bias_score"],
            "actual_bias_score": actual_bias,
            "bias_match": bias_match,
            "expected_manipulation_score": expected["manipulation_score"],
            "actual_manipulation_score": actual_manipulation,
            "manipulation_match": manip_match,
            "overall_pass": overall_pass,
            "confidence": actual.get("confidence_score", 0),
            "backend_explanation": actual.get("explanation", ""),
            "aggregation_rule_triggered": ",".join(actual.get("key_factors", [])),
            "timestamp": datetime.now().isoformat(),
            # Debug info
            "raw_scores": f"truth={actual.get('truth_score')}, ai={actual.get('ai_generated_score')}, bias={actual.get('bias_score')}",
            "feature_metrics": json.dumps(actual.get("features", {})),
        }
        return res

    def _export_csv(self):
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        main_path = os.path.join(self.config["output_dir"], "benchmark_results.csv")
        df.to_csv(main_path, index=False)
        
        if self.failures:
            fail_df = pd.DataFrame(self.failures)
            fail_path = os.path.join(self.config["output_dir"], "failures_only.csv")
            fail_df.to_csv(fail_path, index=False)
            logger.info(f"Saved incremental results: {len(self.results)} total, {len(self.failures)} failures")
            # Sort by verdict mismatch first, then others
            fail_df['v_mismatch'] = ~fail_df['verdict_match']
            fail_df = fail_df.sort_values(by=['v_mismatch', 'id'], ascending=[False, True])
            fail_df = fail_df.drop(columns=['v_mismatch'])
            fail_path = os.path.join(self.config["output_dir"], "failures_only.csv")
            fail_df.to_csv(fail_path, index=False)
            print(f"Exported failure analysis to {fail_path}")

    def _generate_confusion_matrix(self):
        try:
            from sklearn.metrics import confusion_matrix
            df = pd.DataFrame(self.results)
            if df.empty or 'expected_primary_verdict' not in df.columns:
                return

            labels = sorted(df['expected_primary_verdict'].unique())
            cm = confusion_matrix(df['expected_primary_verdict'], df['actual_primary_verdict'], labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            cm_path = os.path.join(self.config["output_dir"], "confusion_matrix.csv")
            cm_df.to_csv(cm_path)
            print(f"Exported confusion matrix to {cm_path}")
        except ImportError:
            print("Skipping confusion matrix (sklearn not found).")
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")

    def _print_summary(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No results to summarize.")
            return

        total = len(df)
        passed = df['overall_pass'].sum()
        failed = total - passed
        accuracy = (passed / total) * 100 if total > 0 else 0

        print("\n" + "="*40)
        print("      BENCHMARK SUMMARY REPORT")
        print("="*40)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        print(f"Accuracy:     {accuracy:.2f}%")
        print("-" * 40)
        
        # Accuracy by Verdict Type
        if 'category' in df.columns:
            print("Accuracy By Category:")
            cat_stats = df.groupby('category')['overall_pass'].agg(['count', 'sum'])
            for cat, row in cat_stats.iterrows():
                cat_acc = (row['sum'] / row['count']) * 100
                print(f"  {cat:35} : {cat_acc:6.2f}% ({row['sum']}/{row['count']})")
        
        # Common Failure Modes
        if failed > 0:
            print("-" * 40)
            print("Common Failure Modes (Mismatches):")
            mismatches = {
                "Verdict": (~df['verdict_match']).sum(),
                "Truth Score": (~df['truth_match']).sum(),
                "AI Likelihood": (~df['ai_match']).sum(),
                "Bias Score": (~df['bias_match']).sum(),
                "Manipulation": (~df['manipulation_match']).sum(),
            }
            for k, v in sorted(mismatches.items(), key=lambda x: x[1], reverse=True):
                if v > 0:
                    print(f"  {k:15}: {v} cases")
        print("="*40 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TruthGuard AI Benchmark Runner")
    parser.add_argument("--url", help="API URL", default=DEFAULT_CONFIG["api_url"])
    parser.add_argument("--input", help="Inputs JSON", default="benchmark/benchmark_inputs.json")
    parser.add_argument("--out", help="Output directory", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--reverse", action="store_true", help="Run tests in reverse order")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner({
        "api_url": args.url,
        "output_dir": args.out
    })
    runner.run(args.input, reverse=args.reverse)
