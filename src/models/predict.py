"""
GitHub Issue Complexity Predictor
Loads a fine-tuned DistilBERT classifier when available, with a
robust heuristic fallback. Persists recent predictions for the dashboard.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import csv

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


PREDICTIONS_FILE = os.path.join("data", "raw", "predictions.jsonl")
BEST_MODEL_DIR = "best_model_bert_3class"
HUMAN_FEEDBACK_FILE = os.path.join("data", "raw", "human_feedback.csv")


class GitHubIssuePredictor:
    """Predict complexity of GitHub issues using a fine-tuned BERT model if present,
    otherwise fall back to heuristics. Also provides batch prediction and persistence.
    """

    def __init__(self) -> None:
        # Ensure predictions file directory exists
        os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try loading fine-tuned classifier from local directory
        self.tokenizer = None
        self.config = None
        self.model = None
        try:
            if os.path.exists(BEST_MODEL_DIR):
                self.tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR, local_files_only=True)
                self.config = AutoConfig.from_pretrained(BEST_MODEL_DIR, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    BEST_MODEL_DIR,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                ).to(self.device)
                self.model.eval()
        except Exception:
            # If loading fails, keep heuristic-only mode
            self.tokenizer = None
            self.config = None
            self.model = None

    def predict_complexity(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Predict complexity for a single issue.
        Returns a dict with keys: complexity, confidence, probabilities.
        Uses the trained model if available; otherwise, heuristics.
        """
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        labels = issue.get("labels") or []
        comments = int(issue.get("comments") or 0)

        # Normalize labels to lower-case strings
        norm_labels = []
        for l in labels:
            if isinstance(l, dict):
                name = l.get("name") or ""
            else:
                name = str(l)
            norm_labels.append(name.lower())

        text = f"{title}\n\n{body}".strip()

        # If model is available, use it
        if self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(
                    text if text else title,
                    truncation=True,
                    max_length=getattr(self.config, "max_position_embeddings", 512) or 512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.cpu().numpy()[0]
                # Softmax
                exps = np.exp(logits - np.max(logits))
                probs_arr = exps / np.sum(exps)

                # Map labels from config
                id2label = getattr(self.config, "id2label", {0: "SIMPLE", 1: "MODERATE", 2: "COMPLEX"})
                labels_map = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {0: "SIMPLE", 1: "MODERATE", 2: "COMPLEX"}
                class_names = [labels_map.get(i, "SIMPLE") for i in range(len(probs_arr))]
                # Normalize to lowercase keys: simple/moderate/complex
                probs: Dict[str, float] = {}
                for name, p in zip(class_names, probs_arr):
                    probs[name.lower()] = round(float(p), 4)
                # Ensure all keys exist
                for k in ["simple", "moderate", "complex"]:
                    probs.setdefault(k, 0.0)

                complexity = max(probs, key=probs.get)
                confidence = probs[complexity]

                return {
                    "complexity": complexity,
                    "confidence": float(confidence),
                    "probabilities": probs,
                }
            except Exception:
                # Fall through to heuristics
                pass

        # Heuristic fallback
        char_len = len(text)
        word_len = len(text.split())

        # Heuristic signals
        has_perf = any(x in norm_labels for x in ["performance", "optimization", "latency"]) or ("perf" in title.lower())
        has_bug = any(x in norm_labels for x in ["bug", "regression"])
        has_feature = any(x in norm_labels for x in ["feature", "enhancement"])
        has_docs = any(x in norm_labels for x in ["docs", "documentation"])
        has_security = any(x in norm_labels for x in ["security", "vulnerability", "cve"])

        # Base score from length and comments
        score = 0.0
        score += min(1.0, char_len / 4000.0) * 0.6
        score += min(1.0, word_len / 800.0) * 0.4

        # Label modifiers
        if has_security:
            score += 0.25
        if has_perf:
            score += 0.15
        if has_bug:
            score += 0.1
        if has_docs:
            score -= 0.1

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Map score to class probabilities
        # We use a smooth triangular distribution across simple/moderate/complex
        p_simple = max(0.0, 1.0 - 2.0 * score)
        p_complex = max(0.0, 2.0 * score - 1.0)
        # Ensure probabilities sum to <= 1.0 then assign remainder to moderate
        remainder = 1.0 - (p_simple + p_complex)
        p_moderate = max(0.0, remainder)

        probs = {
            "simple": round(float(p_simple), 4),
            "moderate": round(float(p_moderate), 4),
            "complex": round(float(p_complex), 4),
        }

        complexity = max(probs, key=probs.get)
        confidence = probs[complexity]

        return {
            "complexity": complexity,
            "confidence": float(confidence),
            "probabilities": probs,
        }

    def predict_batch(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict complexity for a batch of issues."""
        results: List[Dict[str, Any]] = []
        for issue in issues:
            try:
                res = self.predict_complexity(issue)
                # Add metadata passthrough if present
                res["issue_number"] = issue.get("number")
                res["repo"] = issue.get("repo")
                results.append(res)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "issue_number": issue.get("number"),
                    "repo": issue.get("repo"),
                })
        return results

    def save_prediction(self, issue: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Persist prediction to storage for dashboard and continuous learning."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "repo": issue.get("repo"),
            "issue_number": issue.get("number"),
            "title": issue.get("title"),
            "labels": issue.get("labels"),
            "comments": issue.get("comments"),
            "complexity": result.get("complexity"),
            "confidence": result.get("confidence"),
            "probabilities": result.get("probabilities"),
        }
        # Write to predictions.jsonl (dashboard)
        try:
            with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
            with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        # Also write to human_feedback.csv (continuous learning)
        try:
            os.makedirs(os.path.dirname(HUMAN_FEEDBACK_FILE), exist_ok=True)
            initialize_header = (not os.path.exists(HUMAN_FEEDBACK_FILE)) or (os.path.getsize(HUMAN_FEEDBACK_FILE) == 0)
            with open(HUMAN_FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if initialize_header:
                    writer.writerow([
                        "timestamp",
                        "repo",
                        "issue_number",
                        "title",
                        "labels",
                        "comments",
                        "predicted_complexity",
                        "human_label",
                    ])
                labels_val = issue.get("labels") or []
                if isinstance(labels_val, (list, tuple)):
                    labels_str = ",".join([str(x) if not isinstance(x, dict) else str(x.get("name", "")) for x in labels_val])
                else:
                    labels_str = str(labels_val) if labels_val is not None else ""
                writer.writerow([
                    record["timestamp"],
                    record["repo"],
                    record["issue_number"],
                    record["title"],
                    labels_str,
                    record["comments"],
                    record["complexity"],
                    "",  # human_label empty by default; can be filled later
                ])
        except Exception:
            # Silent fail to avoid breaking API response
            pass

    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read recent predictions from storage."""
        if not os.path.exists(PREDICTIONS_FILE):
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        # Return last N
        return records[-limit:]