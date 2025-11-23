"""
Continuous Learning Utilities
Capture incoming issues and feedback for periodic retraining.
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any


HUMAN_FEEDBACK_FILE = os.path.join("data", "raw", "human_feedback.csv")


class ContinuousLearner:
    """Track new issues and human feedback for retraining cycles."""

    def __init__(self) -> None:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(HUMAN_FEEDBACK_FILE), exist_ok=True)
        # Initialize file with header if empty
        if not os.path.exists(HUMAN_FEEDBACK_FILE) or os.path.getsize(HUMAN_FEEDBACK_FILE) == 0:
            with open(HUMAN_FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "repo",
                    "issue_number",
                    "title",
                    "labels",
                    "comments",
                    "predicted_complexity",
                    "human_label",  # optional, can be filled later
                ])

    def process_new_issue(self, issue: Dict[str, Any], predicted_complexity: str | None = None) -> None:
        """Append a new issue record to the feedback file."""
        try:
            with open(HUMAN_FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    issue.get("repo"),
                    issue.get("number"),
                    issue.get("title"),
                    ",".join(issue.get("labels") or []),
                    issue.get("comments"),
                    predicted_complexity or "",
                    "",  # human_label left blank
                ])
        except Exception:
            # Best-effort logging; do not raise to avoid breaking webhook
            pass

    def get_learning_status(self) -> Dict[str, Any]:
        """Return simple status for the feedback dataset."""
        count = 0
        try:
            with open(HUMAN_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                # subtract header
                count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            count = 0
        return {
            "feedback_records": count,
            "file": HUMAN_FEEDBACK_FILE,
            "last_updated": datetime.now().isoformat(),
        }

    def build_consolidated_dataset(self) -> Dict[str, Any]:
        """
        Combine original training data and human feedback, deduplicate, and save consolidated CSV.
        Returns a summary dict.
        """
        try:
            from src.data.dataset_builder import DatasetBuilder
        except Exception:
            # Fallback import path if run from project root
            from data.dataset_builder import DatasetBuilder  # type: ignore

        builder = DatasetBuilder(
            raw_dir=os.path.join("data", "raw"),
            processed_dir=os.path.join("data", "processed"),
        )
        result = builder.build()
        return {
            "output_path": result.output_path,
            "total_rows_before": result.total_rows_before,
            "total_rows_after": result.total_rows_after,
            "duplicates_removed": result.duplicates_removed,
            "by_source": result.by_source,
            "by_label": result.by_label,
        }