import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DatasetBuildResult:
    output_path: str
    total_rows_before: int
    total_rows_after: int
    duplicates_removed: int
    by_source: Dict[str, int]
    by_label: Dict[str, int]


class DatasetBuilder:
    """
    Build a consolidated dataset for continuous learning by combining
    the original training data (github_issues.csv) with human feedback
    (human_feedback.csv), handling duplicates and preferring human labels.

    Final output is saved to data/processed/consolidated_issues.csv
    with normalized columns suitable for training and analysis.
    """

    def __init__(self,
                 raw_dir: str = os.path.join("data", "raw"),
                 processed_dir: str = os.path.join("data", "processed"),
                 train_filename: str = "github_issues.csv",
                 feedback_filename: str = "human_feedback.csv",
                 output_filename: str = "consolidated_issues.csv") -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.train_path = os.path.join(raw_dir, train_filename)
        self.feedback_path = os.path.join(raw_dir, feedback_filename)
        self.output_path = os.path.join(processed_dir, output_filename)

        os.makedirs(self.processed_dir, exist_ok=True)

    def _normalize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        df = df.copy()
        df["source"] = source

        # Standardize core fields
        # repo may be under 'repo' or 'repository'
        if "repo" in df.columns:
            repo_col = df["repo"]
        elif "repository" in df.columns:
            repo_col = df["repository"]
        else:
            repo_col = None
        if repo_col is None:
            df["repo"] = None
        else:
            df["repo"] = repo_col

        # Issue number may be under different names
        issue_num = None
        for cand in ["issue_number", "number", "id"]:
            if cand in df.columns:
                issue_num = df[cand]
                break
        df["issue_number"] = issue_num if issue_num is not None else None
        # Ensure numeric where possible
        try:
            df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
        except Exception:
            pass

        # Text fields
        df["title"] = df.get("title")
        df["body"] = df.get("body")

        # Labels as comma-separated string if list-like
        if "labels" in df.columns:
            labels = df["labels"]
            df["labels"] = labels.apply(lambda x: ",".join(x) if isinstance(x, (list, tuple)) else str(x))
        else:
            df["labels"] = None

        # Comments count
        df["comments"] = df.get("comments")
        try:
            df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0).astype(int)
        except Exception:
            df["comments"] = 0

        # Timestamp if present
        df["timestamp"] = df.get("timestamp")

        # Target label resolution
        target_label = None
        for cand in ["human_label", "complexity", "complexity_label", "predicted_complexity", "label"]:
            if cand in df.columns:
                tl = df[cand]
                # Normalize case
                target_label = tl.astype(str).str.upper()
                break
        df["target_label"] = target_label if target_label is not None else None

        # Reduce to a consistent schema
        cols = [
            "timestamp", "repo", "issue_number", "title", "body",
            "labels", "comments", "target_label", "source",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]

    def _make_key(self, row: pd.Series) -> str:
        repo = str(row.get("repo") or "").strip()
        num = row.get("issue_number")
        title = str(row.get("title") or "").strip().lower()
        if pd.notnull(num):
            return f"{repo}:{int(num)}"
        # Fallback to title-based key
        return f"{repo}:{title[:128]}"

    def build(self) -> DatasetBuildResult:
        # Load datasets
        train_df = pd.read_csv(self.train_path)
        feedback_df = pd.read_csv(self.feedback_path)

        norm_train = self._normalize_columns(train_df, source="original")
        norm_fb = self._normalize_columns(feedback_df, source="feedback")

        combined = pd.concat([norm_train, norm_fb], ignore_index=True)
        total_before = len(combined)

        # Ranking preferences: prefer feedback rows (human_label),
        # then rows with a target_label, then newest timestamp, then more comments.
        def pref_score(row: pd.Series) -> int:
            source_bonus = 2 if row.get("source") == "feedback" else 0
            label_bonus = 1 if pd.notnull(row.get("target_label")) else 0
            comments = row.get("comments") or 0
            # Timestamp sorting: newer preferred
            ts = row.get("timestamp")
            ts_bonus = 0
            if pd.notnull(ts):
                try:
                    ts_bonus = int(pd.to_datetime(ts).timestamp())
                except Exception:
                    ts_bonus = 0
            return source_bonus * 1_000_000_000 + label_bonus * 10_000_000 + ts_bonus + comments

        combined["_key"] = combined.apply(self._make_key, axis=1)
        combined["_score"] = combined.apply(pref_score, axis=1)

        # Keep best row per key
        combined_sorted = combined.sort_values(["_key", "_score"], ascending=[True, False])
        deduped = combined_sorted.drop_duplicates(subset=["_key"], keep="first")
        total_after = len(deduped)

        # Clean up helper columns
        deduped = deduped.drop(columns=["_key", "_score"], errors="ignore")

        # Save consolidated dataset
        deduped.to_csv(self.output_path, index=False)

        by_source = deduped["source"].fillna("unknown").value_counts().to_dict()
        by_label = deduped["target_label"].fillna("UNKNOWN").value_counts().to_dict()

        return DatasetBuildResult(
            output_path=self.output_path,
            total_rows_before=total_before,
            total_rows_after=total_after,
            duplicates_removed=total_before - total_after,
            by_source=by_source,
            by_label=by_label,
        )