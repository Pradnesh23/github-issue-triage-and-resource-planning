"""
Log Kaggle Training Results to MLflow

Usage:
    python scripts/log_kaggle_run.py --accuracy 0.53 --f1 0.54
    python scripts/log_kaggle_run.py  # Uses metrics.json
"""

import argparse
import json
import os
from datetime import datetime

import mlflow


def log_kaggle_run(
    accuracy: float = None,
    f1_weighted: float = None,
    f1_simple: float = None,
    f1_moderate: float = None,
    f1_complex: float = None,
    training_samples: int = None,
    run_name: str = None,
    use_metrics_file: bool = True
):
    """Log Kaggle training results to MLflow."""
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("github-issue-complexity")
    
    # Load from metrics.json if not provided
    if use_metrics_file and os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        accuracy = accuracy or metrics.get("accuracy")
        f1_weighted = f1_weighted or metrics.get("f1_weighted")
        f1_simple = f1_simple or metrics.get("f1_simple")
        f1_moderate = f1_moderate or metrics.get("f1_moderate")
        f1_complex = f1_complex or metrics.get("f1_complex")
        training_samples = training_samples or metrics.get("training_samples")
    
    # Generate run name
    run_name = run_name or f"kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model_name": "distilbert-base-uncased",
            "training_platform": "Kaggle",
            "model_type": "distilbert-xgboost-stacking",
            "training_samples": training_samples or 0,
            "xgb_n_estimators": 500,
            "xgb_learning_rate": 0.03,
            "xgb_max_depth": 4,
            "smote_applied": True
        })
        
        # Log metrics
        logged_metrics = {}
        if accuracy is not None:
            logged_metrics["accuracy"] = accuracy
        if f1_weighted is not None:
            logged_metrics["f1_weighted"] = f1_weighted
        if f1_simple is not None:
            logged_metrics["f1_simple"] = f1_simple
        if f1_moderate is not None:
            logged_metrics["f1_moderate"] = f1_moderate
        if f1_complex is not None:
            logged_metrics["f1_complex"] = f1_complex
        
        if logged_metrics:
            mlflow.log_metrics(logged_metrics)
        
        # Log metrics.json as artifact
        if os.path.exists("metrics.json"):
            mlflow.log_artifact("metrics.json")
        
        # Log notebook if exists
        notebook_path = "notebooks/github-issue-predictor.ipynb"
        if os.path.exists(notebook_path):
            mlflow.log_artifact(notebook_path, "notebooks")
        
        print(f"✅ Logged run: {run_name}")
        print(f"   Accuracy: {accuracy}")
        print(f"   F1 Weighted: {f1_weighted}")
        print(f"   View at: http://localhost:5000")


def main():
    parser = argparse.ArgumentParser(description="Log Kaggle training results to MLflow")
    parser.add_argument("--accuracy", type=float, help="Model accuracy")
    parser.add_argument("--f1", "--f1-weighted", type=float, dest="f1_weighted", help="Weighted F1 score")
    parser.add_argument("--f1-simple", type=float, help="F1 for Simple class")
    parser.add_argument("--f1-moderate", type=float, help="F1 for Moderate class")
    parser.add_argument("--f1-complex", type=float, help="F1 for Complex class")
    parser.add_argument("--samples", type=int, dest="training_samples", help="Training samples")
    parser.add_argument("--name", dest="run_name", help="Run name")
    parser.add_argument("--no-file", action="store_true", help="Don't read from metrics.json")
    
    args = parser.parse_args()
    
    log_kaggle_run(
        accuracy=args.accuracy,
        f1_weighted=args.f1_weighted,
        f1_simple=args.f1_simple,
        f1_moderate=args.f1_moderate,
        f1_complex=args.f1_complex,
        training_samples=args.training_samples,
        run_name=args.run_name,
        use_metrics_file=not args.no_file
    )


if __name__ == "__main__":
    main()
