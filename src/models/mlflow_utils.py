"""
MLflow Utilities for Experiment Tracking and Model Registry
Provides centralized tracking for model training experiments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry utilities."""
    
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "github-issue-complexity"
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (default: local SQLite)
            experiment_name: Name of the experiment to track
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.run = None
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Run ID string
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run.info.run_id})")
        return self.run.info.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {metrics}")
    
    def log_model_pytorch(
        self,
        model,
        artifact_path: str = "model",
        registered_name: Optional[str] = None
    ) -> None:
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path in artifact store
            registered_name: Name to register model under (optional)
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_name
        )
        logger.info(f"Logged PyTorch model to {artifact_path}")
    
    def log_model_sklearn(
        self,
        model,
        artifact_path: str = "model",
        registered_name: Optional[str] = None
    ) -> None:
        """
        Log sklearn/XGBoost model.
        
        Args:
            model: sklearn model to log
            artifact_path: Path in artifact store
            registered_name: Name to register model under (optional)
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_name
        )
        logger.info(f"Logged sklearn model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file as artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        """Log a dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags."""
        mlflow.set_tags(tags)
    
    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str
    ) -> str:
        """
        Register a model in the model registry.
        
        Args:
            run_id: Run ID containing the model
            artifact_path: Path to model in artifacts
            model_name: Name to register under
            
        Returns:
            Model version string
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str  # "Staging", "Production", "Archived"
    ) -> None:
        """Transition model version to a different stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_production_model_uri(self, model_name: str) -> Optional[str]:
        """Get the URI of the production model."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return f"models:/{model_name}/Production"
            return None
        except Exception as e:
            logger.warning(f"Could not get production model: {e}")
            return None
    
    def save_metrics_json(
        self, 
        metrics: Dict[str, float], 
        path: str = "metrics.json"
    ) -> None:
        """
        Save metrics to JSON file (for DVC tracking).
        
        Args:
            metrics: Dictionary of metrics
            path: Output file path
        """
        metrics_with_timestamp = {
            **metrics,
            "timestamp": datetime.now().isoformat()
        }
        with open(path, "w") as f:
            json.dump(metrics_with_timestamp, f, indent=2)
        logger.info(f"Saved metrics to {path}")


def get_tracker(
    tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "github-issue-complexity"
) -> MLflowTracker:
    """
    Factory function to get MLflow tracker instance.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Experiment name
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)
