"""
FastAPI Application for GitHub Issue Complexity Prediction
Provides REST API endpoints for predicting GitHub issue complexity.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import yaml
import os
import logging
from datetime import datetime

from src.models.predict import GitHubIssuePredictor
from api.webhook import router as webhook_router
from src.models.continuous_learning import ContinuousLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="GitHub Issue Complexity Predictor",
    description="API for predicting the complexity of GitHub issues",
    version="1.0.0"
)

# Initialize predictor and continuous learner
predictor = GitHubIssuePredictor()
learner = ContinuousLearner()

# Include webhook router
app.include_router(webhook_router)

# Pydantic models for request/response
class GitHubIssue(BaseModel):
    """Model for GitHub issue data."""
    number: Optional[int] = None
    repo: Optional[str] = None
    title: str
    body: Optional[str] = ""
    labels: Optional[List[str]] = []
    comments: Optional[int] = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    author: Optional[str] = None

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    complexity: str
    confidence: float
    probabilities: Dict[str, float]
    issue_number: Optional[int] = None
    repo: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    predictions: List[PredictionResponse]
    timestamp: str

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GitHub Issue Complexity Predictor API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_complexity(issue: GitHubIssue):
    """Predict complexity of a single GitHub issue."""
    try:
        # Convert issue to dictionary
        issue_dict = issue.dict()
        
        # Make prediction
        result = predictor.predict_complexity(issue_dict)
        
        # Add issue metadata to response
        result["issue_number"] = issue.number
        result["repo"] = issue.repo

        # Persist prediction for dashboard
        try:
            predictor.save_prediction(issue_dict, result)
        except Exception:
            pass
        
        return result
    except Exception as e:
        logger.error(f"Error predicting complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_complexity(issues: List[GitHubIssue]):
    """Predict complexity of multiple GitHub issues."""
    try:
        # Convert issues to dictionaries
        issues_dicts = [issue.dict() for issue in issues]
        
        # Make batch prediction
        results = predictor.predict_batch(issues_dicts)
        
        # Convert results to response format
        predictions = []
        for result in results:
            if "error" not in result:
                predictions.append(PredictionResponse(**result))
            else:
                # Handle errors in individual predictions
                predictions.append(PredictionResponse(
                    complexity="unknown",
                    confidence=0.0,
                    probabilities={"simple": 0.0, "moderate": 0.0, "complex": 0.0},
                    issue_number=result.get("issue_number"),
                    repo=result.get("repo")
                ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error predicting batch complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/github", response_model=Dict[str, Any])
async def github_webhook(payload: Dict[str, Any]):
    """Handle GitHub webhook events for new issues."""
    try:
        # Extract issue data from webhook payload
        if "issue" in payload:
            issue_data = payload["issue"]
            
            # Add repository information
            if "repository" in payload:
                issue_data["repo"] = payload["repository"].get("full_name")
            
            # Make prediction
            result = predictor.predict_complexity(issue_data)
            
            # Log the prediction
            logger.info(f"New issue prediction - Repo: {issue_data.get('repo')}, "
                       f"Issue: {issue_data.get('number')}, Complexity: {result['complexity']}")
            
            # In a real implementation, you would:
            # 1. Store the prediction in a database
            # 2. Send notifications to relevant team members
            # 3. Update project management tools
            # 4. Trigger resource allocation processes
            
            return {
                "status": "processed",
                "prediction": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid webhook payload")
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/recent")
async def recent_predictions(limit: int = 100):
    """Return recent predictions persisted for the dashboard."""
    try:
        records = predictor.get_recent_predictions(limit=limit)
        return {"records": records, "count": len(records), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error reading recent predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/consolidate")
async def dataset_consolidate():
    """Build consolidated dataset from original + human feedback and return summary."""
    try:
        summary = learner.build_consolidated_dataset()
        return {"status": "completed", **summary, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error consolidating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/stats")
async def dataset_stats():
    """Return feedback dataset status and consolidated dataset presence."""
    try:
        status = learner.get_learning_status()
        consolidated_path = os.path.join("data", "processed", "consolidated_issues.csv")
        exists = os.path.exists(consolidated_path)
        size = os.path.getsize(consolidated_path) if exists else 0
        return {
            **status,
            "consolidated_exists": exists,
            "consolidated_path": consolidated_path,
            "consolidated_size": size,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error retrieving dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Get server configuration
    host = config["api"]["host"]
    port = config["api"]["port"]
    reload = config["api"]["reload"]
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )