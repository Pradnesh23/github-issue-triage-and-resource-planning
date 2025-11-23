"""
Pydantic Schemas for GitHub Issue Complexity Prediction API
Defines data models for API request/response validation.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class GitHubIssueBase(BaseModel):
    """Base model for GitHub issue data."""
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

class GitHubIssueCreate(GitHubIssueBase):
    """Model for creating a GitHub issue."""
    pass

class GitHubIssueUpdate(GitHubIssueBase):
    """Model for updating a GitHub issue."""
    pass

class GitHubIssueInDB(GitHubIssueBase):
    """Model for GitHub issue as stored in database."""
    id: int
    created_date: datetime
    updated_date: datetime

    class Config:
        orm_mode = True

class PredictionResult(BaseModel):
    """Model for prediction result."""
    complexity: str
    confidence: float
    probabilities: Dict[str, float]

class PredictionResponse(PredictionResult):
    """Model for prediction response."""
    issue_number: Optional[int] = None
    repo: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    predictions: List[PredictionResponse]
    timestamp: str

class WebhookPayload(BaseModel):
    """Model for GitHub webhook payload."""
    action: Optional[str] = None
    issue: Optional[Dict[str, Any]] = None
    repository: Optional[Dict[str, Any]] = None
    sender: Optional[Dict[str, Any]] = None

class WebhookResponse(BaseModel):
    """Model for webhook response."""
    status: str
    prediction: Optional[PredictionResult] = None
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: str = datetime.now().isoformat()