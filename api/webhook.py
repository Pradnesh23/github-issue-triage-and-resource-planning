"""
GitHub Webhook Handler for Real-time Issue Complexity Prediction
Handles incoming GitHub webhook events and triggers complexity predictions.
"""

import hashlib
import hmac
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import yaml
import os

from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel
from github import Github

from src.models.predict import GitHubIssuePredictor
from src.models.continuous_learning import ContinuousLearner
from src.notifications.notifier import TeamNotifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI router
router = APIRouter(prefix="/webhook")

# Initialize components
predictor = GitHubIssuePredictor()
learner = ContinuousLearner()
notifier = TeamNotifier()

# Get GitHub webhook secret and token from config
WEBHOOK_SECRET = config.get("github", {}).get("webhook_secret", "")
GITHUB_TOKEN = config.get("github", {}).get("token", "")

# Initialize GitHub client
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not secret:
        logger.warning("No webhook secret configured. Skipping signature verification.")
        return True
    
    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    try:
        signature_sha256 = signature.split("=")[1]
        return hmac.compare_digest(expected, signature_sha256)
    except (IndexError, ValueError):
        return False

@router.post("/github")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None)
):
    """Handle GitHub webhook events."""
    try:
        # Get request body
        body = await request.body()
        
        # Verify signature if secret is configured
        if WEBHOOK_SECRET:
            if not x_hub_signature_256:
                raise HTTPException(status_code=400, detail="Missing signature")
            
            if not verify_signature(body, x_hub_signature_256, WEBHOOK_SECRET):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse JSON payload
        payload = await request.json()
        
        # Handle different event types
        if x_github_event == "issues":
            return await handle_issue_event(payload)
        elif x_github_event == "issue_comment":
            return await handle_issue_comment_event(payload)
        else:
            logger.info(f"Unhandled GitHub event type: {x_github_event}")
            return {"status": "ignored", "event_type": x_github_event}
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_issue_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GitHub issues event."""
    action = payload.get("action")
    issue = payload.get("issue", {})
    repository = payload.get("repository", {})
    
    logger.info(f"Processing issue event: {action} for issue #{issue.get('number')} in {repository.get('full_name')}")
    
    # Only process certain actions
    if action not in ["opened", "edited", "reopened"]:
        return {
            "status": "ignored",
            "action": action,
            "reason": "Action not relevant for complexity prediction"
        }
    
    # Extract issue data
    issue_data = {
        "number": issue.get("number"),
        "repo": repository.get("full_name"),
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "labels": [label.get("name", "") for label in issue.get("labels", [])],
        "comments": issue.get("comments", 0),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "author": issue.get("user", {}).get("login") if issue.get("user") else None
    }
    
    try:
        # Make prediction
        prediction_result = predictor.predict_complexity(issue_data)
        
        # Log the prediction
        logger.info(f"Issue prediction - Repo: {issue_data['repo']}, "
                   f"Issue: #{issue_data['number']}, Complexity: {prediction_result['complexity']}")

        # Persist prediction and add to continuous learning
        try:
            predictor.save_prediction(issue_data, prediction_result)
        except Exception:
            pass
        
        # Add to continuous learning
        try:
            learner.process_new_issue(issue_data, predicted_complexity=prediction_result.get("complexity"))
        except Exception:
            pass
        
        # Send notification to appropriate team
        notifier.send_assignment_notification(issue_data, prediction_result)
        
        # Automatically label the GitHub issue based on complexity prediction
        complexity_label_added = False
        if github_client and GITHUB_TOKEN:
            try:
                # Get the repository and issue objects
                repo = github_client.get_repo(repository.get("full_name"))
                gh_issue = repo.get_issue(issue.get("number"))
                
                # Create complexity label
                complexity_label = f"complexity:{prediction_result['complexity']}"
                
                # Add the label to the issue
                gh_issue.add_to_labels(complexity_label)
                complexity_label_added = True
                logger.info(f"Added label '{complexity_label}' to issue #{issue_data['number']} in {issue_data['repo']}")
            except Exception as e:
                logger.error(f"Error adding label to GitHub issue: {e}")
        
        # In a real implementation, you might:
        # 1. Store the prediction in a database
        # 2. Send notifications to relevant team members
        # 3. Update project management tools
        # 4. Trigger resource allocation processes
        # 5. Add labels to the GitHub issue
        
        return {
            "status": "processed",
            "action": action,
            "issue": {
                "number": issue_data["number"],
                "repo": issue_data["repo"]
            },
            "prediction": prediction_result,
            "complexity_label_added": complexity_label_added,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error predicting issue complexity: {e}")
        return {
            "status": "error",
            "action": action,
            "issue": {
                "number": issue_data["number"],
                "repo": issue_data["repo"]
            },
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def handle_issue_comment_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GitHub issue comment event."""
    action = payload.get("action")
    issue = payload.get("issue", {})
    comment = payload.get("comment", {})
    repository = payload.get("repository", {})
    
    logger.info(f"Processing issue comment event: {action} for issue #{issue.get('number')} in {repository.get('full_name')}")
    
    # We might want to re-evaluate issue complexity when comments are added
    if action == "created" and issue.get("comments", 0) > 0:
        # Extract issue data
        issue_data = {
            "number": issue.get("number"),
            "repo": repository.get("full_name"),
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
            "labels": [label.get("name", "") for label in issue.get("labels", [])],
            "comments": issue.get("comments", 0),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "closed_at": issue.get("closed_at"),
            "author": issue.get("user", {}).get("login") if issue.get("user") else None
        }
        
        try:
            # Make prediction
            prediction_result = predictor.predict_complexity(issue_data)
            
            # Log the prediction
            logger.info(f"Issue comment prediction update - Repo: {issue_data['repo']}, "
                       f"Issue: #{issue_data['number']}, Complexity: {prediction_result['complexity']}")

            # Persist and add to learning
            try:
                predictor.save_prediction(issue_data, prediction_result)
            except Exception:
                pass
            
            # Add to continuous learning
            try:
                learner.process_new_issue(issue_data, predicted_complexity=prediction_result.get("complexity"))
            except Exception:
                pass
            
            # Automatically update the GitHub issue label based on complexity prediction
            complexity_label_added = False
            if github_client and GITHUB_TOKEN:
                try:
                    # Get the repository and issue objects
                    repo = github_client.get_repo(repository.get("full_name"))
                    gh_issue = repo.get_issue(issue.get("number"))
                    
                    # Remove existing complexity labels
                    existing_labels = [label.name for label in gh_issue.get_labels()]
                    complexity_labels = [label for label in existing_labels if label.startswith("complexity:")]
                    for label in complexity_labels:
                        gh_issue.remove_from_labels(label)
                    
                    # Create and add new complexity label
                    complexity_label = f"complexity:{prediction_result['complexity']}"
                    gh_issue.add_to_labels(complexity_label)
                    complexity_label_added = True
                    logger.info(f"Updated label to '{complexity_label}' for issue #{issue_data['number']} in {issue_data['repo']}")
                except Exception as e:
                    logger.error(f"Error updating label for GitHub issue: {e}")
            
            return {
                "status": "processed",
                "action": action,
                "issue": {
                    "number": issue_data["number"],
                    "repo": issue_data["repo"]
                },
                "prediction": prediction_result,
                "complexity_label_updated": complexity_label_added,
                "comment_id": comment.get("id"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating issue complexity prediction: {e}")
            return {
                "status": "error",
                "action": action,
                "issue": {
                    "number": issue_data["number"],
                    "repo": issue_data["repo"]
                },
                "error": str(e),
                "comment_id": comment.get("id"),
                "timestamp": datetime.now().isoformat()
            }
    
    return {
        "status": "ignored",
        "action": action,
        "reason": "Comment event not relevant for complexity prediction"
    }

@router.get("/status")
async def webhook_status():
    """Get webhook handler status."""
    status = learner.get_learning_status()
    return {
        "webhook": "active",
        "continuous_learning": status,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/test")
async def test_webhook(payload: Dict[str, Any]):
    """Test endpoint for webhook functionality."""
    logger.info("Webhook test endpoint called")
    return {
        "status": "test_received",
        "payload": payload,
        "timestamp": datetime.now().isoformat()
    }