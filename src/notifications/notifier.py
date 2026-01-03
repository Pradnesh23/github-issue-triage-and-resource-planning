"""
Enhanced Team Notification System with Rich Slack Messages
Sends team assignment notifications via logging or optional Slack webhook.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import yaml
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class TeamNotifier:
    """Notify teams based on predicted complexity with rich Slack messages."""

    COMPLEXITY_COLORS = {
        "simple": "#4CAF50",    # Green
        "moderate": "#FF9800",  # Orange
        "complex": "#F44336"    # Red
    }

    COMPLEXITY_EMOJI = {
        "simple": "🟢",
        "moderate": "🟡", 
        "complex": "🔴"
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initialize the TeamNotifier.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        notifications = self.config.get("notifications", {}) or {}
        slack_config = notifications.get("slack", {}) or {}
        
        self.enabled = slack_config.get("enabled", False)
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL") or slack_config.get("webhook_url", "")
        self.mention_on_complex = slack_config.get("mention_on_complex", True)
        self.team_assignments = notifications.get("team_assignments", {})

    def _build_slack_blocks(self, issue: Dict[str, Any], prediction: Dict[str, Any]) -> Dict:
        """
        Build rich Slack message with blocks.
        
        Args:
            issue: Issue data dictionary
            prediction: Prediction result dictionary
            
        Returns:
            Slack message payload with blocks
        """
        complexity = prediction.get("complexity", "unknown")
        confidence = prediction.get("confidence", 0)
        team = self.team_assignments.get(complexity, "unassigned")
        emoji = self.COMPLEXITY_EMOJI.get(complexity, "⚪")
        color = self.COMPLEXITY_COLORS.get(complexity, "#808080")
        
        repo = issue.get("repo", "unknown")
        number = issue.get("number", 0)
        title = issue.get("title", "Untitled")[:80]
        issue_url = f"https://github.com/{repo}/issues/{number}"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} New Issue Triaged",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{issue_url}|#{number}: {title}>*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Repository:*\n`{repo}`"},
                    {"type": "mrkdwn", "text": f"*Complexity:*\n{emoji} {complexity.upper()}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
                    {"type": "mrkdwn", "text": f"*Assign To:*\n{team}"}
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📋 View Issue", "emoji": True},
                        "url": issue_url,
                        "style": "primary"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ Triaged at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]

        # Add mention for complex issues
        text = None
        if complexity == "complex" and self.mention_on_complex:
            text = "<!channel> 🚨 New complex issue requires senior attention!"

        return {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }

    def send_assignment_notification(
        self, 
        issue: Dict[str, Any], 
        prediction: Dict[str, Any]
    ) -> bool:
        """
        Send notification; logs always, Slack if configured.
        
        Args:
            issue: Issue data dictionary
            prediction: Prediction result dictionary
            
        Returns:
            True if notification was successful
        """
        complexity = prediction.get("complexity", "unknown")
        team = self.team_assignments.get(complexity, "unassigned")
        
        # Always log
        logger.info(
            f"[Notification] Issue #{issue.get('number')} in {issue.get('repo')} - "
            f"Complexity: {complexity} ({prediction.get('confidence', 0):.0%}) - "
            f"Assign to: {team}"
        )

        # Send to Slack if enabled
        if self.enabled and self.webhook_url:
            try:
                payload = self._build_slack_blocks(issue, prediction)
                
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Slack notification sent for issue #{issue.get('number')}")
                    return True
                else:
                    logger.warning(f"Slack API error {response.status_code}: {response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                logger.warning("Slack notification timed out")
                return False
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
                return False
        
        return True

    def test_notification(self) -> bool:
        """
        Send a test notification to verify Slack setup.
        
        Returns:
            True if test was successful
        """
        test_issue = {
            "number": 999,
            "repo": "test/repository",
            "title": "Test Issue - Slack Integration Verification"
        }
        
        test_prediction = {
            "complexity": "moderate",
            "confidence": 0.85
        }
        
        return self.send_assignment_notification(test_issue, test_prediction)