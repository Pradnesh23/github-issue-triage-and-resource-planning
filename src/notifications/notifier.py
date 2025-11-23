"""
Team Notification Utilities
Send team assignment notifications via logging or optional Slack webhook.
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

import yaml
import requests


logger = logging.getLogger(__name__)


class TeamNotifier:
    """Notify teams based on predicted complexity."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        self.assignments = (self.config.get("notifications", {}) or {}).get("team_assignments", {})
        self.slack_webhook = (self.config.get("notifications", {}) or {}).get("slack_webhook_url", "")

    def _format_message(self, issue: Dict[str, Any], prediction: Dict[str, Any]) -> str:
        team = self.assignments.get(prediction.get("complexity"), "unassigned")
        return (
            f"Issue #{issue.get('number')} in {issue.get('repo')}\n"
            f"Title: {issue.get('title')}\n"
            f"Predicted Complexity: {prediction.get('complexity')} (confidence {prediction.get('confidence'):.2f})\n"
            f"Assign to team: {team}"
        )

    def send_assignment_notification(self, issue: Dict[str, Any], prediction: Dict[str, Any]) -> None:
        """Send a notification; logs by default, optional Slack if configured."""
        message = self._format_message(issue, prediction)

        # Always log
        logger.info(f"[Notification] {message}")

        # Optional Slack integration
        if self.slack_webhook:
            try:
                payload = {"text": message}
                requests.post(self.slack_webhook, json=payload, timeout=5)
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {e}")