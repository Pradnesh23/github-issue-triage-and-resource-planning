import json
import os
from typing import List, Set

DATA_DIR = os.path.join("data")
REPO_FILE = os.path.join(DATA_DIR, "monitored_repos.json")

class RepoManager:
    """Manages the list of monitored GitHub repositories."""
    
    def __init__(self):
        self._ensure_data_dir()
        self.repos = self._load_repos()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(DATA_DIR, exist_ok=True)

    def _load_repos(self) -> List[str]:
        """Load repositories from JSON file."""
        if not os.path.exists(REPO_FILE):
            return []
        try:
            with open(REPO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_repos(self):
        """Save repositories to JSON file."""
        try:
            with open(REPO_FILE, "w", encoding="utf-8") as f:
                json.dump(self.repos, f, indent=2)
        except Exception as e:
            print(f"Error saving repos: {e}")

    def get_repos(self) -> List[str]:
        """Get list of monitored repositories."""
        return sorted(self.repos)

    def add_repo(self, repo_name: str) -> bool:
        """Add a repository to the list. Returns True if added, False if already exists."""
        repo_name = repo_name.strip()
        if not repo_name:
            return False
        if repo_name not in self.repos:
            self.repos.append(repo_name)
            self._save_repos()
            return True
        return False

    def remove_repo(self, repo_name: str) -> bool:
        """Remove a repository from the list. Returns True if removed."""
        if repo_name in self.repos:
            self.repos.remove(repo_name)
            self._save_repos()
            return True
        return False
