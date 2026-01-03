"""
GitHub Issue Scraper
Collects issue data from GitHub repositories using the GitHub API with async support.
"""

import os
import yaml
import pandas as pd
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio
import logging
import torch
from ratelimit import limits, sleep_and_retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
RATE_LIMIT = 30  # GitHub's rate limit for authenticated requests (5000/hr)
PERIOD = 3600  # 1 hour in seconds
CONCURRENT_REQUESTS = 5  # Reduced concurrency to avoid rate limiting
BATCH_SIZE = 100  # Number of issues to process in each batch

class GitHubIssueScraper:
    """Scrape GitHub issues from specified repositories with async support."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the scraper with configuration."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Try to get token from config first, then fall back to environment variable
            self.gh_token = self.config['github'].get('token') or os.getenv('GITHUB_TOKEN')
            if not self.gh_token:
                raise ValueError("GitHub token not found. Please set it in config.yaml or GITHUB_TOKEN environment variable")
                
            self.headers = {
                'Authorization': f'token {self.gh_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Load repository list
            self.repos = self.config['github']['repositories']
            self.max_issues = self.config['github']['max_issues_per_repo']
            self.include_closed = self.config['github']['include_closed']
            self.min_comments = self.config['github']['min_comments']
            
            # Set up file paths
            self.output_file = self.config['data_paths']['raw_data_path']
            self.progress_file = os.path.join(
                os.path.dirname(self.output_file), 
                'scraper_progress.json'
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Initialize progress tracking
            self.progress = self._load_progress()
            
            # Check for GPU availability
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize scraper: {str(e)}")
            raise
    
    def _load_progress(self) -> dict:
        """Load scraping progress from file."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
        return {'completed_repos': [], 'last_updated': datetime.now().isoformat(), 'total_issues': 0}
    
    def _save_progress(self, data: List[dict] = None):
        """Save current scraping progress to file."""
        try:
            progress = {
                'completed_repos': self.progress.get('completed_repos', []),
                'last_updated': datetime.now().isoformat(),
                'total_issues': len(data) if data is not None else 0
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    @sleep_and_retry
    @limits(calls=RATE_LIMIT, period=PERIOD)
    async def _make_request(self, session: aiohttp.ClientSession, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make an async HTTP request with rate limiting and retry logic."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    # Handle rate limiting
                    remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                    if remaining < 10:  # If we're running low on requests
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_time = max(0, reset_time - time.time() + 5)
                        if wait_time > 0:
                            logger.warning(f"Approaching rate limit. Waiting {wait_time:.1f} seconds...")
                            await asyncio.sleep(wait_time)
                    
                    if response.status == 200:
                        return await response.json()
                    
                    # Handle rate limiting
                    if response.status == 403:
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_time = max(0, reset_time - time.time() + 5)
                        if wait_time > 0:
                            logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                            await asyncio.sleep(wait_time)
                        
                        # If we still have retries left, continue to the next iteration
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    
                    # Handle other errors
                    error_text = await response.text()
                    logger.error(f"Request failed with status {response.status} for {url}: {error_text}")
                    
                    # If we still have retries left, continue to the next iteration
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    
                    return None
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                logger.error(f"Max retries exceeded for URL: {url}")
                return None
        
        return None
    
    async def _process_repository(self, session: aiohttp.ClientSession, repo_name: str) -> List[dict]:
        """Process a single repository asynchronously."""
        logger.info(f"Starting to process repository: {repo_name}")
        issues = []
        page = 1
        per_page = 100  # Max allowed by GitHub API
        
        try:
            while len(issues) < self.max_issues:
                # Build the API URL
                state = 'all' if self.include_closed else 'open'
                url = f"https://api.github.com/repos/{repo_name}/issues"
                params = {
                    'state': state,
                    'per_page': per_page,
                    'page': page,
                    'sort': 'created',
                    'direction': 'desc'
                }
                
                logger.info(f"Fetching page {page} of issues from {repo_name}...")
                data = await self._make_request(session, url, params)
                if not data or not isinstance(data, list):
                    logger.warning(f"No more issues found or error occurred for {repo_name}")
                    break
                    
                if not data:  # No more issues
                    logger.info(f"No more issues found for {repo_name}")
                    break
                    
                for issue in data:
                    # Skip pull requests
                    if 'pull_request' in issue:
                        continue
                        
                    # Skip issues with too few comments
                    if issue.get('comments', 0) < self.min_comments:
                        continue
                        
                    try:
                        # Format the issue data
                        formatted = {
                            'number': issue.get('number', ''),
                            'repo': repo_name,  # Fixed: Use repo_name instead of undefined 'repo'
                            'title': issue.get('title', ''),
                            'body': issue.get('body', ''),
                            'labels': ', '.join(label.get('name', '') for label in issue.get('labels', [])),
                            'comments': issue.get('comments', 0),
                            'created_at': issue.get('created_at', ''),
                            'updated_at': issue.get('updated_at', ''),
                            'closed_at': issue.get('closed_at', ''),
                            'state': issue.get('state', 'unknown'),
                            'user': issue.get('user', {}).get('login', 'unknown')
                        }
                        issues.append(formatted)
                        
                        # Save progress every 20 issues
                        if len(issues) % 20 == 0:
                            logger.info(f"Collected {len(issues)}/{self.max_issues} issues from {repo_name}")
                            
                        if len(issues) >= self.max_issues:
                            logger.info(f"Reached maximum of {self.max_issues} issues for {repo_name}")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing issue {issue.get('number')} in {repo_name}: {str(e)}")
                        continue
                        
                page += 1
                
                # Add a small delay between pages to be gentle on the API
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error processing repository {repo_name}: {str(e)}", exc_info=True)
            
        return issues
    
    async def scrape_all_async(self):
        """Scrape all repositories one by one."""
        logger.info(f"Starting to scrape from {len(self.repos)} repositories")
        start_time = datetime.now()
        
        # Load existing progress if any
        self.progress = self._load_progress()
        
        # Filter out already processed repositories
        remaining_repos = [r for r in self.repos if r not in self.progress.get('completed_repos', [])]
        
        if not remaining_repos:
            logger.info("All repositories have been processed. Exiting.")
            return []
            
        logger.info(f"Processing {len(remaining_repos)} repositories one by one...")
        
        all_issues = []
        connector = aiohttp.TCPConnector(limit=1)  # Process one repository at a time
        
        async with aiohttp.ClientSession(headers=self.headers, connector=connector) as session:
            for repo in remaining_repos:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing repository: {repo}")
                logger.info(f"Remaining repositories: {len(remaining_repos) - remaining_repos.index(repo) - 1}")
                
                try:
                    # Process one repository at a time
                    repo_issues = await self._process_repository(session, repo)
                    
                    if repo_issues:
                        all_issues.extend(repo_issues)
                        # Mark repository as completed
                        self.progress.setdefault('completed_repos', []).append(repo)
                        self.progress['last_updated'] = datetime.now().isoformat()
                        
                        # Just log the progress, don't save yet
                        logger.info(f"Collected {len(repo_issues)} issues from {repo}")
                        
                        # Save progress (without the data)
                        self._save_progress()
                            
                except Exception as e:
                    logger.error(f"Error processing repository {repo}: {str(e)}", exc_info=True)
                    continue
        
        # Save all results at the end
        try:
            if all_issues:
                # Get the directory of the output file
                output_dir = os.path.dirname(self.output_file)
                
                # Load existing data if it exists
                existing_df = pd.DataFrame()
                if os.path.exists(self.output_file):
                    try:
                        existing_df = pd.read_csv(self.output_file)
                        logger.info(f"Loaded {len(existing_df)} existing issues from {self.output_file}")
                    except Exception as e:
                        logger.error(f"Error loading existing data: {str(e)}")
                
                # Create a backup of the existing file if it exists
                if not existing_df.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = os.path.join(output_dir, f"github_issues_{timestamp}.csv")
                    existing_df.to_csv(backup_file, index=False)
                    logger.info(f"Created backup at: {backup_file}")
                
                # Combine with new data and remove duplicates
                new_df = pd.DataFrame(all_issues)
                if not existing_df.empty:
                    # Only keep new issues that don't exist in the current file
                    if not new_df.empty and not existing_df.empty:
                        # Ensure 'number' and 'repo' columns are strings for comparison
                        existing_df['number'] = existing_df['number'].astype(str)
                        new_df['number'] = new_df['number'].astype(str)
                        existing_df['repo'] = existing_df['repo'].astype(str)
                        new_df['repo'] = new_df['repo'].astype(str)
                        
                        # Create a composite key for comparison
                        existing_df['composite_key'] = existing_df['repo'] + '_' + existing_df['number']
                        new_df['composite_key'] = new_df['repo'] + '_' + new_df['number']
                        
                        # Filter out duplicates
                        new_df = new_df[~new_df['composite_key'].isin(existing_df['composite_key'])]
                        new_df = new_df.drop(columns=['composite_key'])
                        existing_df = existing_df.drop(columns=['composite_key'])
                        
                        if not new_df.empty:
                            logger.info(f"Adding {len(new_df)} new issues to existing {len(existing_df)} issues")
                            df = pd.concat([existing_df, new_df], ignore_index=True)
                        else:
                            logger.info("No new issues to add")
                            df = existing_df
                    else:
                        df = existing_df
                else:
                    df = new_df
                
                # Save the combined data
                df.to_csv(self.output_file, index=False)
                
                # Print summary
                print("\n" + "="*50)
                print("SCRAPING COMPLETE")
                print("="*50)
                print(f"Total issues scraped: {len(all_issues)}")
                print("\nIssues per repository:")
                print(df['repo'].value_counts())
                print(f"\nSaved to: {self.output_file}")
                
                # Calculate and print time taken
                time_taken = (datetime.now() - start_time).total_seconds() / 60
                print(f"Time taken: {time_taken:.1f} minutes")
                
                # Clean up progress file after successful completion
                try:
                    if os.path.exists(self.progress_file):
                        os.remove(self.progress_file)
                        logger.info("Removed progress file after successful completion")
                except Exception as e:
                    logger.warning(f"Could not remove progress file: {e}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
            # Try to save to a backup file
            try:
                backup_file = f"{self.output_file}.backup_{int(time.time())}.csv"
                pd.DataFrame(all_issues).to_csv(backup_file, index=False)
                logger.error(f"Saved backup to {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to save backup: {str(backup_error)}")
                
                return pd.DataFrame(all_issues)  # Return the data even if save failed
        
        logger.warning("No issues were collected")
        return pd.DataFrame()


async def main():
    """Main function to run the scraper asynchronously."""
    try:
        start_time = time.time()
        scraper = GitHubIssueScraper()
        df = await scraper.scrape_all_async()
        
        if not df.empty:
            print(f"\n{'='*60}")
            print(f"SCRAPING COMPLETE")
            print(f"{'='*60}")
            print(f"Total issues scraped: {len(df)}")
            print(f"\nIssues per repository:")
            print(df['repo'].value_counts())
            print(f"\nSaved to: {scraper.output_file}")
            print(f"Time taken: {(time.time() - start_time)/60:.1f} minutes")
        else:
            print("\nNo issues were collected. Check the logs for errors.")
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}\nCheck scraper.log for details.")


if __name__ == "__main__":
    asyncio.run(main())