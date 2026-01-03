"""
AI Solution Advisor using Google Gemini.
Analyzes GitHub issues and suggests technical resolution plans.
"""

import os
import logging
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SolutionAdvisor:
    """Uses Google Gemini to provide technical insights for GitHub issues."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Solution Advisor will be disabled.")
            self.model = None
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Solution Advisor initialized with Gemini 1.5 Flash")
        except Exception as e:
            logger.error(f"Failed to initialize Solution Advisor: {e}")
            self.model = None

    def generate_solution(self, title: str, body: str, labels: List[str]) -> str:
        """
        Generate a technical resolution plan for the given issue.
        """
        if not self.model:
            return "⚠️ AI Advisor not configured. Please set GEMINI_API_KEY in .env file."
        
        prompt = f"""
        Act as a senior software engineer acting as a mentor. Analyze this GitHub issue strategy only.
        
        Issue Title: {title}
        Labels: {', '.join(labels)}
        
        Issue Body:
        {body[:3000]}  # Truncate to avoid token limits if extremely long
        
        Provide a concise technical response in Markdown format:
        
        ### 🔍 Root Cause Analysis (Hypothetical)
        [Brief analysis of what might be wrong]
        
        ### 🛠 Suggested Fix Steps
        1. [Step 1]
        2. [Step 2]
        ...
        
        ### 💻 Code Hint
        ```python
        # Illustrative snippet (if applicable)
        ```
        
        ### ⚠️ Risks
        [Potential side effects or interactions]
        """
        
        try:
            logger.info(f"Generating solution for issue: {title}")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"❌ Error generating solution: {str(e)}"
