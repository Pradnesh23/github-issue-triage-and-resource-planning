"""
Streamlit Web Interface for GitHub Issue Complexity Predictor
Interactive dashboard for predicting and visualizing GitHub issue complexity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yaml
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any
from github import Github

from src.models.solution_advisor import SolutionAdvisor

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# API configuration
API_BASE_URL = f"http://localhost:{config['api']['port']}"

# Set up Streamlit page
st.set_page_config(
    page_title="GitHub Issue Complexity Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .prediction-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .complexity-simple { background-color: #E8F5E9; border-left: 4px solid #4CAF50; }
    .complexity-moderate { background-color: #FFF3E0; border-left: 4px solid #FF9800; }
    .complexity-complex { background-color: #FFEBEE; border-left: 4px solid #F44336; }
</style>
""", unsafe_allow_html=True)

def suggest_remedy(complexity: str, confidence: float, labels: List[str], comments: int) -> str:
    """Return a concise remedy suggestion based on predicted complexity and context."""
    labels = labels or []
    confidence_pct = f"{(confidence or 0) * 100:.0f}%"
    if complexity == "simple":
        return (
            f"Good first issue. Assign junior dev; ETA 1–2 days. "
            f"Add labels: ['complexity:simple', 'good first issue']. Confidence {confidence_pct}."
        )
    if complexity == "moderate":
        needs_design = any(l.lower() in {"feature", "enhancement", "design"} for l in labels)
        return (
            f"Assign mid-level dev; ETA 2–5 days. "
            f"{'Schedule quick design review. ' if needs_design else ''}"
            f"Add labels: ['complexity:moderate', 'needs triage']. Confidence {confidence_pct}."
        )
    if complexity == "complex":
        escalation_note = "High comments, consider escalation. " if (comments or 0) >= 10 else ""
        return (
            f"Assign senior dev; plan spike + breakdown into subtasks; ETA 1–2 weeks. "
            f"{escalation_note}Add labels: ['complexity:complex', 'needs investigation', 'priority:high']. "
            f"Confidence {confidence_pct}."
        )
    return f"Review needed; insufficient data. Confidence {confidence_pct}."

def get_api_status() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_complexity(issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send issue data to API for prediction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=issue_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def predict_batch_complexity(issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send batch of issues to API for prediction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=issues_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_recent_predictions(limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch recent predictions from the API for the dashboard."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions/recent",
            params={"limit": limit},
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("records", [])
        else:
            return []
    except Exception:
        return []

def display_prediction_result(result: Dict[str, Any], issue_title: str = ""):
    """Display prediction result in a formatted card."""
    if not result:
        return
    
    complexity = result["complexity"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    
    # Determine styling based on complexity
    style_class = f"prediction-card complexity-{complexity}"
    
    # Display the prediction card
    st.markdown(f"""
    <div class="{style_class}">
        <h3>{issue_title or 'Issue Prediction'}</h3>
        <h4>Complexity: <span style="color: {'#4CAF50' if complexity == 'simple' else '#FF9800' if complexity == 'moderate' else '#F44336'}">{complexity.title()}</span></h4>
        <p>Confidence: {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability breakdown
    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({
        "Complexity": list(probabilities.keys()),
        "Probability": list(probabilities.values())
    })
    fig = px.bar(
        prob_df,
        x="Complexity",
        y="Probability",
        color="Complexity",
        color_discrete_map={
            "simple": "#4CAF50",
            "moderate": "#FF9800",
            "complex": "#F44336"
        }
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application."""
    st.title("📊 GitHub Issue Complexity Predictor")
    st.markdown("Predict the complexity of GitHub issues to help with resource planning and task assignment.")
    
    # Check API status
    api_status = get_api_status()
    if not api_status:
        st.warning("⚠️ API is not running. Please start the API server for full functionality.")
    
    # Create tabs for different functionalities
    tab_conn, tab1, tab2, tab3, tab4 = st.tabs([
        "Connect GitHub",
        "Single Issue Prediction", 
        "Batch Prediction", 
        "Webhook Setup", 
        "Dashboard"
    ])

    with tab_conn:
        st.header("Connect GitHub Repository")
        token = st.text_input("GitHub Personal Access Token", type="password")
        if token:
            st.session_state["gh_token"] = token
        has_token = bool(st.session_state.get("gh_token"))
        if has_token:
            try:
                gh = Github(st.session_state["gh_token"])
                user = gh.get_user()
                repos = list(user.get_repos())
                repo_names = [r.full_name for r in repos]
                selected_repo = st.selectbox("Select Repository", options=repo_names)
                st.session_state["selected_repo"] = selected_repo
                if st.button("Fetch Open Issues and Run Triage"):
                    with st.spinner("Fetching issues and predicting complexity..."):
                        repo = gh.get_repo(selected_repo)
                        issues = list(repo.get_issues(state="open"))
                        issues_data = []
                        for it in issues:
                            labels = [lb.name for lb in it.get_labels()]
                            issues_data.append({
                                "title": it.title or "",
                                "body": it.body or "",
                                "repo": selected_repo,
                                "number": it.number,
                                "labels": labels
                            })
                        if api_status and issues_data:
                            batch_result = predict_batch_complexity(issues_data)
                            if batch_result and batch_result.get("predictions"):
                                results_df = pd.DataFrame(batch_result["predictions"])
                                st.dataframe(results_df, use_container_width=True)
                                apply = st.checkbox("Apply complexity labels to GitHub")
                                if apply:
                                    for _, row in results_df.iterrows():
                                        try:
                                            gh_issue = repo.get_issue(int(row.get("issue_number") or 0))
                                            gh_issue.add_to_labels(f"complexity:{row.get('complexity')}")
                                        except Exception:
                                            pass
                        else:
                            st.info("No open issues found or API is unavailable.")
            except Exception as e:
                st.error(str(e))
    
    with tab1:
        st.header("Predict Complexity of a Single Issue")
        
        # Issue input form
        with st.form("issue_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Issue Title", placeholder="Enter the issue title")
                repo = st.text_input("Repository", placeholder="owner/repo (optional)")
                author = st.text_input("Author", placeholder="GitHub username (optional)")
                
            with col2:
                labels = st.text_input("Labels", placeholder="bug, feature, enhancement (comma-separated)")
            
            body = st.text_area("Issue Description", placeholder="Enter the full issue description", height=200)
            
            submitted = st.form_submit_button("Predict Complexity")
        
        if submitted and title:
            # Prepare issue data
            issue_data = {
                "title": title,
                "body": body,
                "repo": repo,
                "author": author,
                "labels": [label.strip() for label in labels.split(",") if label.strip()],
            }
            
            # Make prediction
            if api_status:
                with st.spinner("Analyzing issue complexity..."):
                    result = predict_complexity(issue_data)
                    if result:
                        display_prediction_result(result, title)
                        
                        # Store for AI Advisor
                        st.session_state['current_issue'] = issue_data
            else:
                st.info("API is not available. Showing example prediction:")
                # Example result for demonstration
                example_result = {
                    "complexity": "moderate",
                    "confidence": 0.78,
                    "probabilities": {
                        "simple": 0.15,
                        "moderate": 0.78,
                        "complex": 0.07
                    }
                }
                display_prediction_result(example_result, title)
                st.session_state['current_issue'] = issue_data
        
        elif submitted:
            st.warning("Please enter at least an issue title.")

        # AI Solution Section
        if 'current_issue' in st.session_state:
            st.markdown("---")
            st.subheader("🤖 AI Solution Advisor")
            
            if st.button("Generate Technical Resolution Plan"):
                issue = st.session_state['current_issue']
                advisor = SolutionAdvisor()
                
                if not advisor.model:
                    st.warning("⚠️ GEMINI_API_KEY not set. Cannot generate solution.")
                else:
                    with st.spinner("consulting Gemini 1.5 Flash..."):
                        solution = advisor.generate_solution(
                            issue['title'], 
                            issue['body'], 
                            issue['labels']
                        )
                        st.markdown(solution)
    
    with tab2:
        st.header("Batch Prediction for Multiple Issues")
        st.markdown("Upload a CSV file with multiple issues for batch prediction.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded CSV
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} issues from the file.")
                st.dataframe(df.head())
                
                # Check required columns
                required_columns = ['title']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    if st.button("Predict Complexity for All Issues"):
                        # Convert DataFrame to list of issues
                        issues_data = []
                        for _, row in df.iterrows():
                            issue_data = {
                                "title": row["title"],
                                "body": row.get("body", ""),
                                "repo": row.get("repo"),
                                "author": row.get("author"),
                                "labels": row.get("labels", "").split(",") if row.get("labels") else [],
                                "comments": row.get("comments", 0)
                            }
                            issues_data.append(issue_data)
                        
                        # Make batch prediction
                        if api_status:
                            with st.spinner("Analyzing issues... This may take a moment."):
                                batch_result = predict_batch_complexity(issues_data)
                                if batch_result:
                                    st.success(f"Processed {len(batch_result['predictions'])} issues!")
                                    
                                    # Display results
                                    results_df = pd.DataFrame(batch_result["predictions"])
                                    st.dataframe(results_df)
                                    
                                    # Show summary statistics
                                    st.subheader("Complexity Distribution")
                                    complexity_counts = results_df["complexity"].value_counts()
                                    fig = px.pie(
                                        values=complexity_counts.values,
                                        names=complexity_counts.index,
                                        color=complexity_counts.index,
                                        color_discrete_map={
                                            "simple": "#4CAF50",
                                            "moderate": "#FF9800",
                                            "complex": "#F44336"
                                        }
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("API is not available. Batch prediction requires the API to be running.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Webhook Setup for Real-time Integration")
        st.markdown("Configure GitHub webhooks to automatically analyze new issues as they're created.")
        public_url = st.text_input("Public Server URL", placeholder="https://your-domain")
        payload_url = f"{public_url.rstrip('/')}/webhook/github" if public_url else f"http://{config['api']['host']}:{config['api']['port']}/webhook/github"
        st.code(f"Payload URL: {payload_url}")
        st.markdown("""
        1. Go to your GitHub repository settings
        2. Navigate to Webhooks → Add webhook
        3. Content type: application/json
        4. Events: Issues and Issue comments
        5. Active: checked
        """)
        
        st.info("Note: You'll need to deploy this application to a public server for GitHub to reach your webhook endpoint.")
        
        # Display webhook status
        if api_status:
            st.success("✅ API is running and ready to receive webhook events")
        else:
            st.warning("⚠️ API is not running. Start the API server to enable webhook functionality")
    
    with tab4:
        st.header("Resource Planning Dashboard")
        st.markdown("Visualize issue complexity distribution and resource allocation recommendations.")

        # Auto-refresh controls
        col_refresh, col_interval = st.columns([1, 1])
        with col_refresh:
            auto_refresh = st.checkbox("Auto-refresh", value=True, help="Automatically refresh the live feed")
        with col_interval:
            refresh_interval = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=10)
        # Try to enable auto-refresh if available
        if auto_refresh:
            try:
                # Prefer built-in st_autorefresh if available
                if hasattr(st, "autorefresh"):
                    st.autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh")
                else:
                    # Some environments provide st_autorefresh via component
                    from streamlit_autorefresh import st_autorefresh  # type: ignore
                    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh_component")
            except Exception:
                pass
        
        # Prefer real data from API if available
        dashboard_df = None
        if api_status:
            records = get_recent_predictions(limit=300)
            if records:
                try:
                    dashboard_df = pd.DataFrame(records)
                except Exception:
                    dashboard_df = None
        
        if dashboard_df is None:
            # Sample data for demonstration
            sample_data = pd.DataFrame({
                "Repository": ["tensorflow/tensorflow", "pytorch/pytorch", "scikit-learn/scikit-learn"] * 10,
                "Complexity": np.random.choice(["simple", "moderate", "complex"], 30, p=[0.4, 0.4, 0.2]),
                "Issues": np.random.randint(1, 50, 30)
            })
            dashboard_df = sample_data
        
        # Repository filter
        st.subheader("Repository Filter")
        selected_repo = st.session_state.get("selected_repo")
        if dashboard_df is not None and "repo" in dashboard_df.columns:
            repos_list = sorted(list({r for r in dashboard_df["repo"].dropna().unique()}))
            default_index = repos_list.index(selected_repo) if selected_repo in repos_list else 0 if repos_list else None
            if repos_list:
                repo_choice = st.selectbox("Filter by repository", options=repos_list, index=default_index if default_index is not None else 0)
                dashboard_df = dashboard_df[dashboard_df["repo"] == repo_choice]

        # Complexity distribution chart
        st.subheader("Complexity Distribution Across Repositories")
        if {"repo", "complexity"}.issubset(set(dashboard_df.columns)):
            complexity_dist = dashboard_df.groupby(["repo", "complexity"]).size().reset_index(name="Count")
            x_col = "repo"
            color_col = "complexity"
        else:
            complexity_dist = dashboard_df.groupby(["Repository", "Complexity"]).size().reset_index(name="Count")
            x_col = "Repository"
            color_col = "Complexity"
        fig = px.bar(
            complexity_dist,
            x=x_col,
            y="Count",
            color=color_col,
            barmode="group",
            color_discrete_map={
                "simple": "#4CAF50",
                "moderate": "#FF9800",
                "complex": "#F44336"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Resource allocation recommendations
        st.subheader("Resource Allocation Recommendations")
        
        # Calculate distribution percentages
        if "complexity" in dashboard_df.columns:
            complexity_counts = dashboard_df["complexity"].value_counts()
            total_issues = len(dashboard_df)
        else:
            complexity_counts = dashboard_df["Complexity"].value_counts()
            total_issues = len(dashboard_df)
        simple_pct = (complexity_counts.get("simple", 0) / total_issues) * 100
        moderate_pct = (complexity_counts.get("moderate", 0) / total_issues) * 100
        complex_pct = (complexity_counts.get("complex", 0) / total_issues) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Simple Issues", f"{simple_pct:.1f}%", "40% target")
            st.markdown("""
            - Can be assigned to junior developers
            - Estimated effort: 1-2 days
            - Sprint planning candidates
            """)
        
        with col2:
            st.metric("Moderate Issues", f"{moderate_pct:.1f}%", "40% target")
            st.markdown("""
            - Require experienced developers
            - Estimated effort: 2-5 days
            - May need pair programming
            """)
        
        with col3:
            st.metric("Complex Issues", f"{complex_pct:.1f}%", "20% target")
            st.markdown("""
            - Need senior developers
            - Estimated effort: 1-2 weeks
            - Require team collaboration
            """)
        
        # Team distribution suggestion
        st.subheader("Team Distribution Suggestion")
        
        # Create a more detailed resource allocation plan
        st.markdown("""
        ### Recommended Team Structure:
        - **Junior Developers (40% capacity)**: Handle simple issues
        - **Mid-level Developers (40% capacity)**: Handle moderate issues
        - **Senior Developers (20% capacity)**: Handle complex issues
        
        ### Workload Distribution:
        """)
        
        # Create workload distribution chart
        workload_data = pd.DataFrame({
            "Developer Level": ["Junior", "Mid-level", "Senior"],
            "Capacity (%)": [40, 40, 20],
            "Issue Types": ["Simple", "Moderate", "Complex"]
        })
        
        fig2 = px.pie(
            workload_data,
            values="Capacity (%)",
            names="Developer Level",
            color="Developer Level",
            color_discrete_map={
                "Junior": "#4CAF50",
                "Mid-level": "#FF9800",
                "Senior": "#F44336"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Escalation paths
        st.subheader("Escalation Paths")
        st.markdown("""
        1. **Simple Issues**: Junior → Mid-level (if blocked > 2 days)
        2. **Moderate Issues**: Mid-level → Senior (if blocked > 5 days)
        3. **Complex Issues**: Senior → Tech Lead (if blocked > 1 week)

        ### Sprint Planning Guidelines:
        - Allocate 30% of sprint capacity for bug fixes
        - Reserve 20% for unexpected complex issues
        - Balance team workload based on complexity distribution
        """)

        # Live incoming issues feed with remedies
        st.subheader("Live Incoming Issues")
        if api_status:
            feed_records = get_recent_predictions(limit=100)
            if feed_records:
                try:
                    feed_df = pd.DataFrame(feed_records)
                    # Ensure required columns exist
                    for col in ["timestamp", "repo", "issue_number", "title", "complexity", "confidence", "labels", "comments"]:
                        if col not in feed_df.columns:
                            feed_df[col] = None
                    # Remedy suggestions
                    feed_df["remedy"] = feed_df.apply(
                        lambda r: suggest_remedy(
                            str(r.get("complexity") or ""),
                            float(r.get("confidence") or 0.0),
                            r.get("labels") if isinstance(r.get("labels"), list) else (
                                [l.strip() for l in str(r.get("labels") or "").split(",") if l.strip()]
                            ),
                            int(r.get("comments") or 0)
                        ), axis=1
                    )
                    # Sort by latest
                    if "timestamp" in feed_df.columns:
                        try:
                            feed_df["_ts"] = pd.to_datetime(feed_df["timestamp"], errors="coerce")
                            feed_df = feed_df.sort_values("_ts", ascending=False)
                        except Exception:
                            pass
                    # Display concise columns
                    display_cols = ["timestamp", "repo", "issue_number", "title", "complexity", "confidence", "remedy"]
                    st.dataframe(feed_df[display_cols], use_container_width=True)
                except Exception as e:
                    st.warning(f"Unable to render live feed: {e}")
            else:
                st.info("No recent predictions yet. New issues will appear here in real time.")
        else:
            st.info("API is not running. Start the API to enable the live feed.")

if __name__ == "__main__":
    main()