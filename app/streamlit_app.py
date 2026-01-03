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
        st.header("Analytics & Insights")
        st.markdown("Deep dive into GitHub issue trends and AI performance.")

        # Controls for refreshing
        col0a, col0b = st.columns([3, 1])
        with col0a:
             metric_period = st.selectbox("Time Period", ["All Time", "Last 7 Days", "Last 30 Days"])
        with col0b:
             if st.button("🔄 Refresh Data"):
                 st.experimental_rerun()

        # Data Fetching
        dashboard_df = None
        if api_status:
            # Increase limit for analytics
            records = get_recent_predictions(limit=500)
            if records:
                 try:
                    dashboard_df = pd.DataFrame(records)
                 except Exception:
                    pass
        
        if dashboard_df is None or dashboard_df.empty:
            st.info("No sufficient data for analytics. Using sample data.")
             # Generate sample data with timestamps
            dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
            sample_data = pd.DataFrame({
                "timestamp": dates,
                "repo": ["owner/repo"] * 50,
                "complexity": np.random.choice(["simple", "moderate", "complex"], 50, p=[0.4, 0.4, 0.2]),
                "confidence": np.random.uniform(0.5, 0.99, 50),
                "labels": [["bug", "ui"], ["enhancement"], ["bug", "critical"]] * 16 + [["docs"], ["feature"]],
                "title": [f"Issue {i}" for i in range(50)],
                "comments": np.random.randint(0, 15, 50)
            })
            dashboard_df = sample_data
        else:
            # Convert timestamp
            if "timestamp" in dashboard_df.columns:
                dashboard_df["timestamp"] = pd.to_datetime(dashboard_df["timestamp"], errors='coerce')
        
        # --- TOP METRICS ---
        if dashboard_df is not None and not dashboard_df.empty:
            total_issues = len(dashboard_df)
            complex_count = len(dashboard_df[dashboard_df["complexity"] == "complex"])
            avg_conf = dashboard_df["confidence"].mean() if "confidence" in dashboard_df.columns else 0
            high_conf_count = len(dashboard_df[dashboard_df["confidence"] > 0.9]) if "confidence" in dashboard_df.columns else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total API Predictions", total_issues)
            m2.metric("Complex Issues", complex_count, delta_color="inverse")
            m3.metric("Avg AI Confidence", f"{avg_conf:.1%}")
            m4.metric("High Confidence (>90%)", high_conf_count)

            st.markdown("---")

            # --- CHART ROW 1 ---
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Complexity Trend")
                if "timestamp" in dashboard_df.columns:
                    # Scatter plot of confidence over time, colored by complexity
                    fig_trend = px.scatter(
                        dashboard_df, 
                        x="timestamp", 
                        y="confidence", 
                        color="complexity", 
                        title="Incoming Issues Timeline",
                        hover_data=["title"],
                        color_discrete_map={"simple": "#4CAF50", "moderate": "#FF9800", "complex": "#F44336"}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning("No timestamp data available for trend analysis.")

            with c2:
                st.subheader("Complexity Distribution")
                if "complexity" in dashboard_df.columns:
                    fig_dist = px.pie(
                        dashboard_df, 
                        names="complexity", 
                        color="complexity", 
                        hole=0.4,
                        title="Overall Issue Distribution",
                        color_discrete_map={"simple": "#4CAF50", "moderate": "#FF9800", "complex": "#F44336"}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            # --- CHART ROW 2 ---
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("Top Labels")
                # Flatten labels
                all_labels = []
                if "labels" in dashboard_df.columns:
                    for item in dashboard_df["labels"]:
                        if isinstance(item, list):
                            for l in item:
                                if isinstance(l, dict):
                                    all_labels.append(l.get("name", "unknown"))
                                else:
                                    all_labels.append(str(l))
                        elif isinstance(item, str):
                            all_labels.extend([x.strip() for x in item.split(",") if x.strip()])
                
                if all_labels:
                    lbl_counts = pd.Series(all_labels).value_counts().head(10).reset_index()
                    lbl_counts.columns = ["Label", "Count"]
                    fig_lbl = px.bar(
                        lbl_counts, 
                        x="Count", 
                        y="Label", 
                        orientation='h', 
                        title="Most Frequent Labels",
                        color="Count",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_lbl, use_container_width=True)
                else:
                    st.info("No label data available.")

            with c4:
                st.subheader("High Priority Issues")
                # Filter for complex issues
                if "complexity" in dashboard_df.columns:
                    complex_df = dashboard_df[dashboard_df["complexity"] == "complex"]
                    if not complex_df.empty:
                        # Show table
                        display_cols = ["timestamp", "title", "confidence"]
                        # Filter to only existing columns
                        cols_to_show = [c for c in display_cols if c in complex_df.columns]
                        st.dataframe(
                            complex_df[cols_to_show].sort_values("timestamp", ascending=False).head(10),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.success("No complex issues found! 🎉")

            # --- RESOURCE PLANNING ---
            st.markdown("---")
            st.subheader("Resource Planning Advice")
            
            # Simple heuristic advice
            rec_text = ""
            if complex_count > total_issues * 0.3:
                rec_text = "⚠️ **High volume of complex issues.** Consider allocating more Senior Engineers to triage."
            elif total_issues > 0 and (complex_count / total_issues) < 0.1:
                rec_text = "✅ **Healthy mix.** Junior/Mid-level devs should be able to handle the current load."
            else:
                 rec_text = "ℹ️ **Standard load.** Maintain current team distribution."
            
            st.markdown(rec_text)

if __name__ == "__main__":
    main()