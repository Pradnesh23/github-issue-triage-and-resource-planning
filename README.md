# GitHub Issue Triage and Resource Allocation

An intelligent system for GitHub issue triage and resource planning that classifies incoming GitHub issues by complexity (Simple, Moderate, Complex) to help development teams prioritize work and allocate resources effectively.

## Features

- **ML-Powered Classification**: Fine-tuned DistilBERT model with XGBoost stacking ensemble for accurate issue complexity prediction
- **Real-time GitHub Webhook Integration**: Automatically receive and process new GitHub issues in real-time
- **Interactive Streamlit Dashboard**: Visualize predictions, statistics, and resource allocation recommendations
- **REST API Endpoints**: FastAPI-based endpoints for programmatic access to predictions and dataset management
- **Continuous Learning**: Track human feedback and retrain models with new labeled data
- **Heuristic Fallback**: Robust heuristic-based predictions when ML models are unavailable
- **Persistent Prediction History**: Store and track all predictions for audit and analysis

## Technology Stack

- **ML & Deep Learning**: PyTorch, transformers (DistilBERT), scikit-learn, XGBoost
- **NLP Processing**: NLTK, spaCy
- **Feature Engineering**: category-encoders, imbalanced-learn
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib, SHAP
- **GitHub Integration**: PyGithub
- **MLOps & Experimentation**: MLflow, DVC, Optuna
- **Notifications**: Slack Webhooks
- **CI/CD**: GitHub Actions
- **Utilities**: PyYAML, python-dotenv, Joblib, aiohttp, requests

## MLOps Features

### DVC (Data Version Control)

Track large data files and models with Google Drive storage:

```bash
# Pull data/models from remote
dvc pull

# After training, push updates
dvc add best_model_bert_3class
dvc push

# Show metrics
dvc metrics show
```

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# Log Kaggle training results
python scripts/log_kaggle_run.py
```

### Slack Notifications

Receive real-time notifications when issues are triaged:

- 🟢 Simple issues → Junior developers
- 🟡 Moderate issues → Mid-level developers  
- 🔴 Complex issues → Senior developers (@channel mention)

### GitHub Actions

| Workflow | Trigger | Purpose |
|----------|---------|--------|
| `test.yml` | PR/Push | Lint & test |
| `dvc-pipeline.yml` | Push | Show DVC status |
| `retrain.yml` | Weekly | Training reminder |
| `deploy.yml` | Push | Build verification |

## Setup

### Prerequisites

- Python 3.8+
- Git
- GitHub Personal Access Token (for API access)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pradnesh23/github-issue-triage-and-resource-planning.git
   cd github-issue-triage-and-resource-planning
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained model (optional but recommended):
   - The project includes a pre-trained DistilBERT model in `best_model_bert_3class/`
   - If not present, the system will use heuristic-based predictions

5. Configure the application:

   ```bash
   cp config.yaml config.yaml.local
   # Edit config.yaml.local with your settings
   export GITHUB_TOKEN="your_github_token_here"
   export WEBHOOK_SECRET="your_webhook_secret_here"
   ```

## Usage

### 1. Start the FastAPI Server

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### 2. Run the Streamlit Dashboard

In a new terminal:

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`.

### 3. Make Predictions

**Via REST API:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fix authentication bug",
    "body": "Users cannot login with OAuth tokens",
    "labels": ["bug"],
    "comments": 5
  }'
```

**Batch Predictions:**

```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "issues": [...]
  }'
```

### 4. Dataset Operations

```bash
# Consolidate raw and human feedback data
curl -X POST "http://localhost:8000/dataset/consolidate"

# Get dataset statistics
curl -X GET "http://localhost:8000/dataset/stats"
```

### 5. GitHub Webhook Integration

To set up real-time GitHub integration:

1. Deploy the application to a publicly accessible server
2. Go to your GitHub repository settings → **Webhooks**
3. Click **Add webhook**
4. Configure:
   - **Payload URL**: `https://your-domain.com/webhook/github`
   - **Content type**: `application/json`
   - **Events**: Select "Issues" and "Issue comments"
   - **Active**: ✓ Checked
5. Click **Add webhook**

The API will automatically:

- Receive issue events
- Predict complexity
- Store results in `data/raw/predictions.jsonl`
- Display in the Streamlit dashboard

### 6. Continuous Learning & Model Retraining

**Data Consolidation:**

- Merge raw training data with human feedback
- Deduplicate entries (prefer human labels)
- Save to `data/processed/consolidated_issues.csv`

**Retraining Pipeline:**

```bash
python run_pipeline.py
```

**Jupyter Notebook:**

- See `notebooks/github-issue-predictor.ipynb` for EDA and training examples
- Prepare data for DistilBERT or XGBoost feature engineering

## Project Structure

```
.
├── .dvc/                         # DVC configuration
├── .github/workflows/            # GitHub Actions CI/CD
│   ├── test.yml                  # Lint & test workflow
│   ├── dvc-pipeline.yml          # DVC status workflow
│   ├── retrain.yml               # Weekly retrain reminder
│   └── deploy.yml                # Build verification
├── api/                          # FastAPI backend
│   ├── main.py                   # Main API application & endpoints
│   ├── schemas.py                # Pydantic request/response models
│   └── webhook.py                # GitHub webhook handler
├── app/                          # Streamlit web interface
│   └── streamlit_app.py          # Interactive dashboard
├── best_model_bert_3class/       # Pre-trained DistilBERT model (DVC tracked)
├── notebooks/                    # Jupyter notebooks
│   └── github-issue-predictor.ipynb  # Kaggle training notebook
├── scripts/                      # Utility scripts
│   └── log_kaggle_run.py         # Log Kaggle results to MLflow
├── src/                          # Core modules
│   ├── models/                   # Prediction & learning modules
│   │   ├── predict.py            # Issue complexity predictor
│   │   ├── mlflow_utils.py       # MLflow tracking utilities
│   │   └── continuous_learning.py
│   └── notifications/            # Slack notification system
│       └── notifier.py           # Rich Slack messages
├── data/                         # Data directory (DVC tracked)
│   ├── raw/                      # Raw GitHub issues & feedback
│   └── processed/                # Processed & consolidated data
├── config.yaml                   # Configuration file
├── dvc.yaml                      # DVC pipeline definition
├── metrics.json                  # Model metrics (DVC tracked)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Configuration

Edit `config.yaml` to customize:

```yaml
github:
  token: ${GITHUB_TOKEN}           # Set via environment variable
  webhook_secret: ${WEBHOOK_SECRET} # Set via environment variable
  repositories:
    - "owner/repo1"
    - "owner/repo2"

api:
  port: 8000
  host: "127.0.0.1"

model:
  # Uses pre-trained model in best_model_bert_3class/ if available
  # Falls back to heuristic predictions otherwise
```

**Important**: Never commit actual tokens. Use environment variables or `.env` files.

## Model Information

The system uses a **DistilBERT-based ensemble**:

- **Primary Model**: Fine-tuned DistilBERT (3-class: Simple, Moderate, Complex)
- **Secondary Model**: XGBoost stacking ensemble for feature-based predictions
- **Numeric Scaling**: Joblib-persisted scaler for numerical features
- **Fallback**: Heuristic-based predictions if models unavailable

**Performance**:

- Trained on 1000+ GitHub issues across popular repositories
- Considers: title, body, labels, comment count
- Handles edge cases with robust error handling

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict complexity for a single issue |
| POST | `/predict_batch` | Batch predict multiple issues |
| POST | `/dataset/consolidate` | Consolidate raw data with human feedback |
| GET | `/dataset/stats` | Get dataset statistics |
| POST | `/webhook/github` | GitHub webhook receiver |
| GET | `/docs` | Interactive API documentation |

## Testing

```bash
# Test the model
python test_model.py

# Test webhook locally
python start_test_server.py  # Terminal 1
python test_webhook.py       # Terminal 2
```

## Troubleshooting

**Issue**: Model not loading

- Ensure `best_model_bert_3class/` directory exists
- System will automatically fall back to heuristics if model unavailable

**Issue**: Cannot connect to API

- Check if `http://localhost:8000` is accessible
- Verify API server is running: `uvicorn api.main:app --reload`

**Issue**: Webhook not receiving events

- Ensure application is deployed publicly
- Check GitHub repository webhook settings
- Verify `WEBHOOK_SECRET` matches GitHub configuration

**Issue**: Low prediction accuracy

- Ensure model is fine-tuned on relevant issue data
- Check data quality in `data/processed/consolidated_issues.csv`
- Consider retraining with recent labeled data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
