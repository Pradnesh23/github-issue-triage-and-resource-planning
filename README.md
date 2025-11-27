# GitHub Issue Triage System And Resource Planning

An automated system for GitHub issue triage and resource planning that classifies incoming GitHub issues by complexity to help development teams prioritize and assign work appropriately.

## Features

- **Real-time GitHub Integration**: Webhook configuration to receive real-time notifications for new issues
- **Automatic Analysis**: Issues are automatically sent to the prediction API for complexity classification
- **Dashboard Updates**: Results are instantly displayed in the Streamlit dashboard with updated resource allocation recommendations
- **Continuous Learning**: New issues can be used to retrain models periodically for improved accuracy
- **Team Notifications**: Automatic team assignments based on issue complexity predictions

## Technology Stack

- **ML & Data Science**: numpy, pandas, scikit-learn, xgboost
- **Deep Learning (optional)**: torch, transformers (for BERT)
- **NLP Libraries**: nltk, spacy
- **Feature Engineering**: category-encoders, imbalanced-learn
- **Visualization**: matplotlib, seaborn, plotly, shap
- **API & Web**: FastAPI, uvicorn, streamlit, pydantic
- **Data Collection**: PyGithub, requests
- **MLOps**: mlflow, optuna
- **Utilities**: python-dotenv, pyyaml, tqdm, joblib

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Pradnesh23/github-issue-triage-and-resource-planning.git
   cd github-issue-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the application by editing `config.yaml`:
   - Add your GitHub personal access token
   - Configure repositories to monitor
   - Set up notification settings (optional)

## Usage

### 1. Start the API Server
```bash
uvicorn api.main:app --reload --port 8000
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 3. Continuous Learning and Retraining

To retrain using both the original dataset and ongoing human feedback, use the consolidated dataset workflow:

- Merge and deduplicate the raw training data (`data/raw/github_issues.csv`) with human feedback (`data/raw/human_feedback.csv`).
- Prefer human-labeled entries when duplicates exist; otherwise choose the most recent/informative row.
- Save the normalized result to `data/processed/consolidated_issues.csv`.

Build via API:

`POST http://127.0.0.1:8000/dataset/consolidate`

`GET  http://127.0.0.1:8000/dataset/stats`

Or programmatically:

`from src.data.dataset_builder import DatasetBuilder`

`builder = DatasetBuilder(raw_dir="data/raw", processed_dir="data/processed")`

`result = builder.build()`

`print(result)`

Use in the notebook (`notebooks/github-issue-predictor.ipynb`):

`import pandas as pd`

`df = pd.read_csv("data/processed/consolidated_issues.csv")`

`df = df[df["target_label"].notna() & (df["target_label"] != "UNKNOWN")]`

`label_map = {"SIMPLE": 0, "MODERATE": 1, "COMPLEX": 2}`

`df = df[df["target_label"].isin(label_map)].copy()`

`df["label_id"] = df["target_label"].map(label_map)`

`# Use df["title"] and df["body"] for DistilBERT or feature engineering for XGBoost.`

### 3. GitHub Webhook Integration

To set up real-time GitHub integration:

1. Deploy the application to a publicly accessible server
2. Go to your GitHub repository settings
3. Navigate to "Webhooks" section
4. Click "Add webhook"
5. Set the Payload URL to: `YOUR_SERVER_URL/webhook/github`
6. Set Content type to: `application/json`
7. Select "Let me select individual events" and choose "Issues" and "Issue comments"
8. Ensure "Active" is checked
9. Click "Add webhook"

### 4. Testing Webhook Integration

To test the webhook integration locally:

1. Start the API server:
   ```bash
   python start_test_server.py
   ```

2. In another terminal, run the webhook test:
   ```bash
   python test_webhook.py
   ```

### 5. Data Pipeline

The system includes a complete data pipeline:

1. **Data Collection**: `python -m src.data.scraper`
2. **Preprocessing**: `python -m src.data.preprocessor`
3. **Feature Engineering**: `python -m src.features.feature_engineering`
4. **Model Training**: `python -m src.models.train`
5. **Orchestration**: `python run_pipeline.py`

## Project Structure

```
.
├── api/                 # FastAPI endpoints
│   ├── main.py          # Main API application
│   ├── schemas.py       # Data schemas
│   └── webhook.py       # GitHub webhook handler
├── app/                 # Streamlit web interface
│   └── streamlit_app.py
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Core modules
│   ├── data/            # Data scraping and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   └── notifications/   # Notification system
├── config.yaml          # Configuration file
├── requirements.txt     # Dependencies
└── run_pipeline.py      # Orchestration script
```

## Configuration

The `config.yaml` file contains all configuration options:

- GitHub API token and repositories
- Data processing settings
- Feature engineering parameters
- Model training configurations
- Notification settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
