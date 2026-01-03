# 🚀 GitHub Issue Triage & Resource Allocation

![Tests](https://github.com/Pradnesh23/github-issue-triage-and-resource-planning/actions/workflows/test.yml/badge.svg)
![Build](https://github.com/Pradnesh23/github-issue-triage-and-resource-planning/actions/workflows/deploy.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![DVC](https://img.shields.io/badge/DVC-Gdrive-purple.svg)

An intelligent system to **classify GitHub issues**, **prioritize work**, and **automate triage** using Deep Learning (DistilBERT + XGBoost) and MLOps best practices.

## ✨ Key Features

- **🧠 AI-Powered**: Fine-tuned DistilBERT + XGBoost ensemble (53% Acc on complex tasks).
- **⚡ MLOps Integration**: DVC for data versioning, MLflow for tracking, GitHub Actions for CI/CD.
- **🔔 Smart Notifications**: Color-coded Slack alerts with team routing (@channel for complex issues).
- **🤖 GenAI Advisor**: Google Gemini integration for automated resolution strategies.
- **📊 Advanced Analytics**: Streamlit dashboard with trend analysis and resource planning.
- **🔄 Continuous Learning**: Feedback loop to retrain models on new data.

---

## 🛠️ Architecture

| Component | Tech Stack |
| :--- | :--- |
| **Model** | HuggingFace DistilBERT, XGBoost, Scikit-learn |
| **Backend** | FastAPI, Uvicorn, PyGithub |
| **Frontend** | Streamlit, Plotly |
| **MLOps** | DVC (Google Drive), MLflow, GitHub Actions |
| **Alerts** | Slack Webhooks |

```mermaid
graph TD
    %% Nodes
    A[GitHub User] -->|Creates Issue| B(GitHub Repository)
    B -->|Webhook| C[FastAPI Backend]
    C -->|Classify| D{AI Model}
    C -->|Ask| I[Gemini GenAI]
    D -->|Complexity Score| E[Resource Allocator]
    I -->|Resolution Plan| E
    E -->|Notification| F[Slack Channel]
    E -->|Data| G[Streamlit Analytics]
    
    subgraph "AI System"
    D -.->|Loads| H[DistilBERT + XGBoost]
    I -.->|API| J[Google Gemini]
    end
    
    %% Styling
    classDef primary fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;
    classDef secondary fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c;
    classDef support fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20;
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;

    class C,E,G primary;
    class D,H,I,J secondary;
    class F support;
    class A,B external;
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Pradnesh23/github-issue-triage-and-resource-planning.git
pip install -r requirements.txt
dvc pull  # Download model & data from Google Drive
```

### 2. Configuration

Create a `.env` file:

```env
GITHUB_TOKEN=ghp_...
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
GEMINI_API_KEY=AIza...
```

### 3. Run

**API:**

```bash
uvicorn api.main:app --reload
```

**Dashboard:**

```bash
streamlit run app/streamlit_app.py
```

---

## 🤖 MLOps Workflow

1. **Train**: Run `notebooks/github-issue-predictor.ipynb` on Kaggle (GPU).
2. **Track**: Log results to MLflow (`python scripts/log_kaggle_run.py`).
3. **Version**: Push model to Google Drive (`dvc add ... && dvc push`).
4. **Deploy**: Push to GitHub → CI/CD runs tests & builds.

```mermaid
flowchart LR
    %% Nodes
    A[Kaggle GPU] -->|Train| B(Model Artifacts)
    B -->|Log Metrics| C{MLflow}
    B -->|Version Control| D{DVC + GDrive}
    D -->|Push| E[GitHub Repo]
    E -->|Trigger| F[GitHub Actions]
    F -->|Deploy| G[Production]
    
    %% Styling
    classDef train fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#333;
    classDef track fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#333;
    classDef ci fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#333;
    
    class A,B,D train;
    class C track;
    class E,F,G ci;
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Predict complexity for one issue |
| `POST` | `/predict_batch` | Batch predictions |
| `POST` | `/webhook/github` | Handle GitHub events |
| `GET`  | `/dataset/stats` | View training data stats |

> Full docs at `http://localhost:8000/docs`

---

## 📂 Project Structure

```text
├── .github/workflows/   # CI/CD (Test, DVC, Retrain, Deploy)
├── api/                 # FastAPI Backend
├── app/                 # Streamlit Dashboard
├── best_model_bert_3class/ # Production Model (DVC)
├── data/                # Dataset (DVC coverage)
├── notebooks/           # Training (Kaggle)
├── src/                 # Source Code
│   ├── models/          # Predictor & MLflow utils
│   └── notifications/   # Slack Notifier
└── dvc.yaml             # Pipeline Config
```

---

*Licensed under MIT.*
