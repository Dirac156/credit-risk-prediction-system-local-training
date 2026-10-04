# Credit Risk Prediction System (Local Training & Vertex AI Deployment)

An end-to-end machine learning system designed to evaluate loan applicant default risk, train and evaluate a Scikit-Learn pipeline locally, deploy artifacts to Google Cloud Platform (GCP) Vertex AI, serve real-time online predictions, and manage resource teardown.

---

## Table of Contents
- [Architecture & Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Tech Stack & Prerequisites](#tech-stack--prerequisites)
- [Environment Setup](#environment-setup)
- [Data Ingestion & Preprocessing](#data-ingestion--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [GCP Setup & Vertex AI Deployment](#gcp-setup--vertex-ai-deployment)
- [Live Online Inference](#live-online-inference)
- [Resource Cleanup](#resource-cleanup)
- [Roadmap](#roadmap)

---

## Architecture & Workflow

1. **Data Ingestion & Quality Control:**
   - Automated dataset retrieval from Kaggle using the Kaggle API.
   - Cleans extreme data anomalies (e.g., age $> 100$, employment length $> 60$ years) and handles missing records.
2. **Scikit-Learn Pipeline Construction:**
   - Encoders (`OneHotEncoder`) for categorical variables.
   - Scalers (`StandardScaler`) for numerical features.
   - Combined inside a single `Pipeline` object to prevent data leakage during cross-validation and deployment.
3. **Model Evaluation:**
   - Out-of-sample testing measured using Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
4. **Cloud Serialization & Vertex AI Hosting:**
   - Export pipeline as `model.joblib`.
   - Upload artifact to Google Cloud Storage (GCS).
   - Register in Vertex AI Model Registry with pre-built Scikit-Learn container environments.
   - Deploy to an active Vertex AI Endpoint (`n1-standard-4`) for real-time predictions.
5. **Inference & Decommissioning:**
   - Query endpoint using the Vertex AI Python SDK and REST `curl` requests.
   - Automated cleanup script to undeploy models and delete cloud resources to avoid unexpected charges.

---

## Project Structure

```text
credit-risk-prediction-system-local-training/
├── data/
│   └── raw/                   # Raw credit risk dataset
├── models/
│   └── model.joblib           # Serialized Scikit-Learn pipeline
├── .env                       # GCP project and region configurations
├── .gitignore                 # Git ignore rules
├── 01_model_training.ipynb    # Main development and deployment notebook
├── main.py                    # Standalone execution script
├── pyproject.toml             # Project metadata & uv/pip dependency definitions
├── uv.lock                    # Locked dependency graph
└── README.md                  # Project documentation


## Tech Stack & Prerequisites

### Tech Stack
- **Language:** Python 3.10+
- **Package Manager:** `uv` / `pip`
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`
- **Cloud Infrastructure:** Google Cloud Storage (GCS), Google Cloud Vertex AI

### Prerequisites
- Active Google Cloud Platform (GCP) account with billing enabled.
- GCP Project with **Vertex AI API** and **Cloud Storage API** enabled.
- Google Cloud SDK (`gcloud` CLI) installed locally.

---

## Environment Setup

### 1. Clone & Set Up Directory
```bash
git clone [https://github.com/](https://github.com/)<your-username>/credit-risk-prediction-system-local-training.git
cd credit-risk-prediction-system-local-training

### 2. Virtual Environment & Dependencies
Using `uv`:
```bash
uv venv
source .venv/bin/activate
uv sync

Or using standard `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 3. GCP Configuration & Authentication
Create a `.env` file in the root directory:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1

