# MLOps Sentiment Analysis

A complete MLOps pipeline for social media sentiment analysis. The system automatically classifies texts as **positive**, **neutral**, or **negative**, monitors model performance over time, and triggers retraining when drift is detected.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Phase 1 — Sentiment Analysis Model](#phase-1--sentiment-analysis-model)
- [Phase 2 — CI/CD Pipeline](#phase-2--cicd-pipeline)
- [Phase 3 — Deploy and Monitoring](#phase-3--deploy-and-monitoring)
- [GitHub Secrets](#github-secrets)
- [Design Choices](#design-choices)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

This project implements a full MLOps lifecycle for monitoring and improve the social media reputation of a company:

1. **Sentiment Analysis** using a pre-trained RoBERTa model fine-tuned on Twitter data
2. **CI/CD Pipeline** with GitHub Actions for automated training, validation, and deployment
3. **Continuous Monitoring** for sentiment distribution and data drift detection

---

## Architecture

```
Social Media Texts
        ↓
Sentiment Analysis (RoBERTa)
        ↓
Predictions Log (CSV)
        ↓
Monitoring (drift.py)
        ↓
[Drift detected?]
        ↓
Manual Retraining via GitHub Actions (workflow_dispatch)
        ↓
Model Validation (new vs production)
        ↓
Deploy to HuggingFace Hub
```

---

## Project Structure

```
mlops-sentiment-analysis/
│
├── notebooks/
│   └── delivery_notebook.ipynb     # Google Colab delivery notebook
│
├── src/
│   ├── model.py                    # Model loading and prediction
│   ├── train.py                    # Fine-tuning pipeline
│   ├── evaluate.py                 # Evaluation and validation
│   ├── deploy.py                   # HuggingFace Hub deployment
│   └── monitoring/
│       ├── __init__.py
│       ├── monitor.py              # Prediction logging
│       ├── drift.py                # Drift detection and reporting
│       └── simulate.py             # Data simulation for testing
│
├── tests/
│   ├── test_model.py               # Unit tests
│   └── test_integration.py         # Integration tests
│
├── predictions_log/
│   └── predictions_log.csv         # Prediction history (git-tracked)
│
├── .github/
│   └── workflows/
│       ├── CI_CD.yml               # Training, validation and deploy pipeline
│       └── monitoring.yml          # Scheduled monitoring workflow
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [HuggingFace](https://huggingface.co) account with a write-access token
- A GitHub repository with Actions enabled

### Installation

```bash
git clone https://github.com/giovannibonisoli/mlops-sentiment-analysis.git
cd mlops-sentiment-analysis
pip install -r requirements.txt
```

### Run locally

```bash
# Run tests
pytest tests/ -v

# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py

# Deploy to HuggingFace Hub
HF_TOKEN=hf_... HF_REPO=username/sentiment-model python src/deploy.py

# Run monitoring simulation
python src/monitoring/simulate.py           # normal distribution
python src/monitoring/simulate.py drift     # conceptual drift simulation
python src/monitoring/simulate.py data_drift # data drift simulation
```

---

## Base Model

### Model

The model used is [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model pre-trained on ~124M tweets and fine-tuned for sentiment analysis. Although the project brief mentions FastText, this model was explicitly indicated in the specifications and offers superior performance on social media texts due to its attention mechanism and Twitter-specific pre-training.

### Dataset

[`tweet_eval/sentiment`](https://huggingface.co/datasets/tweet_eval) — Twitter texts with three sentiment labels:

| Split      | Size   |
|------------|--------|
| Train      | ~45k   |
| Validation | ~1.5k  |
| Test       | ~12k   |

**Class distribution (test set):**

| Label    | Count | Percentage |
|----------|-------|------------|
| negative | 3,972 | 32.3%      |
| neutral  | 5,937 | 48.3%      |
| positive | 2,375 | 19.3%      |

### Baseline Results

Evaluated on the full test set using the pre-trained model without fine-tuning:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.72  |
| Macro F1  | 0.72  |

**Macro F1 is used as the primary metric** instead of accuracy because the dataset is imbalanced — neutral samples are almost double the positive ones. Accuracy on imbalanced datasets can be misleading, while Macro F1 penalizes models that perform poorly on minority classes.

---

## CI/CD Pipeline

The pipeline is implemented in a single workflow (`CI_CD.yml`) with two separate jobs:

### Job 1 — Test (triggered on every push/PR)

- **Lint** with flake8 — checks for syntax errors and undefined names
- **Unit tests** — validates individual functions in `model.py`
- **Integration tests** — validates the full pipeline from dataset to predictions

This job runs on every push because code correctness should always be verified, and it is fast enough to not waste resources.

### Job 2 — Train, Validate and Deploy (triggered only by `workflow_dispatch`)

This job is deliberately restricted to manual triggers only, because:
- Training takes ~30-60 minutes and consumes significant CPU resources
- Not every code change requires a new model
- Retraining should be a deliberate decision, not an automatic side effect of any commit

**Steps:**

1. **Train** — Fine-tunes the model on `tweet_eval/sentiment`. If a model already exists on HuggingFace Hub, it is used as the starting point instead of the base model, allowing progressive improvement across retraining cycles.

2. **Validate** — Compares the new model's Macro F1 against the production model. If the new model is not better, the pipeline exits with failure and deployment is cancelled.

3. **Deploy** — Pushes the validated model to HuggingFace Hub.

### Training Configuration

| Parameter          | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `dataset_name`     | tweet_eval | HuggingFace dataset to use                   |
| `dataset_config`   | sentiment  | Dataset configuration                        |
| `train_samples`    | 1000    | Training samples (CI/CD compromise)              |
| `validation_samples` | 1000  | Validation samples during training               |
| `num_epochs`       | 3       | Fine-tuning epochs (standard for transformers)   |
| `train_seed`       | 42      | Random seed for reproducibility                  |

**Note on `train_samples`:** 1000 samples is a compromise between CI/CD execution time and training quality. For a full training run, the entire training set (~45k samples) should be used, which can be done locally or on Colab with GPU.

**Note on `train_seed`:** In production, the seed would be fixed in code and data variability would come from new real-world data collected by the monitoring system. Here it is exposed as a parameter to simulate variability in the absence of a real data source.

### Deploy on HuggingFace Hub

The validated model is deployed to HuggingFace Hub, which acts as a centralized model registry. This enables versioning, easy integration, and scalability.

**Model:** [tuousername/sentiment-model](https://huggingface.co/giovannibonisoli/sentiment-model)

---

## Monitoring System

The monitoring system is composed of three modules:

#### `monitor.py`
Logs every prediction to a CSV file with:
- `timestamp` — UTC datetime
- `text` — original text
- `predicted_label` — predicted sentiment
- `confidence` — model confidence score (0.0–1.0)

#### `drift.py`
Analyzes the prediction log and detects anomalies:

| Check | Threshold | Action |
|-------|-----------|--------|
| Average confidence | < 0.65 | Manual review recommended |
| Sentiment distribution shift | > 15% per class | Retraining recommended |
| PSI (confidence scores) | > 0.2 | Retraining recommended |
| PSI (confidence scores) | 0.1–0.2 | Warning, monitor |

**PSI (Population Stability Index)** is computed on confidence score distributions rather than labels alone, because it provides an earlier signal of drift — the model tends to become uncertain before its predictions visibly change.

**Retraining policy:** Automatic retraining was considered but rejected in favor of a human-in-the-loop approach. The monitoring system detects drift and recommends retraining, but the actual trigger is a manual `workflow_dispatch` from GitHub Actions. This avoids uncontrolled retraining on potentially noisy data.

#### `simulate.py`
Simulates incoming social media texts for demonstration purposes:

| Mode | Description |
|------|-------------|
| `normal` | Balanced distribution from `tweet_eval` test set |
| `drift` | 70% negative — simulates a reputational crisis |
| `data_drift` | IMDb movie reviews — simulates domain shift |

**In production**, this module would be replaced by real social media API integrations (Twitter/X API, Reddit API, etc.).

### Monitoring Workflow

The `monitoring.yml` workflow runs automatically every 8 hours and:
1. Downloads the previous prediction log from the repository
2. Runs a simulation and appends new predictions to the log
3. Runs drift detection and prints a monitoring report
4. Commits the updated log back to the repository

The prediction log is stored directly in the repository (`predictions_log/predictions_log.csv`) to persist data across workflow runs. In production, a time-series database (e.g., PostgreSQL with TimescaleDB) would be used instead.

---

## GitHub Secrets

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace write-access token |
| `HF_REPO` | HuggingFace repository (e.g., `giovannibonisoli/sentiment-model`) |
| `GH_REPO` | GitHub repository (e.g., `giovannibonisoli/mlops-sentiment-analysis`) |

**Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions and does not need to be configured manually.

---

## Design Choices

**RoBERTa** —The base model is `cardiffnlp/twitter-roberta-base-sentiment-latest`, an encoder-based LLM fine-tuned for sentiment analysis on Twitter/X data. It is built on the RoBERTa architecture and classifies text into negative, neutral, and positive sentiment labels. The model was trained on roughly 124 million tweets collected between 2018 and 2021 using the TweetEval benchmark

**Macro F1 over Accuracy** — The dataset is imbalanced (neutral samples ≈ 2× positive samples). Macro F1 averages performance equally across classes, penalizing models that ignore minority classes.

**Single workflow over separate CI/CD** — Training, validation, and deployment are combined in a single workflow with two jobs. The test job runs on every push; the training job runs only on `workflow_dispatch`. This avoids the complexity of cross-workflow artifact passing while maintaining clear separation of concerns.

**Manual retraining trigger** — The monitoring system detects drift automatically but requires human confirmation before retraining. This prevents uncontrolled retraining on noisy or unrepresentative data, maintaining a human-in-the-loop approach for critical model updates.

**CSV over database for monitoring** — For simplicity and portability, prediction logs are stored in a CSV file committed to the repository. This is a didactic simplification; in production, a time-series database would provide better query performance and scalability.

---

## Limitations and Future Work

- **No real data source** — The monitoring system uses simulated data from `tweet_eval`. In production, it would integrate with real social media APIs.
- **No ground truth for retraining** — New data collected by monitoring has no real labels. In production, labels would be obtained through human feedback or automated labeling pipelines.
- **CPU-only training** — GitHub Actions standard runners have no GPU. Training 1000 samples for 3 epochs takes ~30-60 minutes. A GPU runner would reduce this to ~5-10 minutes.
- **CSV persistence** — The prediction log is stored in the repository. A production system would use a time-series database for better scalability and query performance.
