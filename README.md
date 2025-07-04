# Churn Prediction MLOps Project

This project provides an end-to-end MLOps solution for customer churn prediction. It includes a machine learning pipeline for data processing, model training, and evaluation, as well as a FastAPI for serving predictions. The project is designed for reproducibility and scalability, incorporating tools like DVC and Kubeflow.

## Features

- **Data Processing**: Handles raw customer data, performs feature engineering, and prepares it for model training.
- **Model Training**: Trains a Random Forest Classifier and includes hyperparameter tuning to optimize performance.
- **Model Evaluation**: Evaluates the model using various metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
- **API Server**: A FastAPI application to serve churn predictions and trigger model retraining.
- **Reproducibility**: Uses DVC to version control data, models, and pipelines.
- **Orchestration**: Integrates with Kubeflow for pipeline orchestration.
- **Cloud Integration**: Pushes artifacts to cloud storage (AWS S3).

## Project Structure

```
├── artifacts/            # Stores data, models, and evaluation results
├── config/               # Configuration files
├── controller/           # FastAPI route handlers
├── src/                  # Source code for the ML pipeline
│   └── Churn/
│       ├── components/   # Individual pipeline components
│       ├── config/       # Configuration management
│       ├── entity/       # Data entities
│       ├── pipeline/     # ML pipelines
│       └── utils/        # Utility functions
├── main.py               # FastAPI application entry point
├── run_pipeline.py       # Script to run the DVC pipeline
├── dvc.yaml              # DVC pipeline definition
├── pyproject.toml        # Project dependencies
└── Dockerfile            # Dockerfile for containerization
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- DVC
- An AWS account with an S3 bucket for cloud storage

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/churn-mlops-project.git
    cd churn-mlops-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure cloud storage:**
    - Create an AWS S3 bucket.
    - Set up your AWS credentials in your environment.
    - Update `config/config.yaml` with your bucket name.

### Running the Pipeline

To run the entire machine learning pipeline, use the following command:

```bash
dvc repro
```

This will execute the stages defined in `dvc.yaml`, including data processing, model training, and evaluation.

### Running the API Server

To start the FastAPI server, run:

```bash
uv run main.py
```

The API will be available at `http://localhost:8888/docs`.

### API Endpoints

-   `POST /predict`: Get a churn prediction for a single customer.
-   `POST /retrain`: Trigger the model retraining pipeline.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.
