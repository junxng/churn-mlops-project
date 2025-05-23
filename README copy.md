# Sentiment Analysis MLOps Project

This project implements a sentiment analysis system using MLOps best practices, including:

- Modular code architecture
- Configuration management
- Model versioning with MLflow
- Data versioning with DVC
- CI/CD pipeline with GitHub Actions
- Docker containerization
- AWS integration (S3, ECR, EC2)

## Project Structure

```
├── .github/workflows/  # GitHub Actions workflows
├── artifacts/          # Model artifacts and data (tracked by DVC)
├── config/             # Configuration files
├── logs/               # Application logs
├── scripts/            # Utility scripts for deployment and data management  
├── src/                # Source code
│   ├── Sentiment_Analysis/
│   │   ├── components/     # Model components
│   │   ├── config/         # Configuration handling
│   │   ├── entity/         # Entity definitions
│   │   ├── pipeline/       # ML pipelines
│   │   └── utils/          # Utility functions
│   └── app.py          # Flask web service
├── .dvc/               # DVC configuration
├── Dockerfile          # Docker configuration
├── dvc.yaml            # DVC pipeline definition
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup and Installation

### Local Development

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up DVC for data versioning
   ```bash
   bash scripts/init_dvc.sh
   ```

5. Run the pipeline
   ```bash
   dvc repro
   ```

### AWS Configuration

1. Set up AWS credentials
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=ap-southeast-2
   ```

2. Upload data to S3
   ```bash
   python scripts/upload_data_to_s3.py
   ```

3. Configure DagsHub for MLflow tracking
   ```bash
   export MLFLOW_TRACKING_URI=https://dagshub.com/Teungtran/Mlops_Prac.mlflow
   export DAGSHUB_USERNAME=Teungtran
   export DAGSHUB_TOKEN=your_token
   ```

## Running the Application

### Local Execution

Run the Flask application:
```bash
python -m src.app
```

The API will be available at http://localhost:8080

### Docker Deployment

Build and run the Docker container:
```bash
docker build -t sentiment-analysis:latest .
docker run -p 8080:8080 sentiment-analysis:latest
```

### AWS EC2 Deployment

Set up an EC2 instance:
```bash
bash scripts/setup_ec2.sh
```

## API Endpoints

- **Health Check**: `GET /health`
- **Prediction**: `POST /predict`
  ```json
  {
    "text": "I really enjoyed this movie, it was fantastic!"
  }
  ```

## CI/CD Pipeline

The GitHub Actions workflow automates:
1. Testing
2. Building Docker images
3. Pushing to Amazon ECR
4. Deploying to EC2

## DVC Data Pipeline

The ML pipeline is defined in `dvc.yaml` with the following stages:
1. Data ingestion
2. Model preparation
3. Training
4. Evaluation

Run the pipeline:
```bash
dvc repro
```

View metrics:
```bash
dvc metrics show
```

## MLflow Experiment Tracking

All experiments are tracked in MLflow, which logs:
- Model parameters
- Metrics (accuracy, precision, recall, F1)
- Artifacts (model, tokenizer, plots)
- Model registry

View the experiments at: https://dagshub.com/Teungtran/Mlops_Prac.mlflow