# Churn MLOps Project

This project implements an end-to-end MLOps pipeline for customer churn prediction.

## Setup Instructions

### 1. Environment Setup
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS S3 Configuration
The project uses AWS S3 for storing data versions and evaluation artifacts. To set up:

1. Create an AWS S3 bucket
2. Create a `.env` file in the project root with the following variables:
```
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=your_aws_region
```
3. Update the `bucket_name` in `config/config.yaml` to match your S3 bucket name

### 3. Running the Pipeline
```bash
python main.py
```

## Pipeline Stages
1. Data Preparation - Ingests and preprocesses data
2. Model Preparation - Creates and scales the base model
3. Train and Evaluate - Trains, evaluates, and fine-tunes the model if needed
4. Cloud Storage Push - Uploads data versions and evaluation artifacts to S3
5. Cleanup - Removes temporary versioned files

## Project Structure
- `artifacts/` - Stores generated artifacts
- `config/` - Configuration files
- `data/` - Input data directory
- `src/` - Source code
  - `Churn/` - Main package
    - `components/` - Core components (data ingestion, model training, etc.)
    - `config/` - Configuration management
    - `entity/` - Data entities and classes
    - `pipeline/` - Pipeline orchestration
    - `utils/` - Utility functions
- `TEST/` - Test scripts and notebooks