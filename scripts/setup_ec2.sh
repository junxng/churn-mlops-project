#!/bin/bash

# Create the directory structure needed for the pipeline
mkdir -p artifacts/data_ingestion
mkdir -p artifacts/data_version
mkdir -p artifacts/model_version
mkdir -p artifacts/evaluation

echo "Directory structure created successfully"

# Copy any necessary initial data files
if [ -f "data/data_ingestion/input_raw.csv" ]; then
  cp data/data_ingestion/input_raw.csv artifacts/data_ingestion/
  echo "Initial data file copied to artifacts/data_ingestion/"
fi

# Set up environment variables if not using IAM roles
if [ -f ".env" ]; then
  echo "Found .env file, loading environment variables"
  set -a
  source .env
  set +a
fi

echo "EC2 setup completed successfully" 