#!/bin/bash

# Create the directory structure needed for the pipeline
mkdir -p artifacts/data_ingestion
mkdir -p artifacts/data_version
mkdir -p artifacts/model_version
mkdir -p artifacts/evaluation
mkdir -p plots

echo "Directory structure created successfully"

# Set up environment variables if not using IAM roles
if [ -f ".env" ]; then
  echo "Found .env file, loading environment variables"
  set -a
  source .env
  set +a
fi

echo "EC2 setup completed successfully" 