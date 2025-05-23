#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Initializing DVC for the project...${NC}"

# Check if DVC is installed
if ! command -v dvc &> /dev/null
then
    echo -e "${RED}DVC is not installed. Installing it now...${NC}"
    pip install dvc dvc[s3]
fi

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo -e "${YELLOW}Initializing DVC repository...${NC}"
    dvc init
    git add .dvc/
    git commit -m "Initialize DVC"
else
    echo -e "${GREEN}DVC already initialized.${NC}"
fi

# Configure AWS credentials
echo -e "${YELLOW}Checking AWS credentials...${NC}"
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo -e "${RED}AWS credentials not found in environment variables.${NC}"
    echo -e "${YELLOW}Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.${NC}"
    echo -e "You can do this by running:"
    echo -e "export AWS_ACCESS_KEY_ID=your_access_key"
    echo -e "export AWS_SECRET_ACCESS_KEY=your_secret_key"
    exit 1
fi

# Add S3 remote
echo -e "${YELLOW}Setting up S3 remote storage...${NC}"
dvc remote add -d s3remote s3://mlop/sentiment-analysis
dvc remote modify s3remote region ap-southeast-2
echo -e "${GREEN}S3 remote added successfully.${NC}"

# Add files to DVC
echo -e "${YELLOW}Adding data files to DVC...${NC}"
mkdir -p artifacts/data_ingestion

# Add data files (if they exist)
if [ -f "artifacts/data_ingestion/train_data.csv" ]; then
    dvc add artifacts/data_ingestion/train_data.csv
    echo -e "${GREEN}Added train_data.csv to DVC tracking.${NC}"
fi

if [ -f "artifacts/data_ingestion/test_data.csv" ]; then
    dvc add artifacts/data_ingestion/test_data.csv
    echo -e "${GREEN}Added test_data.csv to DVC tracking.${NC}"
fi

# Push to remote (if files were added)
if [ -f "artifacts/data_ingestion/train_data.csv.dvc" ] || [ -f "artifacts/data_ingestion/test_data.csv.dvc" ]; then
    echo -e "${YELLOW}Pushing data files to S3 remote...${NC}"
    dvc push
    echo -e "${GREEN}Files pushed to S3 successfully.${NC}"
    
    # Commit DVC files
    git add .gitignore artifacts/data_ingestion/*.dvc
    git commit -m "Add data files to DVC tracking"
fi

echo -e "${GREEN}DVC setup completed successfully!${NC}" 