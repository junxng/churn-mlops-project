import os
import uuid
import boto3
import logging
from datetime import datetime,timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_file_to_s3(local_file, s3_bucket, s3_key, region="ap-southeast-2"):
    """Upload a file to S3"""
    s3_client = boto3.client('s3', region_name=region)

    try:
        logger.info(f"Uploading {local_file} to s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(local_file, s3_bucket, s3_key)
        logger.info(f"Successfully uploaded {local_file} to s3://{s3_bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def upload_dataset_to_s3(train_file=None, test_file=None, file=None, bucket_name=None, prefix="sentiment-analysis/data"):
    """Upload train, test, or single file to S3 with unique filenames"""
    if bucket_name is None:
        logger.error("Bucket name is required.")
        return False

    success = True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if train_file:
        if os.path.exists(train_file):
            unique_train_key = f"{prefix}/train_data_{timestamp}_{uuid.uuid4().hex[:6]}.csv"
            success &= upload_file_to_s3(train_file, bucket_name, unique_train_key)
        else:
            logger.error(f"Train file {train_file} does not exist")
            success = False

    if test_file:
        if os.path.exists(test_file):
            unique_test_key = f"{prefix}/test_data_{timestamp}_{uuid.uuid4().hex[:6]}.csv"
            success &= upload_file_to_s3(test_file, bucket_name, unique_test_key)
        else:
            logger.error(f"Test file {test_file} does not exist")
            success = False

    if file:
        if os.path.exists(file):
            filename = os.path.basename(file)
            unique_file_key = f"{prefix}/{filename.split('.')[0]}_{timestamp}_{uuid.uuid4().hex[:6]}.csv"
            success &= upload_file_to_s3(file, bucket_name, unique_file_key)
        else:
            logger.error(f"File {file} does not exist")
            success = False

    return success
