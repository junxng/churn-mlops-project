import os
import logging
from pathlib import Path
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_file_to_s3(s3_client, bucket_name, file_path, object_key):
    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
        return (file_path, None)
    except (BotoCoreError, ClientError) as e:
        return (file_path, e)

def upload_many_files_to_s3(
    bucket_name, filenames, source_directory="", workers=8, aws_access_key_id=None, aws_secret_access_key=None, region_name=None
):
    """Upload multiple files to Amazon S3 with proper error handling."""
    try:
        # Create S3 client with optional credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        if isinstance(filenames, (Path, str)):
            filenames = [str(filenames)]
        else:
            filenames = [str(f) for f in filenames]

        valid_files = []

        for filename in filenames:
            file_path = os.path.join(source_directory, filename) if source_directory else filename
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist for upload: {file_path}")
                continue
            object_key = f"churn_data_store/{filename}"
            valid_files.append((file_path, object_key))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(upload_file_to_s3, s3_client, bucket_name, file_path, object_key): file_path
                for file_path, object_key in valid_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    uploaded_file, error = future.result()
                    if error:
                        logger.error(f"Failed to upload {uploaded_file}: {error}")
                    else:
                        logger.info(f"Uploaded {uploaded_file} to bucket {bucket_name}.")
                except Exception as e:
                    logger.error(f"Unexpected error uploading {file_path}: {e}")

    except Exception as e:
        logger.error(f"S3 upload process failed: {e}")
        raise e
