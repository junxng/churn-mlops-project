import os
import glob
from pathlib import Path
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import CloudStoragePushConfig
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

class CloudStoragePush:
    def __init__(self, config: CloudStoragePushConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.aws_key_id,
            aws_secret_access_key=config.aws_secret_key,
            region_name=config.region_name
        )
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        os.makedirs(self.config.evaluation_dir, exist_ok=True)

    def validate_bucket_exists(self) -> bool:
        """Validate that the S3 bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            logger.info(f"Bucket '{self.config.bucket_name}' exists and is accessible.")
            return True
        except ClientError as e:
            logger.error(f"Bucket validation failed: {e}. Ensure the bucket exists and credentials are correct.")
            return False

    def get_files_to_upload(self) -> list:
        """Get all files from specified directories to be uploaded."""
        files_to_upload = []
        for dir_path in [self.config.data_version_dir, self.config.evaluation_dir]:
            if os.path.exists(dir_path):
                files = glob.glob(str(Path(dir_path) / "**"), recursive=True)
                files_to_upload.extend([f for f in files if os.path.isfile(f)])
        return files_to_upload

    def upload_file_to_s3(self, file_path: str, object_key: str) -> tuple:
        """Upload a single file to S3."""
        try:
            self.s3_client.upload_file(file_path, self.config.bucket_name, object_key)
            return file_path, None
        except (BotoCoreError, ClientError) as e:
            return file_path, e

    def push_to_cloud_storage(self):
        """Push all artifacts to S3 cloud storage."""
        logger.info("Starting cloud storage push process...")
        if not self.validate_bucket_exists():
            raise Exception("S3 bucket validation failed. Aborting cloud storage push.")

        files_to_upload = self.get_files_to_upload()
        if not files_to_upload:
            logger.warning("No files found to upload.")
            return

        logger.info(f"Found {len(files_to_upload)} files to upload.")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    self.upload_file_to_s3, 
                    file_path, 
                    f"{self.config.s3_object_prefix}/{os.path.relpath(file_path, self.config.root_dir).replace(os.sep, '/')}"
                ): file_path for file_path in files_to_upload
            }

            for future in as_completed(futures):
                file_path, error = future.result()
                if error:
                    logger.error(f"Failed to upload {file_path}: {error}")
                else:
                    logger.info(f"Successfully uploaded {file_path} to S3.")
        
        logger.info("Cloud storage push process completed.")