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
        # Ensure directories exist
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        os.makedirs(self.config.evaluation_dir, exist_ok=True)
        logger.info(f"Ensured directory structure exists for cloud storage push")

    def validate_bucket_exists(self):
        """Validate that the S3 bucket exists before attempting upload."""
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            logger.info(f"Bucket '{self.config.bucket_name}' exists and is accessible")
            return True
        except ClientError as e:
            logger.error(f"Bucket validation failed: {e}")
            logger.error(f"Bucket '{self.config.bucket_name}' does not exist or is not accessible")
            return False

    def get_files_to_upload(self):
        """Get all files that need to be uploaded to cloud storage."""
        files_to_upload = []

        if os.path.exists(self.config.data_version_dir):
            data_files = glob.glob(str(self.config.data_version_dir / "**"), recursive=True)
            files_to_upload.extend([f for f in data_files if os.path.isfile(f)])

        if os.path.exists(self.config.evaluation_dir):
            eval_files = glob.glob(str(self.config.evaluation_dir / "**"), recursive=True)
            files_to_upload.extend([f for f in eval_files if os.path.isfile(f)])

        return files_to_upload

    def upload_file_to_s3(self, file_path, object_key):
        try:
            self.s3_client.upload_file(file_path, self.config.bucket_name, object_key)
            return file_path, None
        except (BotoCoreError, ClientError) as e:
            return file_path, e

    def push_to_cloud_storage(self):
        """Push all artifacts to S3 cloud storage."""
        try:
            logger.info("Starting cloud storage push process...")

            if not self.validate_bucket_exists():
                raise Exception(
                    f"S3 BUCKET SETUP REQUIRED:\n"
                    f"1. Create S3 bucket '{self.config.bucket_name}' in your AWS account.\n"
                    f"2. Ensure your AWS credentials are set (e.g., via environment, ~/.aws/credentials, or IAM role).\n"
                    f"3. Ensure the user has 's3:PutObject' permission on the bucket."
                )

            files_to_upload = self.get_files_to_upload()
            if not files_to_upload:
                logger.warning("No files found to upload to cloud storage.")
                return

            logger.info(f"Found {len(files_to_upload)} files to upload to cloud storage.")

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                for file_path in files_to_upload:
                    rel_path = os.path.relpath(file_path, self.config.root_dir)
                    object_key = f"churn_data_store/{rel_path.replace(os.sep, '/')}"
                    futures[executor.submit(self.upload_file_to_s3, file_path, object_key)] = file_path

                for future in as_completed(futures):
                    file_path = futures[future]
                    uploaded_file, error = future.result()
                    if error:
                        logger.error(f"Failed to upload {uploaded_file}: {error}")
                    else:
                        logger.info(f"Uploaded {uploaded_file} to bucket {self.config.bucket_name}.")

            logger.info(f"Successfully uploaded {len(files_to_upload)} files to S3 bucket: {self.config.bucket_name}")

        except Exception as e:
            logger.error(f"Failed to push to cloud storage: {e}")
            raise e