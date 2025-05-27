import os
import uuid
import boto3
import logging
from datetime import datetime,timezone
from google.cloud.storage import Client, transfer_manager
from pathlib import Path
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_many_blobs_with_transfer_manager(
    bucket_name, filenames, source_directory="", workers=8
):
    """Upload multiple files to Google Cloud Storage with proper error handling."""
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Admin\Downloads\llmops-460406-f379299f4261.json"
        
        if isinstance(filenames, (Path, str)):
            filenames = [str(filenames)]
        else:
            filenames = [str(f) for f in filenames]
        
        for filename in filenames:
            file_path = os.path.join(source_directory, filename) if source_directory else filename
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist for upload: {file_path}")
                continue
        
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        results = transfer_manager.upload_many_from_filenames(
            bucket, filenames, source_directory=source_directory, max_workers=workers, blob_name_prefix="churn_data_store/"
        )

        for name, result in zip(filenames, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload {name}: {result}")
            else:
                logger.info(f"Uploaded {name} to bucket {bucket.name}.")
                
    except Exception as e:
        logger.error(f"Cloud storage upload failed: {e}")
        raise e