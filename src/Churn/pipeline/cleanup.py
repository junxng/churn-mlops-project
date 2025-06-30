from src.Churn.utils.logging import logger
import os
import glob
from src.Churn.config.configuration import ConfigurationManager

def cleanup_temp_files():
    """Clean up temporary versioned files after pipeline completion."""
    logger.info("=" * 50)
    logger.info("CLEANUP: Removing temporary versioned files")
    logger.info("=" * 50)
    config_manager = ConfigurationManager()
    data_version_dir = config_manager.config.data_ingestion.data_version_dir
    evaluation_dir = config_manager.config.evaluation.evaluation_dir
    data_ingestion_dir = config_manager.config.data_ingestion.root_dir
    try:
        data_version_files = glob.glob(os.path.join(data_version_dir, "*_version_????????T??????.csv"))
        logger.info(f"Found {len(data_version_files)} timestamp-versioned data files to clean")
        
        for file_path in data_version_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        eval_json_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.json"))
        eval_png_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.png"))
        eval_files = eval_json_files + eval_png_files
        logger.info(f"Found {len(eval_files)} timestamp-versioned evaluation files to clean")
        
        for file_path in eval_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
                
        input_data_files = glob.glob(os.path.join(data_ingestion_dir, "*"))
        for file_path in input_data_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
                
        remaining_data_files = len(glob.glob(os.path.join(data_version_dir, "*.csv")))
        remaining_eval_files = len(glob.glob(os.path.join(evaluation_dir, "*")))
        logger.info(f"Kept {remaining_data_files} essential data files for DVC tracking")
        logger.info(f"Kept {remaining_eval_files} essential evaluation files for DVC tracking")
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")