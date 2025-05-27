from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.data_ingestion import DataIngestion
from src.Churn.utils.logging import logger

STAGE_NAME = "Data Ingestion stage"

class DataPreparationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        pipeline = DataIngestion(data_ingestion_config)
        
        _, _, _, _, train_path, test_path, y_train_path, y_test_path = pipeline.data_ingestion_pipeline()
        
        return train_path, test_path, y_train_path, y_test_path


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = DataPreparationPipeline()
        train_path, test_path, y_train_path, y_test_path = pipeline.main()
        logger.info(f"Data preparation completed successfully")
        logger.info(f"Train data: {train_path}")
        logger.info(f"Test data: {test_path}")
        logger.info(f"Train targets: {y_train_path}")
        logger.info(f"Test targets: {y_test_path}")
    except Exception as e:
        logger.exception(f"Error in Data Preparation Pipeline: {e}")
        raise e
