from src.Sentiment_Analysis.config.configuration import ConfigurationManager
from src.Sentiment_Analysis.components.data_ingestion import DataIngestion
from src.Sentiment_Analysis.utils.logging import logger

STAGE_NAME = "Data Ingestion stage"

class DataPreparationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        pipeline = DataIngestion(data_ingestion_config)
        pipeline.data_ingestion_pipeline()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = DataPreparationPipeline()
        train_path, test_path = pipeline.main()
        logger.info(f"Data preparation completed successfully")
        logger.info(f"Train data: {train_path}")
        logger.info(f"Test data: {test_path}")
    except Exception as e:
        logger.exception(f"Error in Data Preparation Pipeline: {e}")
        raise e
