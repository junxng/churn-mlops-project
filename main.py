import os
from src.Sentiment_Analysis.utils.logging import logger
from src.Sentiment_Analysis.pipeline.prepare_data import DataPreparationPipeline
from src.Sentiment_Analysis.pipeline.main_pipeline import WorkflowRunner
STAGE_NAME = "Data Ingestion stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   
   # Check if train and test data files already exist
   train_data_path = os.path.join("artifacts", "training", "train_data.csv")
   test_data_path = os.path.join("artifacts", "training", "test_data.csv")
   
   if os.path.exists(train_data_path) and os.path.exists(test_data_path):
      logger.info(f"Train data and test data files already exist. Skipping {STAGE_NAME}")
   else:
      logger.info(f"Train data or test data files not found. Running {STAGE_NAME}")
      data_ingestion = DataPreparationPipeline()
      data_ingestion.main()
       
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
      logger.exception(e)
      raise e
    
    
STAGE_NAME = "Full Workflow Run"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   
   runner = WorkflowRunner()
   metrics = runner.run()  
    
   logger.info(f"Final metrics: {metrics}")
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
   logger.exception(e)
   raise e
