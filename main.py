import os
import multiprocessing
from src.Churn.utils.logging import logger
from src.Churn.pipeline.prepare_data import DataPreparationPipeline
from src.Churn.pipeline.main_pipeline import WorkflowRunner

def main():
    """Main execution function with proper multiprocessing protection."""

        
        
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

if __name__ == '__main__':
    # Multiprocessing protection for Windows
    multiprocessing.freeze_support()
    
    # Run the main function
    main()
