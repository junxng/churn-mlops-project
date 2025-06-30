import multiprocessing
from src.Churn.utils.logging import logger
from src.Churn.pipeline.main_pipeline import WorkflowRunner
import asyncio

async def async_main():
    """Async main execution function."""
    STAGE_NAME = "Full Workflow Run"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        runner = WorkflowRunner()
        await runner.run()  
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e

def main():
    """Main execution function with proper multiprocessing protection."""
    # Run the async main function
    asyncio.run(async_main())

if __name__ == '__main__':
    # Multiprocessing protection for Windows
    multiprocessing.freeze_support()
    
    # Run the main function
    main()
