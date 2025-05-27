from src.Churn.config.configuration import ConfigurationManager
from src.Churn.pipeline.prepare_data import DataPreparationPipeline
from src.Churn.pipeline.prepare_model import ModelPreparationPipeline
from src.Churn.pipeline.train_evaluation import TrainEvaluationPipeline
from src.Churn.utils.logging import logger
import os
import glob

class WorkflowRunner:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def cleanup_temp_files(self, data_version_dir):
        """Clean up temporary files after pipeline completion."""
        logger.info("=" * 50)
        logger.info("CLEANUP: Removing temporary files")
        logger.info("=" * 50)
        
        try:
            # Clean up data version files
            version_files = glob.glob(os.path.join(data_version_dir, "*_version_*.csv"))
            for file_path in version_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def run(self):
        """Run the complete workflow with proper path passing between stages."""
        
        # Stage 1: Data Preparation
        logger.info("=" * 50)
        logger.info("STAGE 1: Data Preparation")
        logger.info("=" * 50)
        
        data_prep = DataPreparationPipeline()
        train_path, test_path, y_train_path, y_test_path = data_prep.main()
        
        logger.info("Data preparation completed successfully")
        logger.info(f"Train data: {train_path}")
        logger.info(f"Test data: {test_path}")
        logger.info(f"Train targets: {y_train_path}")
        logger.info(f"Test targets: {y_test_path}")

        # Stage 2: Model Preparation
        logger.info("=" * 50)
        logger.info("STAGE 2: Model Preparation")
        logger.info("=" * 50)
        
        model_prep = ModelPreparationPipeline()
        base_model_path, scaled_train_path, scaled_test_path, scaler_path = model_prep.main(
            train_path=train_path,
            test_path=test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path
        )
        
        logger.info("Model preparation completed successfully")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Scaler: {scaler_path}")
        logger.info(f"Scaled train data: {scaled_train_path}")
        logger.info(f"Scaled test data: {scaled_test_path}")

        # Stage 3: Train and Evaluate
        logger.info("=" * 50)
        logger.info("STAGE 3: Train and Evaluate")
        logger.info("=" * 50)
        
        train_eval = TrainEvaluationPipeline()
        final_model, metrics, final_model_path = train_eval.main(
            base_model_path=base_model_path,
            scaled_train_path=scaled_train_path,
            scaled_test_path=scaled_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path
        )
        
        logger.info("Training and evaluation completed successfully")
        logger.info(f"Final model: {final_model_path}")
        logger.info(f"Metrics: {metrics}")

        self.cleanup_temp_files(self.config_manager.config.data_ingestion.data_version_dir)

        return final_model_path

