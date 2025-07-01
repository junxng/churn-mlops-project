from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.model_training import TrainAndEvaluateModel
from src.Churn.utils.logging import logger
import mlflow
import dagshub

STAGE_NAME = "TRAIN_AND_EVALUATE_MODEL"


class TrainEvaluationPipeline:
    def __init__(self, mlflow_config):
        self.mlflow_config = mlflow_config
    def main(self, base_model, X_train_scaled, X_test_scaled, y_train, y_test):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        training_config = ConfigurationManager().get_training_config()
        evaluation_config = ConfigurationManager().get_evaluation_config()
        
        # No need to start a new MLflow run here as we'll use the one from main_pipeline
        model_processor = TrainAndEvaluateModel(
            config_train=training_config,
            config_eval=evaluation_config
        )
        
        model, metrics, final_model_path = model_processor.train_and_evaluate(
            base_model=base_model,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test
        )
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")
        return model, metrics, final_model_path

