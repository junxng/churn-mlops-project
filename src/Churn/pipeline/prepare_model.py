from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.base_model import PrepareBaseModel
from src.Churn.utils.logging import logger
import mlflow
import dagshub
STAGE_NAME = "Prepare base model"


class ModelPreparationPipeline:
    def __init__(self, mlflow_config):
        self.mlflow_config = mlflow_config
    def main(self, X_train, X_test):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        prepare_base_model_config = ConfigurationManager().get_prepare_base_model_config()
        
        mlflow.log_params({
            "n_estimators": prepare_base_model_config.n_estimators,
            "random_state": prepare_base_model_config.random_state,
            "criterion": prepare_base_model_config.criterion,
            "max_depth": prepare_base_model_config.max_depth,
            "max_features": prepare_base_model_config.max_features,
            "min_samples_leaf": prepare_base_model_config.min_samples_leaf
        })
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        model, base_model_path, scaler_path, X_train_scaled, X_test_scaled = prepare_base_model.full_model(
            X_train=X_train,
            X_test=X_test,
        )
        
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")
        return model, base_model_path, scaler_path, X_train_scaled, X_test_scaled
