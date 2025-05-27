from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.base_model import PrepareBaseModel
from src.Churn.utils.logging import logger

STAGE_NAME = "Prepare base model"


class ModelPreparationPipeline:
    def __init__(self):
        pass
    def main(self, train_path, test_path, y_train_path, y_test_path):
        prepare_base_model_config = ConfigurationManager().get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        model, scaler, X_train, X_test, y_train, y_test, base_model_path, scaled_train_path, scaled_test_path, scaler_path = prepare_base_model.full_model(
            train_path=train_path,
            test_path=test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path
        )

        return base_model_path, scaled_train_path, scaled_test_path, scaler_path



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelPreparationPipeline()
        base_model_path, scaled_train_path, scaled_test_path, scaler_path = pipeline.main()
        logger.info("Model preparation completed successfully")
        logger.info(f"Base model saved at: {base_model_path}")
        logger.info(f"Scaler saved at: {scaler_path}")
        logger.info(f"Scaled train data: {scaled_train_path}")
        logger.info(f"Scaled test data: {scaled_test_path}")
    except Exception as e:
        logger.exception(f"Error in Model Preparation Pipeline: {e}")
        raise e
