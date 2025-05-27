from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.model_training import TrainAndEvaluateModel
from src.Churn.utils.logging import logger



STAGE_NAME = "TRAIN_AND_EVALUATE_MODEL"

class TrainEvaluationPipeline:
    def __init__(self):
        pass

    def main(self, base_model_path=None, scaled_train_path=None, scaled_test_path=None, y_train_path=None, y_test_path=None):
        training_config = ConfigurationManager().get_training_config()
        evaluation_config = ConfigurationManager().get_evaluation_config()

        model_processor = TrainAndEvaluateModel(
            config_train=training_config,
            config_eval=evaluation_config
        )
        
        # Use provided paths or default config paths
        if all([base_model_path, scaled_train_path, scaled_test_path, y_train_path, y_test_path]):
            model, metrics, final_model_path = model_processor.train_and_evaluate(
                base_model_path=base_model_path,
                scaled_train_path=scaled_train_path,
                scaled_test_path=scaled_test_path,
                y_train_path=y_train_path,
                y_test_path=y_test_path
            )
        else:
            base_model_path = training_config.base_model_path
            scaled_train_path = training_config.scaled_train_data
            scaled_test_path = training_config.scaled_test_data  
            y_train_path = training_config.target_train
            y_test_path = training_config.target_test
            
            model, metrics, final_model_path = model_processor.train_and_evaluate(
                base_model_path=base_model_path,
                scaled_train_path=scaled_train_path,
                scaled_test_path=scaled_test_path,
                y_train_path=y_train_path,
                y_test_path=y_test_path
            )

        return model, metrics, final_model_path



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = TrainEvaluationPipeline()
        model, metrics, final_model_path = pipeline.main()
        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Final model saved at: {final_model_path}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(f"Error in Train-Evaluation Pipeline: {e}")
        raise e 