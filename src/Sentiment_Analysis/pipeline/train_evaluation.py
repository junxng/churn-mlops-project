from src.Sentiment_Analysis.config.configuration import ConfigurationManager
from src.Sentiment_Analysis.components.model_training import TrainAndEvaluateModel
from src.Sentiment_Analysis.utils.logging import logger
import mlflow
import dagshub


STAGE_NAME = "TRAIN_AND_EVALUATE_MODEL"

class TrainEvaluationPipeline:
    def __init__(self, mlflow_config):
        self.mlflow_config = mlflow_config

    def main(self):
        training_config = ConfigurationManager().get_training_config()
        evaluation_config = ConfigurationManager().get_evaluation_config()

        dagshub.init(
            repo_owner=self.mlflow_config.dagshub_username,
            repo_name=self.mlflow_config.dagshub_repo_name,
            mlflow=True
        )
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        with mlflow.start_run(run_name="TRAIN_AND_EVALUATE"):
            mlflow.log_params({
                "batch_size": training_config.params_batch_size,
                "maxlen": training_config.params_maxlen,
                "epochs": training_config.params_epochs
            })

            model_processor = TrainAndEvaluateModel(
                config_train=training_config,
                config_eval=evaluation_config
            )
            model, history, metrics = model_processor.train_and_evaluate()

        return metrics



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = TrainEvaluationPipeline()
        metrics = pipeline.main()
        logger.info(f"Final metrics: {metrics}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(f"Error in Train-Evaluation Pipeline: {e}")
        raise e 