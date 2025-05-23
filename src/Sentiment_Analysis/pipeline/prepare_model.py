from src.Sentiment_Analysis.config.configuration import ConfigurationManager
from src.Sentiment_Analysis.components.base_model import PrepareBaseModel
from src.Sentiment_Analysis.utils.logging import logger
import mlflow
import dagshub

STAGE_NAME = "Prepare base model"


class ModelPreparationPipeline:
    def __init__(self, mlflow_config):
        self.mlflow_config = mlflow_config

    def main(self):
        prepare_base_model_config = ConfigurationManager().get_prepare_base_model_config()

        dagshub.init(
            repo_owner=self.mlflow_config.dagshub_username,
            repo_name=self.mlflow_config.dagshub_repo_name,
            mlflow=True
        )
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        with mlflow.start_run(run_name="MODEL_PREPARATION"):
            mlflow.log_params({
                "maxlen": prepare_base_model_config.params_maxlen,
                "num_words": prepare_base_model_config.params_num_words,
                "embedding_dim": prepare_base_model_config.params_embedding_dim
            })

            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            model, tokenizer = prepare_base_model.full_model()

        return model, tokenizer



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelPreparationPipeline()
        pipeline.main()
        logger.info("Model preparation completed successfully")
    except Exception as e:
        logger.exception(f"Error in Model Preparation Pipeline: {e}")
        raise e
