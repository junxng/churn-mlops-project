from src.Sentiment_Analysis.config.configuration import ConfigurationManager
from ...Sentiment_Analysis.pipeline.prepare_model import ModelPreparationPipeline
from ...Sentiment_Analysis.pipeline.train_evaluation import TrainEvaluationPipeline

class WorkflowRunner:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        # Get MLflow config once
        mlflow_config = self.config_manager.get_mlflow_config()

        # Prepare model
        model_prep = ModelPreparationPipeline(mlflow_config=mlflow_config)
        model, tokenizer = model_prep.main()

        # Train and evaluate
        train_eval = TrainEvaluationPipeline(mlflow_config=mlflow_config)
        metrics = train_eval.main()

        return metrics

