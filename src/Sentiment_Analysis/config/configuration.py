from src.Sentiment_Analysis.utils.common import read_yaml, create_directories
from src.Sentiment_Analysis.utils.logging import logger
from datetime import datetime

from src.Sentiment_Analysis.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ModelParams,
    MLFlowConfig
)
from pathlib import Path
import os


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=Path("config/config.yaml")
    ):
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_mlflow_config(self, model_version: str | None = None) -> MLFlowConfig:
        config = self.config.mlflow_config
        base_experiment_name = config.experiment_name

        if model_version is not None:
            version = model_version
            logger.info(f"Using user-provided version: {version}")
            experiment_name_with_version = f"{base_experiment_name}_v{version}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name_with_version = f"{base_experiment_name}_{timestamp}"
            logger.info(f"No version input, using timestamp: {experiment_name_with_version}")

        mlflow_config = MLFlowConfig(
            dagshub_username=config.dagshub_username,
            dagshub_repo_name=config.dagshub_repo_name,
            tracking_uri=config.tracking_uri,
            experiment_name=experiment_name_with_version
        )

        logger.info(f"MLFlow configuration: {mlflow_config}")
        return mlflow_config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            test_size=config.test_size,
            random_state=config.random_state
        )

        logger.info(f"Data Ingestion config: {config}")
        return data_ingestion_config
        
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        model_params = self.config.model_params
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            tokenizer_path=Path(config.tokenizer_path),
            params_maxlen=model_params.maxlen,
            params_num_words=model_params.num_words,
            params_embedding_dim=model_params.embedding_dim,
            dropout_rate=model_params.dropout_rate,
            filters=model_params.filters,
            kernel_size=model_params.kernel_size
        )

        logger.info(f"Prepare base model config: {config}")
        return prepare_base_model_config

    def get_model_params(self) -> ModelParams:
        config = self.config.model_params
        
        model_params = ModelParams(
            maxlen=config.maxlen,
            num_words=config.num_words,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
            epochs=config.epochs,
            filters=config.filters,
            kernel_size=config.kernel_size,
            dropout_rate=config.dropout_rate
        )

        logger.info(f"Model parameters: {config}")
        return model_params

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        model_params = self.config.model_params
        prepare_base_model = self.config.prepare_base_model
        
        training_data = os.path.join(self.config.data_ingestion.root_dir, "train_data.csv")
        
        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            tokenizer_path=Path(prepare_base_model.tokenizer_path),
            training_data=Path(config.train_data_path),
            test_data=Path(config.test_data_path),
            bucket_name=config.bucket_name,
            params_epochs=model_params.epochs,
            params_batch_size=model_params.batch_size,
            params_maxlen=model_params.maxlen
        )

        logger.info(f"Training config: {config}")
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        model_params = self.config.model_params
        training_config = self.config.training
        mlflow_config = self.config.mlflow_config
        prepare_base_model = self.config.prepare_base_model
        
        create_directories([config.root_dir, config.plots_dir])

        evaluation_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            tokenizer_path=Path(prepare_base_model.tokenizer_path),
            trained_model_path=Path(training_config.trained_model_path),
            training_data=Path(training_config.train_data_path),
            test_data=Path(training_config.test_data_path),
            metrics_file=Path(config.metrics_file),
            plots_dir=Path(config.plots_dir),
            params_maxlen=model_params.maxlen,
            params_batch_size=model_params.batch_size,
            all_params=dict(self.config.model_params),
            mlflow_uri=mlflow_config.tracking_uri
        )

        logger.info(f"Evaluation config: {config}")
        return evaluation_config
