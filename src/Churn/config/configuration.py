from src.Churn.utils.common import read_yaml, create_directories
from src.Churn.utils.logging import logger

from src.Churn.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
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


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            test_size=config.test_size,
            random_state=config.random_state,
            bucket_name=config.bucket_name,
            data_version_dir=Path(config.data_version_dir)
        )

        logger.info(f"Data Ingestion config: {config}")
        return data_ingestion_config
        
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            criterion=config.criterion,
            max_depth=config.max_depth,
            max_features=config.max_features,
            min_samples_leaf=config.min_samples_leaf
        )

        logger.info(f"Prepare base model config: {config}")
        return prepare_base_model_config


    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        
        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            bucket_name=config.bucket_name
        )

        logger.info(f"Training config: {config}")
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        
        create_directories([config.root_dir, config.plots_dir])

        evaluation_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            plots_dir=Path(config.plots_dir)
        )

        logger.info(f"Evaluation config: {config}")
        return evaluation_config
