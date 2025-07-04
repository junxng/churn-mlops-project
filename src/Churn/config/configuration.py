from src.Churn.utils.common import read_yaml, create_directories
from src.Churn.utils.logging import logger
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from src.Churn.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
    CloudStoragePushConfig,
    MLFlowConfig,
    PredictionConfig,
    VisualizationConfig
)
from pathlib import Path

_env_loaded = False

def ensure_env_loaded():
    """Ensure environment variables are loaded only once."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv(override=True)
        _env_loaded = True

class CloudConfig(BaseModel):
    aws_access_key_id: str = Field(default_factory=lambda: (ensure_env_loaded(), os.getenv("AWS_ACCESS_KEY_ID", ""))[1])
    aws_secret_access_key: str = Field(default_factory=lambda: (ensure_env_loaded(), os.getenv("AWS_SECRET_ACCESS_KEY", ""))[1])
    region_name: str = Field(default_factory=lambda: (ensure_env_loaded(), os.getenv("AWS_REGION", ""))[1])

class WebhookConfig(BaseModel):
    url: str = Field(default_factory=lambda: (ensure_env_loaded(), os.getenv("WEB_HOOK", ""))[1])

class ConfigurationManager:
    def __init__(self, config_filepath=Path("config/config.yaml")):
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_mlflow_config(self) -> MLFlowConfig:
        config = self.config.mlflow_config
        return MLFlowConfig(
            dagshub_username=config.dagshub_username,
            dagshub_repo_name=config.dagshub_repo_name,
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            prediction_experiment_name=config.prediction_experiment_name
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            test_size=config.test_size,
            random_state=self.config.random_state,
            data_version_dir=Path(config.data_version_dir),
            columns_to_drop=config.columns_to_drop,
            imbalance_handling=config.imbalance_handling
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        return PrepareBaseModelConfig(
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            n_estimators=config.n_estimators,
            random_state=self.config.random_state,
            criterion=config.criterion,
            max_depth=config.max_depth,
            max_features=config.max_features,
            min_samples_leaf=config.min_samples_leaf
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        return TrainingConfig(
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            accuracy_threshold=config.accuracy_threshold,
            fine_tuning_params=config.fine_tuning_params
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        return EvaluationConfig(
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            evaluation_dir=Path(config.evaluation_dir),
        )

    def get_cloud_storage_push_config(self) -> CloudStoragePushConfig:
        config = self.config.cloud_storage_push
        cloud_config = CloudConfig()
        return CloudStoragePushConfig(
            root_dir=Path(config.root_dir),
            bucket_name=config.bucket_name,
            data_version_dir=Path(config.data_version_dir),
            evaluation_dir=Path(config.evaluation_dir),
            aws_key_id=cloud_config.aws_access_key_id,
            aws_secret_key=cloud_config.aws_secret_access_key,
            region_name=cloud_config.region_name,
            s3_object_prefix=config.s3_object_prefix
        )

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        return PredictionConfig(
            retraining_confidence_threshold=config.retraining_confidence_threshold,
            default_model_version=config.default_model_version,
            default_scaler_version=config.default_scaler_version,
            default_run_id=config.default_run_id,
            default_model_name=config.default_model_name
        )

    def get_visualization_config(self) -> VisualizationConfig:
        config = self.config.visualization
        return VisualizationConfig(
            output_dir=config.output_dir,
            pie_chart=config.pie_chart
        )
    
