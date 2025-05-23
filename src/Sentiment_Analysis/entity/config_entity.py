from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLFlowConfig:
    dagshub_username: str
    dagshub_repo_name: str
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelParams:
    maxlen: int
    num_words: int
    embedding_dim: int
    batch_size: int
    epochs: int
    filters: int
    kernel_size: int
    dropout_rate: float


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    tokenizer_path: Path
    params_maxlen: int
    params_num_words: int
    params_embedding_dim: int
    dropout_rate: float
    filters: int
    kernel_size: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    bucket_name: str
    tokenizer_path: Path
    training_data: Path
    test_data: Path
    params_epochs: int
    params_batch_size: int
    params_maxlen: int

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    tokenizer_path: Path
    trained_model_path: Path
    training_data: Path
    test_data: Path
    metrics_file: Path
    plots_dir: Path
    params_batch_size: int
    params_maxlen: int
    all_params: dict
    mlflow_uri: str