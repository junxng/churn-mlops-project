from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_version_dir: Path
    local_data_file: Path
    test_size: float
    random_state: int
    bucket_name: str



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    model_version_dir: Path
    data_version_dir: Path
    random_state: int
    n_estimators: int
    criterion: str
    max_depth: int
    max_features: str
    min_samples_leaf: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_version_dir: Path
    data_version_dir: Path
    bucket_name: str


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_version_dir: Path
    data_version_dir: Path
    plots_dir: Path