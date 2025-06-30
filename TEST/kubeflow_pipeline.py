from kfp import dsl
from kfp.dsl import (
    Input,
    Output,
    Artifact,
    Model,
    Metrics,
    component
)

import os
# from collections import namedtuple
from typing import NamedTuple
import shutil
# import zipfile
# from src.Churn.pipeline.prepare_data import DataPreparationStage
# from src.Churn.pipeline.prepare_model import ModelPreparationStage
# from src.Churn.pipeline.train_evaluation import TrainEvaluationStage
# from src.Churn.pipeline.cloud_storage_push import CloudStoragePushPipeline
# from src.Churn.config.configuration import ConfigurationManager
# from src.Churn.utils.logging import logger
# import glob

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def data_preparation_component(
    train_data: Output[Artifact],
    test_data: Output[Artifact], 
    y_train_data: Output[Artifact],
    y_test_data: Output[Artifact]
) -> NamedTuple('Outputs', [('train_path', str), ('test_path', str), ('y_train_path', str), ('y_test_path', str)]):
    """Data preparation component for churn prediction using existing DataPreparationStage"""
    
    from collections import namedtuple
    from src.Churn.pipeline.prepare_data import DataPreparationStage
    from src.Churn.utils.logging import logger
    
    logger.info(">>> Stage Data Ingestion started <<<")
    
    train_path_local, test_path_local, y_train_path_local, y_test_path_local = DataPreparationStage()
    
    # Copy data to KFP artifacts
    shutil.copy2(train_path_local, train_data.path)
    shutil.copy2(test_path_local, test_data.path)
    shutil.copy2(y_train_path_local, y_train_data.path)
    shutil.copy2(y_test_path_local, y_test_data.path)
    
    logger.info(">>> Stage Data Ingestion completed <<<")
    
    outputs = namedtuple('Outputs', ['train_path', 'test_path', 'y_train_path', 'y_test_path'])
    return outputs(train_data.path, test_data.path, y_train_data.path, y_test_data.path)


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def model_preparation_component(
    train_data: Input[Artifact],
    test_data: Input[Artifact],
    y_train_data: Input[Artifact],
    y_test_data: Input[Artifact],
    base_model: Output[Model],
    scaler_artifact: Output[Artifact],
    scaled_train: Output[Artifact],
    scaled_test: Output[Artifact]
) -> NamedTuple('Outputs', [('base_model_path', str), ('scaled_train_path', str), ('scaled_test_path', str), ('scaler_path', str)]):
    """Model preparation component using existing ModelPreparationStage"""

    # Import required modules inside the component
    import shutil
    from collections import namedtuple
    from src.Churn.pipeline.prepare_model import ModelPreparationStage
    from src.Churn.utils.logging import logger

    logger.info(">>> Stage Prepare base model started <<<")
    
    # Call the existing pipeline function - matches main_pipeline.py exactly
    base_model_path_local, scaled_train_path_local, scaled_test_path_local, scaler_path_local = ModelPreparationStage(
        train_path=train_data.path,
        test_path=test_data.path,
        y_train_path=y_train_data.path,
        y_test_path=y_test_data.path
    )
    
    shutil.copy2(base_model_path_local, base_model.path)
    shutil.copy2(scaler_path_local, scaler_artifact.path)
    shutil.copy2(scaled_train_path_local, scaled_train.path)
    shutil.copy2(scaled_test_path_local, scaled_test.path)
    
    logger.info(">>> Stage Prepare base model completed <<<")
    
    outputs = namedtuple('Outputs', ['base_model_path', 'scaled_train_path', 'scaled_test_path', 'scaler_path'])
    return outputs(base_model.path, scaled_train.path, scaled_test.path, scaler_artifact.path)


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib", "matplotlib", "seaborn"]
)
def train_evaluation_component(
    base_model: Input[Model],
    scaled_train: Input[Artifact],
    scaled_test: Input[Artifact],
    y_train_data: Input[Artifact],
    y_test_data: Input[Artifact],
    final_model: Output[Model],
    metrics: Output[Metrics],
    evaluation_plots: Output[Artifact]
) -> NamedTuple('Outputs', [('final_model_path', str), ('accuracy', float), ('roc_auc', float)]):
    """Training and evaluation component using existing TrainEvaluationStage"""
    
    # Import required modules inside the component
    import os
    import shutil
    import zipfile
    from collections import namedtuple
    from src.Churn.pipeline.train_evaluation import TrainEvaluationStage
    from src.Churn.config.configuration import ConfigurationManager
    from src.Churn.utils.logging import logger
    
    logger.info(">>> Stage TRAIN_AND_EVALUATE_MODEL started <<<")
    
    # Call the existing pipeline function - matches main_pipeline.py exactly
    model, metrics_dict, final_model_path_local = TrainEvaluationStage(
        base_model_path=base_model.path,
        scaled_train_path=scaled_train.path,
        scaled_test_path=scaled_test.path,
        y_train_path=y_train_data.path,
        y_test_path=y_test_data.path
    )
    
    # Copy final model to KFP artifact
    shutil.copy2(final_model_path_local, final_model.path)
    
    # Log metrics to KFP
    metrics.log_metric('accuracy', metrics_dict.get('accuracy', 0.0))
    metrics.log_metric('roc_auc', metrics_dict.get('roc_auc', 0.0))
    metrics.log_metric('precision', metrics_dict.get('precision', 0.0))
    metrics.log_metric('recall', metrics_dict.get('recall', 0.0))
    metrics.log_metric('f1_score', metrics_dict.get('f1_score', 0.0))
    
    # Create evaluation plots zip file from the plots directory
    config_manager = ConfigurationManager()
    eval_dir = config_manager.get_evaluation_config().plots_dir
    
    if os.path.exists(eval_dir):
        with zipfile.ZipFile(evaluation_plots.path, 'w') as zipf:
            for root, dirs, files in os.walk(eval_dir):
                for file in files:
                    if file.endswith('.png') or file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, eval_dir)
                        zipf.write(file_path, arcname)
    
    logger.info(">>> Stage TRAIN_AND_EVALUATE_MODEL completed <<<")

    outputs = namedtuple('Outputs', ['final_model_path', 'accuracy', 'roc_auc'])
    return outputs(final_model.path, metrics_dict.get('accuracy', 0.0), metrics_dict.get('roc_auc', 0.0))


@component(
    base_image="python:3.11",
    packages_to_install=["boto3", "google-cloud-storage"]
)
def cloud_storage_push_component() -> str:
    """Cloud storage push component using existing CloudStoragePushPipeline - matches main_pipeline.py exactly"""
    
    # Import required modules inside the component
    from src.Churn.pipeline.cloud_storage_push import cloud_storage_push_component
    from src.Churn.utils.logging import logger
    
    logger.info(">>> Stage Cloud Storage Push started <<<")
    
    # Call the existing pipeline class and main method - matches main_pipeline.py exactly
    cloud_storage_push_component()
    
    
    logger.info(">>> Stage Cloud Storage Push completed <<<")
    return "Cloud storage push completed successfully"


@component(
    base_image="python:3.11"
)
def cleanup_component() -> str:
    """Cleanup component using existing cleanup logic from main_pipeline.py"""

    # Import required modules inside the component
    import os
    import glob
    from src.Churn.config.configuration import ConfigurationManager
    from src.Churn.utils.logging import logger

    logger.info("=" * 50)
    logger.info("CLEANUP: Removing temporary versioned files")
    logger.info("=" * 50)
    
    try:
        config_manager = ConfigurationManager()
        data_version_dir = config_manager.config.data_ingestion.data_version_dir
        evaluation_dir = config_manager.config.evaluation.plots_dir
        
        # Pattern: *_version_YYYYMMDDTHHMMSS.csv
        data_version_files = glob.glob(os.path.join(data_version_dir, "*_version_????????T??????.csv"))
        logger.info(f"Found {len(data_version_files)} timestamp-versioned data files to clean")
        
        for file_path in data_version_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        # Pattern: *_YYYYMMDDTHHMMSS.json and *_YYYYMMDDTHHMMSS.png
        eval_json_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.json"))
        eval_png_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.png"))
        eval_files = eval_json_files + eval_png_files
        logger.info(f"Found {len(eval_files)} timestamp-versioned evaluation files to clean")
        
        for file_path in eval_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        remaining_data_files = len(glob.glob(os.path.join(data_version_dir, "*.csv")))
        remaining_eval_files = len(glob.glob(os.path.join(evaluation_dir, "*")))
        logger.info(f"Kept {remaining_data_files} essential data files for DVC tracking")
        logger.info(f"Kept {remaining_eval_files} essential evaluation files for DVC tracking")
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    logger.info("Cleanup completed")
    return "Cleanup completed successfully"


@dsl.pipeline(
    name='churn-prediction-pipeline',
    description='Churn prediction pipeline with artifact tracking - matches main_pipeline.py logic'
)
def churn_prediction_pipeline():
    """Main pipeline definition with proper artifact tracking and same logic as main_pipeline.py"""
    
    # Step 1: Data preparation
    data_prep_task = data_preparation_component()
    
    # Step 2: Model preparation
    model_prep_task = model_preparation_component(
        train_data=data_prep_task.outputs['train_data'],
        test_data=data_prep_task.outputs['test_data'],
        y_train_data=data_prep_task.outputs['y_train_data'],
        y_test_data=data_prep_task.outputs['y_test_data']
    )
    model_prep_task.after(data_prep_task)
    
    # Step 3: Training and evaluation
    train_eval_task = train_evaluation_component(
        base_model=model_prep_task.outputs['base_model'],
        scaled_train=model_prep_task.outputs['scaled_train'],
        scaled_test=model_prep_task.outputs['scaled_test'],
        y_train_data=data_prep_task.outputs['y_train_data'],
        y_test_data=data_prep_task.outputs['y_test_data']
    )
    train_eval_task.after(model_prep_task)
    
    # Step 4: Cloud storage push
    cloud_push_task = cloud_storage_push_component()
    cloud_push_task.after(train_eval_task)
    
    # Step 5: Cleanup
    cleanup_task = cleanup_component()
    cleanup_task.after(cloud_push_task)


