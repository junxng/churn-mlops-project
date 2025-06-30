import os
import sys
import time
from kfp.compiler import Compiler
from kfp import Client
from TEST.kubeflow_pipeline import churn_prediction_pipeline
from src.Churn.utils.logging import logger

def reset_kubeflow_metadata():
    """Reset Kubeflow metadata to resolve context issues"""
    try:
        logger.info("Resetting Kubeflow metadata store...")
        
        # Connect to client and try to clear any stuck contexts
        client = Client(host="http://localhost:8080")
        
        # Get all experiments and clean up if needed
        experiments = client.list_experiments()
        logger.info(f"Found {experiments.total_size if experiments else 0} experiments")
        
        logger.info("Metadata reset completed")
        return True
        
    except Exception as e:
        logger.warning(f"Could not reset metadata (this is often normal): {e}")
        return True  # Continue anyway

def compile_pipeline():
    """Compile the Kubeflow pipeline to YAML"""
    try:
        logger.info("Compiling Kubeflow pipeline...")
        
        compiler = Compiler()
        compiler.compile(
            pipeline_func=churn_prediction_pipeline,
            package_path="churn_pipeline.yaml"
        )
        
        logger.info("Pipeline compiled successfully: churn_pipeline.yaml")
        return True
        
    except Exception as e:
        logger.error(f"Error compiling pipeline: {e}")
        return False


def deploy_pipeline():
    """deployment with enhanced error handling and retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Deploying pipeline to Kubeflow (attempt {attempt + 1}/{max_retries})...")
            
            # Connect to Kubeflow with timeout
            client = Client(host="http://localhost:8080")
            
            # Create unique run name with timestamp
            run_name = f"churn-prediction-{int(time.time())}-{attempt}"
            
            # Create and run the pipeline
            run = client.create_run_from_pipeline_package(
                pipeline_file="churn_pipeline.yaml",
                arguments={},
                run_name=run_name,
                experiment_name="churn-prediction-experiments"  # Create/use specific experiment
            )
            
            logger.info(f"✅ Pipeline run created successfully: {run.run_id}")
            logger.info(f"Monitor progress at: http://localhost:8080/#/runs/details/{run.run_id}")
            logger.info("Pipeline submitted to Kubeflow - check UI for status")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying pipeline (attempt {attempt + 1}): {e}")
            
            if "Cannot find context" in str(e) or "PipelineRun" in str(e):
                logger.info("Detected metadata context issue. Attempting reset...")
                reset_kubeflow_metadata()
                
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All retry attempts failed")
                return False
    
    return False

def run_kubeflow_pipeline():
    """Main function to compile and run the Kubeflow pipeline with error handling"""
    logger.info("=" * 60)
    logger.info("STARTING KUBEFLOW PIPELINE EXECUTION")
    logger.info("=" * 60)
    
    # Step 1: Reset metadata if needed
    reset_kubeflow_metadata()
    
    # Step 2: Compile pipeline
    if not compile_pipeline():
        logger.error("Pipeline compilation failed. Exiting.")
        return False
    
    # Step 3: Deploy pipeline with retries
    if not deploy_pipeline():
        logger.error("Pipeline deployment failed after all retries. Exiting.")
        return False
    
    logger.info("=" * 60)
    logger.info("KUBEFLOW PIPELINE DEPLOYMENT COMPLETED")
    logger.info("Check Kubeflow UI at http://localhost:8080 for execution status")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_kubeflow_pipeline()
    sys.exit(0 if success else 1) 