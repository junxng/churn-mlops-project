from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import  Optional
from src.Churn.pipeline.main_pipeline import WorkflowRunner
from src.Churn.utils.logging import logger
from pathlib import Path

router = APIRouter(
    prefix="/workflow",
    tags=["Model Training Workflow"],
    responses={404: {"description": "Not found"}},
)

class WorkflowResponse(BaseModel):
    status: str
    message: str
    final_model_path: Optional[str] = None



@router.post("/train", response_model=WorkflowResponse)
async def train_model(
    file: Optional[UploadFile] = File(None)
):
    """
    Run the complete model training workflow.
    
    - **file**: Optional CSV/Excel file for training data. If not provided, will use existing data file.
    """
    try:
        workflow_runner = WorkflowRunner()
        final_model_path = await workflow_runner.run(uploaded_file=file)
        
        return WorkflowResponse(
            status="success",
            message="Model training workflow completed successfully",
            final_model_path=str(final_model_path) if isinstance(final_model_path, Path) else final_model_path,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Workflow failed with unexpected error: {str(e)}"
        )

@router.get("/status")
async def get_workflow_status():
    """
    Check the status of the workflow system.
    """
    try:
        workflow_runner = WorkflowRunner()
        data_file_exists = workflow_runner.check_data_file_exists()
        
        return {
            "status": "ready",
            "data_file_exists": data_file_exists,
            "message": "Workflow system is ready" if data_file_exists else "No data file found - upload required"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )