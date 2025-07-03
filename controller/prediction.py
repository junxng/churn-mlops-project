from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.Churn.pipeline.prediction import ChurnController

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"],
    responses={404: {"description": "Not found"}},
)


class ChurnResponse(BaseModel):
    message: str

@router.post("/", response_model=ChurnResponse)
async def predict_churn(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_version: str = Form(default="1"),
    scaler_version: str = Form(default="scaler_churn_version_20250703T150200.pkl"),
    run_id: str = Form(default="d70ecaf6c6e84537826bdbad64166436"),
    model_name: str = Form(default="RandomForestClassifier"),
):
    return await ChurnController.predict_churn(background_tasks=background_tasks, file=file, model_version=model_version, scaler_version=scaler_version, run_id=run_id, model_name=model_name)
