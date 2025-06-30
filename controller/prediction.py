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
    model_version: Optional[str] = Form(default="1"),
    scaler_version: Optional[str] = Form(default="scaler_churn_version_20250625T154746.pkl"),
    run_id: Optional[str] = Form(default="f3ab09385e414fd2abf29d80f74cd67a"),
):
    return await ChurnController.predict_churn(background_tasks=background_tasks, file=file, model_version=model_version, scaler_version=scaler_version, run_id=run_id)
