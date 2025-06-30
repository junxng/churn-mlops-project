from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.Churn.pipeline.prediction import ChurnController

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"],
    responses={404: {"description": "Not found"}},
)

class ChurnResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/", response_model=ChurnResponse)
async def predict_churn(
    file: UploadFile = File(...),
    model_uri: Optional[str] = Form(default="models:/RandomForestClassifier/1"),
    scaler_uri: Optional[str] = Form(default="runs:/f3ab09385e414fd2abf29d80f74cd67a/scaler_churn_version_20250625T154746.pkl")
):
    return await ChurnController.predict_churn(file=file, model_uri=model_uri, scaler_uri=scaler_uri)
