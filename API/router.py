from fastapi import APIRouter, File, UploadFile, Form, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .ML_controller import ChurnController

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"],
    responses={404: {"description": "Not found"}},
)

class ChurnResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/", response_model=ChurnResponse)
async def predict_churn(
    file: UploadFile = File(...)
):
    return await ChurnController.predict_churn(file=file)