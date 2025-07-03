from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import mlflow
from dotenv import load_dotenv
from controller.prediction import router as prediction_router
from controller.retraining import router as retraining_router

# Load environment variables
load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/junxng/churn-mlops-project.mlflow")

# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction and model retraining",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(prediction_router)
app.include_router(retraining_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Churn Prediction API. Use /docs to view the API documentation."}

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
