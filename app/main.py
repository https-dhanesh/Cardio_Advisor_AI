import logging
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import services
from schemas import PatientData, RiskResponse, AdviceResponse
from ai_core import models 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cardio-advisor-server")

app = FastAPI(
    title="Cardio Advisor AI (v3 - Modular)",
    description="High-performance, two-stage cardiovascular risk assessment system.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():

    logger.info("Server starting up...")

    models.load_fast_models()

    logger.info("Starting background thread for SLOW models...")
    init_thread = threading.Thread(target=models.load_slow_models, daemon=True)
    init_thread.start()

@app.get("/")
async def root():
    return {
        "message": "Cardio Advisor AI v3",
        "endpoints": {
            "/status": "Check status of all AI models",
            "/predict": "FAST: Get Risk Score + SHAP",
            "/advice": "SLOW: Get LLM/RAG AI Advice"
        }
    }

@app.get("/status")
async def status_check():

    return {
        "fast_models_loaded": {
            "xgb_loaded": models.xgb_loaded,
            "shap_loaded": models.shap_loaded
        },
        "slow_models_loaded": {
            "rag_loaded": models.rag_loaded,
            "llm_loaded": models.llm_loaded
        }
    }

@app.post("/predict", response_model=RiskResponse)
async def predict_risk(patient_data: PatientData):

    if not (models.xgb_loaded and models.shap_loaded):
        raise HTTPException(status_code=503, detail="Fast models are not yet loaded.")
    
    try:
        result = services.get_risk_prediction_and_explanation(patient_data)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/advice", response_model=AdviceResponse)
async def get_advice(patient_data: PatientData):

    if not models.llm_loaded:
        raise HTTPException(
            status_code=503, 
            detail="AI advice model is still loading. Please try again in a minute."
        )
    
    try:
        result = services.generate_ai_advice(patient_data)
        return result
    except Exception as e:
        logger.error(f"Advice generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Cardio Advisor AI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")