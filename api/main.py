from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_model import predict

app = FastAPI(
title="Ventas Prediction API",
description="API para predecir ventas según inversión en publicidad",
version="1.0"
)

class PublicidadInput(BaseModel):
    publicidad: float

@app.get("/")
def root():
    return {"message": "API de predicción activa"}

@app.post("/predict")
def predict_sales(data: PublicidadInput):
    result = predict(data.publicidad)
    return {
    "publicidad": data.publicidad,
    "ventas_estimadas": result
    }
    
    # http://localhost:8000/docs#/default/root__get