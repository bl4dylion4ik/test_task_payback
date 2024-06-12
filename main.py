from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('gradient_boosting_tune.pkl')

app = FastAPI()


class PredictionRequest(BaseModel):
    features: List[float]

# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

