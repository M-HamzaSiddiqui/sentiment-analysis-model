from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "model/imdb_sentiment_svm.pkl")

model = joblib.load(model_path)

class RequestBody(BaseModel):
    responses: List[str]
    
@app.post("/api/sentiment")
def analyze_sentiment(data: RequestBody):
    predictions = model.predict(data.responses)
    result = {"positive": 0, "negative": 0}
    
    for pred in predictions:
        if pred == 1:
            result["positive"] += 1
        else:
            result["negative"] += 1
    
    return result