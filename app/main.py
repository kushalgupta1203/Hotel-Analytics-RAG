from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.rag import ask_question
from app.analytics import get_analytics_report

app = FastAPI()

class AskQuery(BaseModel):
    query: str

class AnalyticsRequest(BaseModel):
    metrics: List[str]

@app.post("/ask")
async def ask_booking_question(request: AskQuery):
    response = ask_question(request.query)
    return {"answer": response}

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    report = get_analytics_report(request.metrics)
    return {"analytics": report}
