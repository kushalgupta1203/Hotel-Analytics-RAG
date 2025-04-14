from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.rag import ask_question
from app.analytics import get_analytics_report
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os


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

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
