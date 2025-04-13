from pydantic import BaseModel
from typing import List

class AskQuery(BaseModel):
    query: str

class AnalyticsRequest(BaseModel):
    metrics: List[str]
