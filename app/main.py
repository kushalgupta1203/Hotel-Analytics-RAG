import sys
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss

# === Setup Paths ===
FAISS_INDEX_PATH = "D:/Projects/Hotel-Analytics-RAG/pipeline/hotel_faiss.index"
EMBEDDINGS_PATH = "D:/Projects/Hotel-Analytics-RAG/pipeline/embeddings.pkl"
DB_PATH = "D:/Projects/Hotel-Analytics-RAG/dataset/analytics.db"
DATA_PATH = "D:/Projects/Hotel-Analytics-RAG/dataset/hotel_bookings_dataset.csv"

# === Load Resources ===
df = pd.read_csv(DATA_PATH)

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)

texts = df.astype(str).apply(lambda x: ". ".join(x), axis=1).tolist()

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# === FastAPI App ===
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


class Query(BaseModel):
    question: str


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def fetch_all_results(cursor, query, params=()):
    cursor.execute(query, params)
    return cursor.fetchall()


def ask_question(query, max_tokens=200):
    conn = get_db_connection()
    cursor = conn.cursor()
    q_lower = query.lower()

    # === Revenue Queries ===
    if "total revenue" in q_lower:
        import re
        match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", q_lower)
        if match:
            month = match.group(1).capitalize()
            year = int(match.group(2))
            cursor.execute("SELECT revenue FROM revenue_insights WHERE month = ? AND year = ?", (month, year))
            result = cursor.fetchone()
            return f"Total revenue for {month} {year} is {result[0]}." if result else f"No revenue data found for {month} {year}."

    # === Highest Cancellations by Location ===
    elif "highest booking cancellations" in q_lower:
        cursor.execute("SELECT country, COUNT(*) FROM hotel_bookings WHERE is_canceled = 1 GROUP BY country ORDER BY COUNT(*) DESC LIMIT 5")
        results = cursor.fetchall()
        if results:
            return "Top countries with the highest booking cancellations:\n" + "\n".join([f"{r[0]}: {r[1]} cancellations" for r in results])
        else:
            return "No cancellation data found."

    # === Average ADR ===
    elif "average adr" in q_lower or "average price" in q_lower:
        avg_adr = df['adr'].mean()
        return f"The average price of a hotel booking (ADR) is approximately {avg_adr:.2f}."

    # === Cancellations by Date ===
    elif "most cancellations" in q_lower and "date" in q_lower:
        cancelled_df = df[df["is_canceled"] == 1]
        cancelled_by_date = cancelled_df.groupby("reservation_status_date").size().sort_values(ascending=False).head(5)
        return "Top dates with most cancellations:\n" + "\n".join([f"{date}: {count}" for date, count in cancelled_by_date.items()])

    # === Most Bookings by Country ===
    elif "most bookings" in q_lower and "country" in q_lower:
        top_countries = df['country'].value_counts().head(5)
        return "Countries with most bookings:\n" + "\n".join([f"{country}: {count}" for country, count in top_countries.items()])

    else:
        query_embedding = embedder.encode([query])
        D, I = faiss_index.search(np.array(query_embedding), 5)
        context = "\n".join([texts[i] for i in I[0]])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        result = llama_pipeline(prompt, max_new_tokens=max_tokens)
        return result[0]['generated_text']


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_api(q: Query):
    return {"answer": ask_question(q.question)}


@app.post("/analytics")
async def analytics():
    conn = get_db_connection()
    cursor = conn.cursor()

    # === Revenue Insights ===
    cursor.execute("SELECT month, year, revenue FROM revenue_insights ORDER BY year, month")
    revenue_data = cursor.fetchall()

    # === Cancellation Insights ===
    cursor.execute("SELECT country, COUNT(*) FROM hotel_bookings WHERE is_canceled = 1 GROUP BY country ORDER BY COUNT(*) DESC LIMIT 5")
    cancellation_data = cursor.fetchall()

    conn.close()

    return {
        "revenue_trends": [{"month": m, "year": y, "revenue": r} for m, y, r in revenue_data],
        "top_cancellation_locations": [{"location": loc, "cancellations": c} for loc, c in cancellation_data]
    }
