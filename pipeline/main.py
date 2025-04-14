import sys
import os

# Dynamically add the 'pipeline' folder to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'pipeline'))

import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

from utils import row_to_text, get_db_connection, fetch_all_results
from embeddings import generate_or_load_embeddings
from faiss_index import create_or_load_faiss_index, retrieve_faiss
from db_operations import generate_and_store_analytics

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
csv_path = r"D:/Projects/Hotel-Analytics-RAG/dataset/hotel_bookings_dataset.csv"
df = pd.read_csv(csv_path)

# === Precompute and Store All Analytics in DB ===
generate_and_store_analytics(df)

# === Generate or Load Embeddings ===
texts = df.apply(row_to_text, axis=1).tolist()
embeddings = generate_or_load_embeddings(texts)

# === Create or Load FAISS Index ===
faiss_index = create_or_load_faiss_index(embeddings)

# === Load LLaMA Model ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# === Embedder for Query ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Ask Question Function ===
def ask_question(query, max_tokens=200):
    conn = get_db_connection()
    cursor = conn.cursor()
    query_lower = query.lower()

    # === Revenue Query ===
    if "total revenue" in query_lower:
        match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", query_lower)
        if match:
            month = match.group(1).capitalize()
            year = int(match.group(2))
            cursor.execute("SELECT revenue FROM revenue_insights WHERE month = ? AND year = ?", (month, year))
            result = cursor.fetchone()
            answer = f"Total revenue for {month} {year} is {result[0]:.2f}." if result else f"No revenue data found for {month} {year}."
        else:
            answer = "Please specify the month and year (e.g., July 2017) for revenue."

    # === Highest Cancellations by Location ===
    elif "highest booking cancellations" in query_lower:
        cursor.execute("SELECT location, cancellations FROM cancellation_insights ORDER BY cancellations DESC LIMIT 5")
        results = cursor.fetchall()
        if results:
            lines = "\n".join([f"{r[0]}: {r[1]} cancellations" for r in results])
            answer = f"Top locations with highest booking cancellations:\n{lines}"
        else:
            answer = "No cancellation data found."

    # === Average Price ===
    elif "average price" in query_lower or "average adr" in query_lower:
        avg_adr = df['adr'].mean()
        answer = f"The average price of a hotel booking (ADR) is approximately {avg_adr:.2f}."

    # === Cancellations by Date ===
    elif "most cancellations" in query_lower and "date" in query_lower:
        cancelled_df = df[df["is_canceled"] == 1]
        cancelled_by_date = cancelled_df.groupby("reservation_status_date").size().sort_values(ascending=False).head(5)
        answer = "Top dates with most cancellations:\n" + "\n".join([f"{date}: {count}" for date, count in cancelled_by_date.items()])

    # === Specific Cancellations on a Date ===
    elif "cancellations on" in query_lower:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            specific_date = date_match.group(1)
            cursor.execute("SELECT count FROM cancellation_by_date WHERE date = ?", (specific_date,))
            result = cursor.fetchone()
            answer = f"{result[0]} bookings were canceled on {specific_date}." if result else f"No data for cancellations on {specific_date}."
        else:
            answer = "Please specify the date in YYYY-MM-DD format."

    # === Most Bookings by Country ===
    elif "most bookings" in query_lower and "country" in query_lower:
        top_countries = df['country'].value_counts().head(5)
        answer = "Countries with most bookings:\n" + "\n".join([f"{country}: {count}" for country, count in top_countries.items()])

    # === Fallback to FAISS + LLaMA ===
    else:
        query_embedding = embedder.encode([query])
        context = "\n".join(retrieve_faiss(query_embedding, faiss_index, texts))
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        result = llama_pipeline(prompt, max_new_tokens=max_tokens)
        answer = result[0]['generated_text']

    conn.close()
    return answer


if __name__ == "__main__":
    query = "Which locations had the highest booking cancellations?"
    answer = ask_question(query)
    print("\nAnswer:\n", answer)

# Clear GPU memory
torch.cuda.empty_cache()
