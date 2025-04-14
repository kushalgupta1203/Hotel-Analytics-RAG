import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pipeline.utils import row_to_text, get_db_connection, fetch_all_results
from pipeline.embeddings import generate_or_load_embeddings
from pipeline.faiss_index import create_or_load_faiss_index, retrieve_faiss
from pipeline.db_operations import store_revenue_insights, store_cancellation_insights

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
csv_path = r"D:/Projects/Hotel-Analytics-RAG/dataset/hotel_bookings_dataset.csv"
df = pd.read_csv(csv_path)

# === Generate or Load Embeddings ===
texts = df.apply(row_to_text, axis=1).tolist()
embeddings = generate_or_load_embeddings(texts)

# === Create or Load FAISS Index ===
faiss_index = create_or_load_faiss_index(embeddings)

# === Load LLaMA Model ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# === Precompute and Store Insights ===
store_revenue_insights(df)
store_cancellation_insights(df)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Ask Question Functionality ===
def ask_question(query, max_tokens=200):
    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Lowercase the query for easy matching
    query = query.lower()
    
    # Check for revenue-related queries
    if "total revenue" in query and "july" in query and "2017" in query:
        query = "SELECT revenue FROM revenue_insights WHERE month = 'July' AND year = 2017"
        results = fetch_all_results(cursor, query)
        total_revenue = results[0][0] if results else "No data found"
        answer = f"Total revenue for July 2017 is {total_revenue}."
    
    # Check for cancellation-related queries
    elif "highest booking cancellations" in query:
        query = "SELECT location, cancellations FROM cancellation_insights ORDER BY cancellations DESC LIMIT 5"
        results = fetch_all_results(cursor, query)
        locations = "\n".join([f"{r[0]}: {r[1]} cancellations" for r in results])
        answer = f"Locations with the highest booking cancellations are:\n{locations}"
    
    # If query is not recognized, use FAISS and LLaMA model for generic questions
    else:
        query_embedding = embedder.encode([query])
        context = "\n".join(retrieve_faiss(query_embedding, faiss_index, texts))
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        result = llama_pipeline(prompt, max_new_tokens=max_tokens)
        answer = result[0]['generated_text']
    
    # Close the database connection
    conn.close()
    
    return answer

if __name__ == "__main__":
    query = "Which locations had the highest booking cancellations?"
    answer = ask_question(query)
    
    print("\nAnswer:\n", answer)

# Regularly clear GPU cache
torch.cuda.empty_cache()
