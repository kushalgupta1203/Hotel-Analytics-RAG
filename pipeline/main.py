import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pipeline.utils import row_to_text
from pipeline.embeddings import generate_or_load_embeddings
from pipeline.faiss_index import create_or_load_faiss_index, retrieve_faiss
from sentence_transformers import SentenceTransformer

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
csv_path = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv"
df = pd.read_csv(csv_path)

# === Prepare Text Data ===
texts = df.apply(row_to_text, axis=1).tolist()

# === Generate or Load Embeddings ===
embeddings = generate_or_load_embeddings(texts)

# === Create or Load FAISS Index ===
faiss_index = create_or_load_faiss_index(embeddings)

# === Load LLaMA Model ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
# === Ask Question Functionality ===
def ask_question(query, max_tokens=200):
    query_embedding = embedder.encode([query])
    
    context = "\n".join(retrieve_faiss(query_embedding, faiss_index, texts))
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    result = llama_pipeline(prompt, max_new_tokens=max_tokens)
    
    return result[0]['generated_text']

if __name__ == "__main__":
    query = "Which months have the highest cancellations?"
    answer = ask_question(query)
    
    print("\nAnswer:\n", answer)

# Regularly clear GPU cache
torch.cuda.empty_cache()