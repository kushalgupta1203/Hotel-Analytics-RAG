import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pickle
import os

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FAISS index and embeddings
faiss_index_path = "D:/Projects/Hotel-Analytics-RAG/pipeline/hotel_faiss.index"
embeddings_path = "D:/Projects/Hotel-Analytics-RAG/pipeline/embeddings.pkl"

with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)

faiss_index = faiss.read_index(faiss_index_path)

# Load LLaMA model for question answering
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# SentenceTransformer for query embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def retrieve_faiss(query, top_k=5):
    q_emb = embedder.encode([query])
    D, I = faiss_index.search(np.array(q_emb), top_k)
    return [texts[i] for i in I[0]]

def ask_question(query, max_tokens=200):
    context = "\n".join(retrieve_faiss(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = llama_pipeline(prompt, max_new_tokens=max_tokens)
    return result[0]['generated_text']
