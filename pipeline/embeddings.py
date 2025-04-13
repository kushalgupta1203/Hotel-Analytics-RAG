import os
import pickle
from sentence_transformers import SentenceTransformer

def generate_or_load_embeddings(texts, cache_path="D:\Projects\Hotel-Analytics-RAG\pipeline\embeddings.pkl", model_name="all-MiniLM-L6-v2"):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
            print("Loaded cached embeddings.")
            return embeddings

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=128)


    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
        print("Generated and cached embeddings.")
    
    return embeddings
