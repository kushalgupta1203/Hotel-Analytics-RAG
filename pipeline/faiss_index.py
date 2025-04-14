import os
import faiss
import numpy as np

def create_or_load_faiss_index(embeddings, index_path="D:/Projects/Hotel-Analytics-RAG/pipeline/hotel_faiss.index"):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("Loaded FAISS index from cache.")
        return index

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    faiss.write_index(index, index_path)
    print("Created and saved FAISS index.")
    
    return index

def retrieve_faiss(query_embedding, faiss_index, texts, top_k=5):
    D, I = faiss_index.search(np.array(query_embedding), top_k)
    return [texts[i] for i in I[0]]
