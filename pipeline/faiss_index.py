import os
import faiss
import numpy as np
import time

# Assuming default path might be defined elsewhere, e.g., utils.py
# If not, define relative path:
DEFAULT_FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "hotel_faiss.index")

def create_or_load_faiss_index(
    embeddings: np.ndarray,
    index_path: str = DEFAULT_FAISS_INDEX_PATH,
    force_recreate: bool = False
    ):
    """
    Creates or loads a FAISS index for the given embeddings.

    Args:
        embeddings: A float32 numpy array of embeddings.
        index_path: Path to save/load the FAISS index file.
        force_recreate: If True, ignore existing index file and create a new one.

    Returns:
        A trained FAISS index object, or None if creation fails.
    """
    index_exists = os.path.exists(index_path)

    # Load index if exists and not forced
    if index_exists and not force_recreate:
        print(f"Attempting to load FAISS index from: {index_path}")
        try:
            start_time = time.time()
            index = faiss.read_index(index_path)
            load_time = time.time() - start_time
            # Optional: Check if index dimension matches embeddings
            if index.d != embeddings.shape[1]:
                 print(f"Warning: Index dimension ({index.d}) differs from embedding dimension ({embeddings.shape[1]}). Recreating index.")
                 force_recreate = True
            # Optional: Check if number of vectors roughly matches (useful if data changed drastically)
            # elif index.ntotal != embeddings.shape[0]:
            #      print(f"Warning: Index size ({index.ntotal}) differs from embeddings count ({embeddings.shape[0]}). Recreating index.")
            #      force_recreate = True
            else:
                 print(f"Loaded FAISS index successfully in {load_time:.2f} seconds.")
                 return index
        except (RuntimeError, IsADirectoryError, PermissionError, FileNotFoundError) as e:
            print(f"Warning: Failed to load FAISS index '{index_path}' (Error: {e}). Recreating index.")
            force_recreate = True
        except Exception as e:
            print(f"Warning: An unexpected error occurred loading FAISS index '{index_path}' (Error: {e}). Recreating index.")
            force_recreate = True

    # Create index if not loaded or forced
    if not index_exists or force_recreate:
        if force_recreate and index_exists:
             print("Forcing recreation of FAISS index.")
        else:
             print("FAISS index not found or invalid. Creating new index...")

        if embeddings is None or embeddings.shape[0] == 0:
             print("Error: Cannot create FAISS index with empty or None embeddings.")
             return None

        try:
            start_time = time.time()
            dimension = embeddings.shape[1]
            # Ensure embeddings are float32 numpy array
            embeddings_np = np.asarray(embeddings, dtype='float32')

            # Using IndexFlatL2 - suitable for exact search on moderate datasets
            # For very large datasets, consider alternatives like IndexIVFFlat after training
            print(f"Creating FAISS IndexFlatL2 with dimension {dimension}...")
            index = faiss.IndexFlatL2(dimension)

            print(f"Adding {embeddings_np.shape[0]} vectors to the index...")
            index.add(embeddings_np)
            build_time = time.time() - start_time
            print(f"FAISS index created and populated in {build_time:.2f} seconds. Index size: {index.ntotal}")

            # Save the index
            try:
                 # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                print(f"Saving FAISS index to: {index_path}")
                faiss.write_index(index, index_path)
                print("FAISS index saved successfully.")
            except (IsADirectoryError, PermissionError, OSError) as e:
                print(f"Error: Could not write FAISS index file to {index_path}. Check permissions or path. (Error: {e})")
            except Exception as e:
                 print(f"Error: An unexpected error occurred saving FAISS index to {index_path}. (Error: {e})")


            return index

        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            import traceback
            traceback.print_exc()
            # raise e # Or return None
            return None

# Edited retrieve_faiss to return indices
def retrieve_faiss(
    query_embedding: np.ndarray,
    faiss_index, # faiss.Index object
    top_k: int = 5
    ) -> list: # Returns list of indices
    """
    Searches the FAISS index for the top_k nearest neighbors to the query embedding.

    Args:
        query_embedding: A numpy array representing the query embedding(s).
                         Should be shape (n_queries, dimension).
        faiss_index: The loaded FAISS index object.
        top_k: The number of nearest neighbors to retrieve.

    Returns:
        A list containing the indices of the top_k nearest neighbors
        for the first query (or None if search fails).
    """
    if faiss_index is None:
        print("Error: Cannot search with None FAISS index.")
        return []
    if query_embedding is None or query_embedding.size == 0 :
         print("Error: Cannot search with None or empty query embedding.")
         return []

    try:
        # Ensure query is float32 numpy array, handle single vs multiple queries
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0) # Add batch dim
        query_embedding_np = np.asarray(query_embedding, dtype='float32')

        # Perform the search
        distances, indices = faiss_index.search(query_embedding_np, top_k)

        # Return the indices for the first (or only) query
        # Convert indices to standard Python int list
        return [int(i) for i in indices[0]]

    except Exception as e:
        print(f"Error during FAISS search: {e}")
        import traceback
        traceback.print_exc()
        return [] # Return empty list on error