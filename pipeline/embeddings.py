import os
import pickle
import numpy as np # Added numpy for saving/loading alternative
from sentence_transformers import SentenceTransformer
import time # For timing

# Assuming default path might be defined elsewhere, e.g., utils.py
# If not, define relative path:
DEFAULT_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
# DEFAULT_EMBEDDINGS_NPY_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npy") # Alternative path for .npy

def generate_or_load_embeddings(
    texts: list,
    cache_path: str = DEFAULT_EMBEDDINGS_PATH, # Use pickle path by default
    # cache_path: str = DEFAULT_EMBEDDINGS_NPY_PATH, # Or use .npy path
    model_name: str = "all-MiniLM-L6-v2",
    force_recompute: bool = False,
    batch_size: int = 128
    ) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a SentenceTransformer model,
    using a cache file (.pkl or .npy) to avoid recomputation.

    Args:
        texts: A list of strings to embed.
        cache_path: Path to the cache file (.pkl or .npy).
        model_name: The name of the SentenceTransformer model to use.
        force_recompute: If True, ignore the cache and regenerate embeddings.
        batch_size: Batch size for encoding.

    Returns:
        A numpy array containing the embeddings.
    """
    cache_exists = os.path.exists(cache_path)
    use_npy = cache_path.endswith(".npy") # Check if using numpy format

    # Load from cache if exists and not forced
    if cache_exists and not force_recompute:
        print(f"Attempting to load embeddings from cache: {cache_path}")
        try:
            start_time = time.time()
            if use_npy:
                with open(cache_path, 'rb') as f:
                    embeddings = np.load(f)
            else: # Assume pickle
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
            load_time = time.time() - start_time
            print(f"Loaded cached embeddings successfully in {load_time:.2f} seconds.")
            # Basic sanity check (optional)
            if len(texts) == embeddings.shape[0]:
                return embeddings
            else:
                print(f"Warning: Cache size ({embeddings.shape[0]}) doesn't match text count ({len(texts)}). Regenerating.")
                # Force recompute if counts don't match
                force_recompute = True
        except (pickle.UnpicklingError, EOFError, ValueError, IsADirectoryError, PermissionError, FileNotFoundError, KeyError) as e:
             print(f"Warning: Failed to load cache file '{cache_path}' (Error: {e}). Regenerating embeddings.")
             force_recompute = True # Force recompute if loading fails
        except Exception as e: # Catch other potential errors during loading
             print(f"Warning: An unexpected error occurred loading cache '{cache_path}' (Error: {e}). Regenerating embeddings.")
             force_recompute = True

    # Generate embeddings if cache doesn't exist, loading failed, or forced
    if not cache_exists or force_recompute:
        if force_recompute and cache_exists:
            print("Forcing recomputation of embeddings.")
        else:
            print("Cache not found or invalid. Generating new embeddings...")

        try:
            start_time = time.time()
            print(f"Loading sentence transformer model: {model_name}")
            # Consider moving model loading outside if called frequently without cache hits
            embedder = SentenceTransformer(model_name)
            print(f"Encoding {len(texts)} texts (batch size: {batch_size})...")
            embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
            gen_time = time.time() - start_time
            print(f"Generated embeddings in {gen_time:.2f} seconds.")

            # Ensure embeddings are float32 for FAISS compatibility later
            embeddings = embeddings.astype('float32')

            # Save to cache
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                print(f"Saving embeddings to cache: {cache_path}")
                if use_npy:
                     with open(cache_path, 'wb') as f:
                        np.save(f, embeddings)
                else: # Assume pickle
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embeddings, f)
                print("Embeddings cached successfully.")
            except (IsADirectoryError, PermissionError, OSError) as e:
                print(f"Error: Could not write cache file to {cache_path}. Check permissions or path. (Error: {e})")
            except Exception as e:
                 print(f"Error: An unexpected error occurred saving cache to {cache_path}. (Error: {e})")


        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Depending on use case, either raise the error or return None/empty array
            # raise e
            return np.array([], dtype='float32') # Return empty array on failure

    return embeddings