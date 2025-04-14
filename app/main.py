import sys
import os
import pandas as pd
import torch
import time
from contextlib import asynccontextmanager
import sqlite3 # Import sqlite3 for error handling

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# --- Path Setup ---
# (Keep your existing path setup)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
pipeline_dir = os.path.join(project_root, 'pipeline')
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)
if project_root not in sys.path:
     sys.path.insert(0, project_root)

# --- Import Custom Modules AFTER path setup ---
# (Keep your existing imports)
try:
    from pipeline.utils import (
        get_db_connection, fetch_all_results, fetch_one_result, row_to_text, normalize_country_code,
        DEFAULT_DB_PATH, DEFAULT_COUNTRY_MAPPING_PATH
    )
    from pipeline.main import ask_question
    from pipeline.db_operations import generate_and_store_analytics
    from pipeline.embeddings import generate_or_load_embeddings
    from pipeline.faiss_index import create_or_load_faiss_index
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print(f"Please ensure pipeline directory is correctly structured and accessible from: {current_dir}")
    sys.exit(1)


# --- Configuration ---
# (Keep your existing configuration)
DB_PATH = DEFAULT_DB_PATH
CSV_PATH = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv" # Or uncomment this if relative path fails
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"

FORCE_RECOMPUTE_ANALYTICS = os.getenv("FORCE_RECOMPUTE_ANALYTICS", "false").lower() == "true"
FORCE_RECOMPUTE_EMBEDDINGS = os.getenv("FORCE_RECOMPUTE_EMBEDDINGS", "false").lower() == "true"
FORCE_RECREATE_INDEX = os.getenv("FORCE_RECREATE_INDEX", "false").lower() == "true"

# --- Lifespan Context Manager (EDITED DATA LOADING) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Load models, data, precompute analytics
    print("--- Starting Application Lifespan ---")
    start_time = time.time()

    # 1. Load Dataset and Clean Problematic Columns
    print(f"Loading dataset from {CSV_PATH}...")
    try:
        # Specify low_memory=False if you have mixed types, might help sometimes
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"Dataset loaded initially: {len(df)} rows.")

        # --- Revised Cleaning for 'children', 'agent', 'company' ---
        cols_to_clean_as_int = ['children', 'agent', 'company']
        for col in cols_to_clean_as_int:
            if col in df.columns:
                print(f"Cleaning column: {col}")
                # Step 1: Convert to numeric. Errors (like 'unknown') become NaN.
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Step 2: Fill ALL NaN values (original NaNs AND coerced errors) with 0.
                df[col].fillna(0, inplace=True)
                # Step 3: Now safely convert the column to integer type.
                df[col] = df[col].astype(int)
                print(f"Column '{col}' cleaned and converted to int.")
            else:
                print(f"Warning: Column '{col}' not found in dataset during cleaning.")

        # Handle other NaNs that should be strings or specific values
        df.fillna({'country': 'Unknown'}, inplace=True) # Handle country NaN separately
        # Add other specific fillna calls if needed for other columns (e.g., meal)
        df.fillna({'meal': 'Undefined/SC'}, inplace=True) # Example for meal

        print(f"Dataset cleaned. Final rows: {len(df)} rows.")

    except FileNotFoundError:
        print(f"FATAL Error: Dataset file not found at {CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL Error loading or cleaning dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Calculate composite columns if needed (AFTER cleaning)
    print("Calculating composite columns (total_nights, revenue)...")
    # Total Nights
    if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        # Ensure component columns are numeric and fill NaNs before summing
        weekend_nights = pd.to_numeric(df['stays_in_weekend_nights'], errors='coerce').fillna(0)
        week_nights = pd.to_numeric(df['stays_in_week_nights'], errors='coerce').fillna(0)
        df['total_nights'] = (weekend_nights + week_nights).astype(int)
        print("'total_nights' calculated.")
    else:
        print("Warning: Cannot calculate 'total_nights', required columns missing.")
        # Handle case where total_nights might be needed later - maybe add a default column?
        if 'total_nights' not in df.columns: df['total_nights'] = 0


    # Revenue
    if 'adr' in df.columns and 'total_nights' in df.columns:
         # Ensure adr is numeric before multiplication, handle potential NaNs
        df['adr'] = pd.to_numeric(df['adr'], errors='coerce').fillna(0.0)
        # Ensure total_nights exists and is numeric (should be from above)
        df['revenue'] = df['adr'] * df['total_nights']
        print("'revenue' calculated.")
    else:
        print("Warning: Cannot calculate 'revenue', required columns 'adr' or 'total_nights' missing/not calculable.")
        if 'revenue' not in df.columns: df['revenue'] = 0.0 # Add default if missing

    # --- Continue with rest of the lifespan setup ---

    # 2. Precompute and Store Analytics
    print("Checking analytics precomputation...")
    # (Rest of your existing lifespan code: generate_and_store_analytics call, text generation, embedding, FAISS, model loading)
    # Make sure generate_and_store_analytics uses the cleaned df
    if FORCE_RECOMPUTE_ANALYTICS or not os.path.exists(DB_PATH):
        print("Generating and storing analytics in DB...")
        try:
            generate_and_store_analytics(df.copy(), DB_PATH) # Use the cleaned df
        except Exception as e:
            print(f"Error during analytics generation: {e}")
            # sys.exit(1) # Decide if critical
    else:
        print("Analytics DB exists. Skipping generation.")

    # 3. Generate Text Representations
    print("Generating text representations for RAG...")
    try:
        texts = df.apply(row_to_text, axis=1).tolist()
        app.state.texts = texts # Store in app.state
        print(f"Generated {len(texts)} text representations.")
    except Exception as e:
        print(f"FATAL error generating texts: {e}")
        sys.exit(1)


    # 4. Generate or Load Embeddings
    print("Generating/Loading embeddings...")
    try:
        embeddings = generate_or_load_embeddings(texts, force_recompute=FORCE_RECOMPUTE_EMBEDDINGS)
        print(f"Embeddings ready. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"FATAL Error loading/generating embeddings: {e}")
        sys.exit(1)

    # 5. Create or Load FAISS Index
    print("Creating/Loading FAISS index...")
    try:
        faiss_index = create_or_load_faiss_index(embeddings, force_recreate=FORCE_RECREATE_INDEX)
        app.state.faiss_index = faiss_index # Store in app.state
        print(f"FAISS index ready. Index size: {faiss_index.ntotal}")
        del embeddings # Free memory
    except Exception as e:
        print(f"FATAL Error loading/creating FAISS index: {e}")
        sys.exit(1)

    # 6. Load Models
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        app.state.embedder = embedder # Store in app.state
        print("Embedder loaded successfully.")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully.")

        llama_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype=torch.float16)
        print("Llama model loaded successfully.")

        llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer, device_map="auto")
        app.state.llama_pipeline = llama_pipeline # Store in app.state
        print("Llama pipeline created and stored in app.state.")
        app.state.db_path = DB_PATH # Store db path

    except Exception as e:
        print(f"FATAL Error loading models: {e}")
        sys.exit(1)


    end_time = time.time()
    print(f"--- Application Startup Complete ({end_time - start_time:.2f} seconds) ---")

    yield # Application runs here

    # Shutdown - Clean up resources
    print("--- Starting Application Shutdown ---")
    app.state.clear()
    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
    print("--- Application Shutdown Complete ---")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# --- Pydantic Models ---
# (Keep existing Query model)
class Query(BaseModel):
    question: str

# --- API Endpoints ---
# (Keep existing endpoints: /, /ask, /analytics)
# Make sure they correctly use app_state or request.app.state to access resources

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_api(q: Query, request: Request):
    """Handles user questions, uses precomputed insights or RAG."""
    # Use request.app.state if accessing within endpoint
    state = request.app.state
    print(f"App State in /ask: {state._state.keys()}") # CORRECTED PRINT STATEMENT
    if not state or not hasattr(state, "llama_pipeline") or state.llama_pipeline is None: # Check if attribute exists
         raise HTTPException(status_code=503, detail="Resources not initialized yet. Please wait and retry.")

    question = q.question
    print(f"Received question: {question}")

    try:
        # Call the refactored ask_question function with resources from app_state
        answer = ask_question(
            query=question,
            db_path=state.db_path,
            texts=state.texts,
            embedder=state.embedder,
            faiss_index=state.faiss_index,
            llama_pipeline=state.llama_pipeline
            # Add other params like max_tokens if needed
        )
        return {"answer": answer}
    except KeyError as e:
        print(f"Error: Missing resource in app_state: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Missing resource {e}")
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


@app.post("/analytics")
async def get_analytics_api(request: Request):
    """Provides pre-calculated analytics data in JSON format."""
    state = request.app.state
    print(f"App State in /analytics: {state._state.keys()}") # CORRECTED PRINT STATEMENT
    if not state or not hasattr(state, "db_path") or state.db_path is None:
         raise HTTPException(status_code=503, detail="Resources not initialized yet or DB path missing.")

    print("Fetching analytics data...")
    db_path = state.db_path

    conn = None # Initialize conn to None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        # Fetch data from precomputed tables
        revenue_trends = fetch_all_results(cursor, "SELECT month, year, revenue FROM revenue_insights ORDER BY year, printf('%02d', CASE month WHEN 'January' THEN 1 WHEN 'February' THEN 2 WHEN 'March' THEN 3 WHEN 'April' THEN 4 WHEN 'May' THEN 5 WHEN 'June' THEN 6 WHEN 'July' THEN 7 WHEN 'August' THEN 8 WHEN 'September' THEN 9 WHEN 'October' THEN 10 WHEN 'November' THEN 11 WHEN 'December' THEN 12 ELSE 0 END)")
        top_cancellations = fetch_all_results(cursor, "SELECT location, cancellations FROM cancellation_insights ORDER BY cancellations DESC LIMIT 10")
        avg_adr = fetch_one_result(cursor, "SELECT average_adr FROM average_adr_insight LIMIT 1")
        top_countries = fetch_all_results(cursor, "SELECT country, count FROM top_booking_countries ORDER BY count DESC LIMIT 10")
        top_cancel_dates = fetch_all_results(cursor, "SELECT date, count FROM cancellation_by_date ORDER BY count DESC LIMIT 10")

        # Format results
        analytics_data = {
            "revenue_trends": [{"month": m, "year": y, "revenue": f"{r:,.2f}"} for m, y, r in revenue_trends],
            "top_cancellation_locations": [{"location": loc, "cancellations": c} for loc, c in top_cancellations],
            "average_adr": f"{avg_adr[0]:.2f}" if avg_adr else "N/A",
            "top_booking_countries": [{"country": c, "count": cnt} for c, cnt in top_countries],
            "top_cancellation_dates": [{"date": d, "count": cnt} for d, cnt in top_cancel_dates]
        }

        return analytics_data

    except sqlite3.Error as e:
        print(f"Database error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {e}")
    finally:
        if conn:
            conn.close()

# --- Optional: Add run command for development ---
# (Keep existing __main__ block)
if __name__ == "__main__":
    print("Starting Uvicorn server for development...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)