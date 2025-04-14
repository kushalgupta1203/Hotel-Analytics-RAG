import sys
import os
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 # Added for error handling in ask_question
import traceback # Added for error handling

# --- Configuration ---
# Determine project root dynamically
# Assuming this script is in the project root alongside 'pipeline' and 'dataset' folders
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # Adjust if script is elsewhere

# Add pipeline directory to sys.path
pipeline_dir = os.path.join(project_root, 'pipeline')
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir) # Use insert(0,...)

# Import custom modules AFTER path setup
try:
    from utils import (
        row_to_text, get_db_connection, fetch_all_results, fetch_one_result,
        DEFAULT_DB_PATH, normalize_country_code
    )
    # Assuming these modules exist in the 'pipeline' directory or are installable
    from embeddings import generate_or_load_embeddings # Assuming this handles its own paths/logic
    from faiss_index import create_or_load_faiss_index, retrieve_faiss # Assuming these handle paths/logic
    from db_operations import generate_and_store_analytics
except ImportError as e:
    print(f"Error importing custom modules (utils/embeddings/faiss_index/db_operations): {e}")
    print(f"Please ensure pipeline directory is correctly structured ({pipeline_dir}) and accessible.")
    sys.exit(1)


# Define paths (consider using a config file or args for more robustness)
# Using DEFAULT_DB_PATH from utils
DB_PATH = DEFAULT_DB_PATH
CSV_PATH = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf" # Make sure you have access/downloaded this

# --- Initialization Function (EDITED DATA LOADING) ---
def initialize_system(csv_path, db_path, embedding_model_name, llm_model_name, force_recompute_analytics=False, force_recompute_embeddings=False, force_recreate_index=False):
    """Loads data, computes analytics/embeddings/index, and loads models."""
    print("Initializing system...")
    df = None # Initialize df to None

    # === Load Dataset and Clean ===
    print(f"Loading dataset from {csv_path}...")
    try:
        # Specify low_memory=False if dealing with mixed types
        df = pd.read_csv(csv_path, low_memory=False)
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

        # Handle other specific NaNs that should be strings or specific values
        df.fillna({'country': 'Unknown'}, inplace=True) # Handle country NaN separately
        df.fillna({'meal': 'Undefined/SC'}, inplace=True) # Example for meal

        print(f"Dataset cleaned. Final rows: {len(df)} rows.")

    except FileNotFoundError:
        print(f"FATAL Error: Dataset file not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL Error loading or cleaning dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Store DataFrame in SQLite (for SQL-based analytics) ===
    print(f"Storing the cleaned DataFrame into SQLite table 'hotel_bookings' at {db_path}...")
    try:
        conn = get_db_connection(db_path)
        df.to_sql('hotel_bookings', conn, if_exists='replace', index=False) # 'replace' will drop and recreate the table
        conn.close()
        print("DataFrame stored in SQLite successfully.")
    except Exception as e:
        print(f"FATAL Error storing DataFrame to SQLite: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Calculate composite columns (AFTER cleaning) ===
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
        if 'total_nights' not in df.columns: df['total_nights'] = 0 # Add default

    # Revenue
    if 'adr' in df.columns and 'total_nights' in df.columns:
        # Ensure adr is numeric before multiplication, handle potential NaNs
        df['adr'] = pd.to_numeric(df['adr'], errors='coerce').fillna(0.0)
        # Ensure total_nights exists and is numeric (should be from above)
        df['revenue'] = df['adr'] * df['total_nights']
        print("'revenue' calculated.")
    else:
        print("Warning: Cannot calculate 'revenue', required columns 'adr' or 'total_nights' missing/not calculable.")
        if 'revenue' not in df.columns: df['revenue'] = 0.0 # Add default

    # === Precompute and Store Analytics ===
    print("Checking analytics precomputation...")
    if force_recompute_analytics or not os.path.exists(db_path):
        print("Generating and storing analytics in DB...")
        try:
            generate_and_store_analytics(df.copy(), db_path) # Use the cleaned df
        except Exception as e:
            print(f"Error during analytics generation: {e}")
            # Decide if this is critical or app can continue without updated analytics
            # sys.exit(1)
    else:
        print("Analytics DB exists. Skipping generation (use flag to force recompute).")


    # === Generate Text Representations ===
    print("Generating text representations for RAG...")
    start_time = time.time()
    try:
        # Ensure all required columns for row_to_text exist after potential calculations
        texts = df.apply(row_to_text, axis=1).tolist()
        print(f"Text generation took {time.time() - start_time:.2f} seconds. ({len(texts)} texts generated)")
    except Exception as e:
        print(f"FATAL error generating texts: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Generate or Load Embeddings ===
    print("Generating/Loading embeddings...")
    start_time = time.time()
    try:
        # Pass force flag to embedding generation
        embeddings = generate_or_load_embeddings(texts, force_recompute=force_recompute_embeddings)
        print(f"Embeddings ready. Took {time.time() - start_time:.2f} seconds. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"FATAL Error loading/generating embeddings: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Create or Load FAISS Index ===
    print("Creating/Loading FAISS index...")
    start_time = time.time()
    try:
        # Pass force flag to index creation
        faiss_index = create_or_load_faiss_index(embeddings, force_recreate=force_recreate_index)
        print(f"FAISS index ready. Took {time.time() - start_time:.2f} seconds. Index size: {faiss_index.ntotal}")
        del embeddings # Free memory
    except Exception as e:
        print(f"FATAL Error loading/creating FAISS index: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Load Models ===
    print("Loading models...")
    start_time = time.time()
    # Ensure GPU is used if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    embedder = None
    llama_pipeline = None
    llama_model = None # Define llama_model here to handle potential deletion later
    try:
        # Embedder for Query
        embedder = SentenceTransformer(embedding_model_name, device=device)

        # LLaMA Model & Pipeline
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llama_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", torch_dtype=torch.float16) # Use float16 for less memory
        llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer, device_map="auto") # Let pipeline handle device mapping

    except Exception as e:
        print(f"FATAL Error loading LLaMA model ({llm_model_name}): {e}")
        print("Make sure you have the necessary libraries (transformers, accelerate) and model weights.")
        traceback.print_exc()
        sys.exit(1)

    print(f"Models loaded. Took {time.time() - start_time:.2f} seconds.")
    print("Initialization complete.")

    # Return all necessary components including the model for potential cleanup
    return df, db_path, texts, embedder, faiss_index, llama_pipeline, llama_model


# === Graphical Analytics Function ===
# (Keep the get_analytics function exactly as provided in the previous response)
def get_analytics(df: pd.DataFrame):
    """
    Generates and displays several simple graphical analytics based on the DataFrame.
    """
    print("\n--- Generating Graphical Analytics ---")

    # Set style for plots
    sns.set_theme(style="whitegrid")

    # --- Figure 1: Bookings Overview ---
    plt.figure(figsize=(14, 6))

    # Plot 1.1: Bookings by Hotel Type
    plt.subplot(1, 2, 1) # 1 row, 2 cols, plot 1
    ax = sns.countplot(x='hotel', data=df, palette='viridis')
    plt.title('Booking Counts by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('Number of Bookings')
    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container)

    # Plot 1.2: Bookings Status (Cancelled vs. Not Cancelled)
    plt.subplot(1, 2, 2) # 1 row, 2 cols, plot 2
    ax = sns.countplot(x='is_canceled', data=df, palette='viridis')
    plt.title('Booking Status (Cancelled vs. Not Cancelled)')
    plt.xlabel('Booking Status (1=Cancelled, 0=Not Cancelled)')
    plt.ylabel('Number of Bookings')
    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show(block=False) # Use block=False if running in some environments


    # --- Figure 2: Bookings Over Time (Monthly) ---
    if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns:
        plt.figure(figsize=(14, 7))
        # Ensure 'arrival_date_month' is categorical for correct sorting
        month_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
        # Use try-except for categorical conversion as it might fail if column has unexpected values
        try:
            df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)

            # Group by month and count bookings - handle potential NaNs from categorical conversion
            monthly_bookings = df.dropna(subset=['arrival_date_month']).groupby('arrival_date_month').size().reset_index(name='count')

            sns.lineplot(x='arrival_date_month', y='count', data=monthly_bookings, sort=False, marker='o')
            plt.title('Total Bookings per Month (Across All Years)')
            plt.xlabel('Month')
            plt.ylabel('Number of Bookings')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
            print(f"Skipping 'Bookings Over Time' plot due to error: {e}")

    else:
        print("Skipping 'Bookings Over Time' plot: Required date columns missing.")


    # --- Figure 3: Cancellation Rate by Hotel Type ---
    if 'hotel' in df.columns and 'is_canceled' in df.columns:
        plt.figure(figsize=(8, 6))
        # Calculate cancellation rate per hotel type
        # Ensure 'is_canceled' is numeric
        df['is_canceled_numeric'] = pd.to_numeric(df['is_canceled'], errors='coerce')
        cancellation_rates = df.dropna(subset=['hotel', 'is_canceled_numeric']).groupby('hotel')['is_canceled_numeric'].mean().reset_index()
        cancellation_rates.rename(columns={'is_canceled_numeric': 'cancellation_rate'}, inplace=True)
        cancellation_rates['cancellation_rate'] = cancellation_rates['cancellation_rate'] * 100 # Convert to percentage

        ax = sns.barplot(x='hotel', y='cancellation_rate', data=cancellation_rates, palette='viridis')
        plt.title('Cancellation Rate (%) by Hotel Type')
        plt.xlabel('Hotel Type')
        plt.ylabel('Cancellation Rate (%)')
        # Add percentage labels
        for index, row in cancellation_rates.iterrows():
            ax.text(index, row.cancellation_rate + 0.5, f'{row.cancellation_rate:.2f}%', color='black', ha="center") # Adjust position slightly
        plt.ylim(0, cancellation_rates['cancellation_rate'].max() * 1.1) # Adjust y-limit for labels
        plt.tight_layout()
        plt.show(block=False)
        df.drop(columns=['is_canceled_numeric'], inplace=True) # Clean up temp column
    else:
        print("Skipping 'Cancellation Rate' plot: Required columns missing.")


    # --- Figure 4: Top 10 Booking Countries ---
    if 'country' in df.columns:
        plt.figure(figsize=(12, 7))
        try:
            # Normalize country codes first
            df['country_normalized'] = df['country'].apply(normalize_country_code)
            # Exclude 'Unknown' after normalization
            top_countries = df[df['country_normalized'].notna() & (df['country_normalized'] != 'Unknown')]['country_normalized'].value_counts().nlargest(10)

            if not top_countries.empty:
                sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis', orient='h')
                plt.title('Top 10 Countries by Number of Bookings')
                plt.xlabel('Number of Bookings')
                plt.ylabel('Country')
                plt.tight_layout()
                plt.show(block=False)
            else:
                print("Skipping 'Top Countries' plot: No valid country data found after normalization.")
            # Don't drop country_normalized if db_operations needs it, maybe drop later
            # df.drop(columns=['country_normalized'], inplace=True)
        except Exception as e:
            print(f"Skipping 'Top Countries' plot due to error: {e}")
    else:
        print("Skipping 'Top Countries' plot: 'country' column missing.")


    # --- Figure 5: Market Segment Distribution ---
    if 'market_segment' in df.columns:
        plt.figure(figsize=(10, 7))
        segment_counts = df['market_segment'].value_counts()
        sns.barplot(x=segment_counts.values, y=segment_counts.index, palette='viridis', orient='h')
        plt.title('Distribution of Bookings by Market Segment')
        plt.xlabel('Number of Bookings')
        plt.ylabel('Market Segment')
        plt.tight_layout()
        plt.show(block=False)
    else:
        print("Skipping 'Market Segment' plot: 'market_segment' column missing.")


    # --- Figure 6: ADR Distribution ---
    if 'adr' in df.columns:
        plt.figure(figsize=(10, 6))
        # Ensure adr is numeric
        adr_numeric = pd.to_numeric(df['adr'], errors='coerce')
        # Filter out extreme ADR values for better visualization
        reasonable_adr = adr_numeric[(adr_numeric > 0) & (adr_numeric < 500)].dropna()
        if not reasonable_adr.empty:
            sns.histplot(reasonable_adr, kde=True, bins=50)
            plt.title('Distribution of Average Daily Rate (ADR > 0 & < 500)')
            plt.xlabel('ADR')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show(block=False) # Use block=False
        else:
            print("Skipping 'ADR Distribution' plot: No valid ADR data in range (0, 500).")
    else:
        print("Skipping 'ADR Distribution' plot: 'adr' column missing.")

    # Ensure all plots are displayed if block=False was used
    plt.show() # Add a final plt.show() to display all non-blocking plots
    print("--- Finished Generating Analytics ---")


# === Ask Question Function ===
# (Keep the ask_question function exactly as provided in the previous response)
def ask_question(
    query: str,
    db_path: str,
    texts: list,
    embedder: SentenceTransformer,
    faiss_index, # Add type hint if possible (e.g., faiss.Index)
    llama_pipeline, # Add type hint if possible (e.g., transformers.Pipeline)
    max_context_docs: int = 5,
    max_tokens: int = 250 # Increased default max tokens
) -> str:
    """
    Answers a question using precomputed analytics or RAG fallback.
    """
    conn = None # Initialize conn
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        query_lower = query.lower()
        answer = "Sorry, I couldn't find an answer to that specific question based on the available data or query patterns." # Default answer


        # === Precomputed Query Handlers ===

        # Total Revenue for Month/Year
        if "total revenue" in query_lower:
            # Relaxed regex to capture month variations
            match = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})", query_lower)
            if match:
                month_abbr = match.group(1).capitalize()[:3] # Get 3-letter abbreviation
                # Map common abbreviations to full names used in dataset
                month_map = { "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
                              "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
                              "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
                month_full = month_map.get(month_abbr, month_abbr) # Fallback if needed
                year = int(match.group(2))

                sql = "SELECT revenue FROM revenue_insights WHERE month = ? AND year = ?"
                result = fetch_one_result(cursor, sql, (month_full, year))

                answer = f"Total revenue for {month_full} {year} was {result[0]:,.2f}." if result else f"No revenue data found for {month_full} {year}."
            else:
                answer = "For revenue, please specify the full month and year (e.g., 'total revenue for July 2017')."

        # Highest Cancellations by Location (Top 5)
        elif "highest booking cancellations" in query_lower and ("location" in query_lower or "country" in query_lower):
            # Query the precomputed, sorted table
            sql = "SELECT location, cancellations FROM cancellation_insights ORDER BY cancellations DESC LIMIT 5"
            results = fetch_all_results(cursor, sql)
            if results:
                lines = "\n".join([f"- {r[0]}: {r[1]} cancellations" for r in results])
                answer = f"Top 5 locations with highest booking cancellations:\n{lines}"
            else:
                answer = "No cancellation data by location found."

        # Average Price (ADR)
        elif "average price" in query_lower or "average adr" in query_lower:
            # Query the precomputed average
            sql = "SELECT average_adr FROM average_adr_insight LIMIT 1"
            result = fetch_one_result(cursor, sql)
            answer = f"The overall average daily rate (ADR) across all bookings is approximately {result[0]:.2f}." if result else "Average ADR data not found."

        # Top Dates with Most Cancellations (Top 5)
        elif "most cancellations" in query_lower and ("date" in query_lower or "when" in query_lower):
            # Query the precomputed, sorted table
            sql = "SELECT date, count FROM cancellation_by_date ORDER BY count DESC LIMIT 5"
            results = fetch_all_results(cursor, sql)
            if results:
                lines = "\n".join([f"- {r[0]}: {r[1]} cancellations" for r in results])
                answer = f"Top 5 dates with the most cancellations:\n{lines}"
            else:
                answer = "No cancellation data by date found."

        # Specific Cancellations on a Date
        elif "cancellations on" in query_lower:
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query) # Assumes YYYY-MM-DD format
            if date_match:
                specific_date = date_match.group(1)
                # Query the precomputed table
                sql = "SELECT count FROM cancellation_by_date WHERE date = ?"
                result = fetch_one_result(cursor, sql, (specific_date,))
                answer = f"{result[0]} bookings were canceled on {specific_date}." if result else f"No cancellation data found for the specific date {specific_date}."
            else:
                answer = "Please specify the date for cancellations in YYYY-MM-DD format."

        # Most Bookings by Country (Top 5)
        elif "most bookings" in query_lower and "country" in query_lower:
            # Query the precomputed table
            sql = "SELECT country, count FROM top_booking_countries ORDER BY count DESC LIMIT 5"
            results = fetch_all_results(cursor, sql)
            if results:
                lines = "\n".join([f"- {r[0]}: {r[1]} bookings" for r in results])
                answer = f"Top 5 countries with the most bookings:\n{lines}"
            else:
                answer = "Could not retrieve top booking countries."

        # === Fallback to FAISS + LLaMA (RAG) ===
        else:
            print("Query not matched by specific handlers. Using RAG fallback...")
            start_time = time.time()
            query_embedding = embedder.encode([query])

            # Retrieve from FAISS
            # Make sure retrieve_faiss is correctly implemented to return indices
            retrieved_indices = retrieve_faiss(query_embedding, faiss_index, k=max_context_docs)
            # Ensure indices are valid and handle potential nested list structure from FAISS
            if isinstance(retrieved_indices, tuple): # faiss often returns (distances, indices)
                context_indices = retrieved_indices[1][0] # Get indices from the first query result
            elif isinstance(retrieved_indices, list) and len(retrieved_indices) > 0 and isinstance(retrieved_indices[0], list):
                context_indices = retrieved_indices[0] # Handle case where it might return list of lists
            else:
                context_indices = retrieved_indices # Assume it's already a list of indices


            valid_indices = [int(idx) for idx in context_indices if 0 <= int(idx) < len(texts)] # Convert to int
            context = "\n---\n".join([texts[i] for i in valid_indices]) # Join relevant text snippets

            retrieval_time = time.time() - start_time
            print(f"FAISS retrieval took {retrieval_time:.2f} seconds. Found {len(valid_indices)} valid contexts.")

            if not context:
                answer = "I couldn't find relevant information in the dataset to answer that question."
            else:
                # Prepare prompt for LLaMA
                prompt = f"""Use the following context extracted from hotel booking data to answer the question. Provide a concise answer based *only* on the context given. Do not add information not present in the context. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""

                # Generate answer using LLaMA pipeline
                print("Generating answer with LLM...")
                start_time = time.time()
                llm_result = llama_pipeline(prompt, max_new_tokens=max_tokens, num_return_sequences=1, do_sample=False)
                llm_time = time.time() - start_time
                print(f"LLM generation took {llm_time:.2f} seconds.")

                # Extract the generated text, cleaning up the prompt repetition
                generated_text = llm_result[0]['generated_text']
                # Find the start of the actual answer after the prompt template
                answer_start_index = generated_text.rfind("Answer:") # Use rfind for safety
                if answer_start_index != -1:
                    answer = generated_text[answer_start_index + len("Answer:"):].strip()
                else:
                    # Fallback if 'Answer:' marker isn't found
                    answer = generated_text.replace(prompt, "").strip()

                if not answer: # Handle cases where the LLM might return empty string
                    answer = "The LLaMA model could not generate an answer based on the provided context."

    except sqlite3.Error as e:
        answer = f"A database error occurred: {e}"
        traceback.print_exc() # Print traceback for debugging
    except Exception as e:
        answer = f"An unexpected error occurred during question processing: {e}"
        traceback.print_exc() # Print full traceback for debugging
    finally:
        if conn:
            conn.close()

    return answer


# === Main Execution ===
if __name__ == "__main__":
    # --- Initialize ---
    # Set flags for recomputing (optional)
    force_recompute_analytics = False # Set to True to force recalculation of DB analytics
    force_recompute_embeddings = False # Set to True to force regeneration of embeddings
    force_recreate_index = False     # Set to True to force recreation of FAISS index
    llama_model_ref = None # Variable to hold model reference for cleanup

    # Make sure initialize_system returns 'df' as the first argument, and model ref
    try:
        df, db_path, texts, embedder, faiss_index, llama_pipeline, llama_model_ref = initialize_system(
            CSV_PATH,
            DB_PATH,
            EMBEDDING_MODEL,
            LLM_MODEL,
            force_recompute_analytics,
            force_recompute_embeddings,
            force_recreate_index
        )
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        traceback.print_exc()
        sys.exit(1) # Exit if initialization fails

    # --- Generate and Display Graphical Analytics ---
    # Check if df was loaded successfully before calling analytics
    if df is not None and not df.empty:
        # Call the analytics function with a copy of the DataFrame
        try:
            get_analytics(df.copy())
        except Exception as e:
            print(f"Error during graphical analytics generation: {e}")
            traceback.print_exc()
    else:
        print("Warning: DataFrame is None or empty. Skipping graphical analytics.")


    # --- Ask Questions ---
    questions = [
        "What was the total revenue for July 2017?",
        "Which country had the highest booking cancellations?", # Should hit precomputed
        "What is the average ADR?", # Should hit precomputed
        "Which date had the most cancellations?", # Should hit precomputed
        "How many cancellations were on 2015-07-06?", # Should hit precomputed
        "Which country made the most bookings?", # Should hit precomputed
        "Tell me about bookings from Portugal (PRT).", # Should trigger RAG
        "Were there any bookings with babies from Great Britain?", # Should trigger RAG
        "What are the different meal types offered?", # Should trigger RAG
        "Invalid query check?" # Should trigger RAG or default message
    ]

    print("\n--- Starting Question Answering ---") # Header for clarity
    if llama_pipeline: # Check if pipeline loaded successfully
        for q in questions:
            print(f"\n❓ Question: {q}")
            start_time = time.time()
            try:
                response = ask_question(
                    query=q,
                    db_path=db_path,
                    texts=texts,
                    embedder=embedder,
                    faiss_index=faiss_index,
                    llama_pipeline=llama_pipeline
                )
            except Exception as e:
                response = f"An error occurred while processing the question: {e}"
                traceback.print_exc() # Print traceback for debugging

            end_time = time.time()
            print(f"\n✅ Answer:\n{response}")
            print(f"(Time taken: {end_time - start_time:.2f} seconds)")
    else:
        print("LLaMA pipeline not loaded. Skipping question answering.")


    # --- Clean up ---
    print("\nCleaning up resources...")
    # Use try-except blocks in case initialization failed partially
    try:
        del llama_pipeline
    except NameError:
        pass
    try:
        del llama_model_ref # Delete the model reference returned by initialize_system
    except NameError:
        pass
    try:
        del embedder
    except NameError:
        pass
    try:
        del faiss_index # May need explicit cleanup depending on FAISS implementation
    except NameError:
        pass
    # Clear DataFrame and texts list if large
    try:
        del df
        del texts
    except NameError:
        pass

    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()

    print("Done.")