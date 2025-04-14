import sqlite3
import pandas as pd
import json
import os

# === Configuration (Paths) ===
# It's better to manage paths via environment variables, config files, or arguments,
# but defining them here for clarity in this example.
DEFAULT_PIPELINE_DIR = os.path.join(os.path.dirname(__file__)) # Assumes utils.py is in 'pipeline'
DEFAULT_DATASET_DIR = os.path.join(DEFAULT_PIPELINE_DIR, '..', 'dataset') # Assumes 'dataset' is one level up

DEFAULT_COUNTRY_MAPPING_PATH = os.path.join(DEFAULT_PIPELINE_DIR, "country_mapping.json")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DATASET_DIR, "analytics.db")

# === Load Country Mapping ===
def load_country_mapping(path=DEFAULT_COUNTRY_MAPPING_PATH):
    """Loads country code to name mapping from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Country mapping file not found at {path}. Using codes directly.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {path}. Using codes directly.")
        return {}

country_map = load_country_mapping()

def safe(x):
    """Converts input to string, handling pandas NaNs."""
    return str(x) if pd.notnull(x) else "unknown"

def normalize_country_code(country_code):
    """Converts country code to full name using the loaded map."""
    # Ensure input is treated as string code before lookup
    code_str = safe(country_code)
    return country_map.get(code_str, code_str) # Return original code if not found

def row_to_text(row):
    """Converts a DataFrame row into a descriptive text string."""
    # Ensure country is normalized before adding to text
    country_name = normalize_country_code(row['country'])

    return (
        f"Hotel: {safe(row['hotel'])}. "
        f"Canceled: {'Yes' if row['is_canceled'] == 1 else 'No'}. "
        f"Lead time: {safe(row['lead_time'])} days. "
        # Corrected date formatting slightly for consistency
        f"Arrival: {safe(row['arrival_date_day_of_month'])}-{safe(row['arrival_date_month'])}-{safe(row['arrival_date_year'])} "
        f"(Week {safe(row['arrival_date_week_number'])}). "
        f"Stay: {safe(row['stays_in_week_nights'])} weeknights, {safe(row['stays_in_weekend_nights'])} weekend nights. "
        f"Guests: {safe(row['adults'])} adults, {safe(row['children'])} children, {safe(row['babies'])} babies. "
        f"Meal: {safe(row['meal'])}. Country: {country_name}. " # Use normalized name
        f"Market segment: {safe(row['market_segment'])}, Channel: {safe(row['distribution_channel'])}. "
        f"Repeated guest: {'Yes' if row['is_repeated_guest'] == 1 else 'No'}. "
        f"Previous cancellations: {safe(row['previous_cancellations'])}, "
        f"Previous bookings not canceled: {safe(row['previous_bookings_not_canceled'])}. "
        f"Room reserved: {safe(row['reserved_room_type'])}, Assigned: {safe(row['assigned_room_type'])}. "
        f"Booking changes: {safe(row['booking_changes'])}. "
        f"Deposit type: {safe(row['deposit_type'])}. "
        f"Agent: {safe(row['agent'])}, Company: {safe(row['company'])}. "
        f"Waiting list days: {safe(row['days_in_waiting_list'])}. "
        f"Customer type: {safe(row['customer_type'])}. "
        f"ADR: {safe(row['adr'])}. "
        f"Parking spaces: {safe(row['required_car_parking_spaces'])}. "
        f"Special requests: {safe(row['total_of_special_requests'])}. "
        f"Reservation status: {safe(row['reservation_status'])} on {safe(row['reservation_status_date'])}. "
        f"Total nights: {safe(row['total_nights'])}. "
        # Removed less useful fields for RAG context unless needed
        # f"Arrival month number: {safe(row['arrival_month_num'])}. "
        # f"Arrival date: {safe(row['arrival_date'])}. "
        f"Revenue: {safe(row['revenue'])}."
    )

def get_db_connection(db_path=DEFAULT_DB_PATH):
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn

def fetch_all_results(cursor, query, params=None):
    """Executes a query with optional parameters and fetches all results."""
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor.fetchall()

def fetch_one_result(cursor, query, params=None):
    """Executes a query with optional parameters and fetches one result."""
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor.fetchone()

def execute_query(cursor, query, params=None):
    """Executes a query with optional parameters."""
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)

def execute_many_query(cursor, query, data):
    """Executes a query for multiple rows of data."""
    cursor.executemany(query, data)