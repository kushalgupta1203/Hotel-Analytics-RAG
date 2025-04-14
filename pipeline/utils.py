import sqlite3
import pandas as pd
import json

# === Load Country Mapping ===
def load_country_mapping(path="D:/Projects/Hotel-Analytics-RAG/pipeline/country_mapping.json"):
    with open(path, 'r') as f:
        return json.load(f)

country_map = load_country_mapping()

def safe(x):
    return str(x) if pd.notnull(x) else "unknown"

def normalize_country_code(country_code):
    return country_map.get(country_code, country_code)

def row_to_text(row):
    return (
        f"Hotel: {safe(row['hotel'])}. "
        f"Canceled: {'Yes' if row['is_canceled'] == 1 else 'No'}. "
        f"Lead time: {safe(row['lead_time'])} days. "
        f"Arrival: {safe(row['arrival_date_day_of_month'])} {safe(row['arrival_date_month'])} {safe(row['arrival_date_year'])} "
        f"(Week {safe(row['arrival_date_week_number'])}). "
        f"Stay: {safe(row['stays_in_week_nights'])} weeknights, {safe(row['stays_in_weekend_nights'])} weekend nights. "
        f"Guests: {safe(row['adults'])} adults, {safe(row['children'])} children, {safe(row['babies'])} babies. "
        f"Meal: {safe(row['meal'])}. Country: {normalize_country_code(safe(row['country']))}. "
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
        f"Arrival month number: {safe(row['arrival_month_num'])}. "
        f"Arrival date: {safe(row['arrival_date'])}. "
        f"Revenue: {safe(row['revenue'])}."
    )

def get_db_connection(db_path=r"D:/Projects/Hotel-Analytics-RAG/dataset/analytics.db"):
    conn = sqlite3.connect(db_path)
    return conn

def fetch_all_results(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()
