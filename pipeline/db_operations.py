import sqlite3
import pandas as pd
# Import get_db_connection from utils instead of redefining it
from utils import get_db_connection, execute_query, execute_many_query, normalize_country_code

def _create_and_clear_table(cursor, table_name, create_sql):
    """Helper to create table if not exists and clear existing data."""
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}") # Drop instead of delete for schema changes
    cursor.execute(create_sql)
    # cursor.execute(f"DELETE FROM {table_name}") # Use DELETE if you want to keep the table schema

def store_revenue_insights(df, db_path):
    """Calculates and stores total revenue per month/year."""
    # Ensure revenue exists or calculate it
    if 'revenue' not in df.columns:
         # Check if required columns exist
        if 'adr' in df.columns and 'total_nights' in df.columns:
            # Fill NaNs before multiplication
            df['adr'] = df['adr'].fillna(0)
            df['total_nights'] = df['total_nights'].fillna(0)
            df["revenue"] = df["adr"] * df["total_nights"]
        else:
            print("Warning: Cannot calculate revenue, 'adr' or 'total_nights' column missing.")
            return # Exit if calculation isn't possible

    # Ensure necessary date columns exist
    if not all(col in df.columns for col in ['arrival_date_month', 'arrival_date_year']):
        print("Warning: 'arrival_date_month' or 'arrival_date_year' missing. Cannot store revenue insights.")
        return

    # Handle potential NaN in grouping columns
    df_grouped = df.dropna(subset=['arrival_date_month', 'arrival_date_year', 'revenue'])

    grouped = df_grouped.groupby(["arrival_date_month", "arrival_date_year"])["revenue"].sum().reset_index()
    grouped.columns = ["month", "year", "revenue"]

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    table_name = "revenue_insights"
    create_sql = "CREATE TABLE revenue_insights (month TEXT, year INTEGER, revenue REAL)"
    _create_and_clear_table(cursor, table_name, create_sql)

    insert_sql = "INSERT INTO revenue_insights (month, year, revenue) VALUES (?, ?, ?)"
    execute_many_query(cursor, insert_sql, grouped.to_records(index=False))

    conn.commit()
    conn.close()
    print(f"Stored {len(grouped)} revenue insights.")


def store_cancellation_insights(df, db_path):
    """Calculates and stores cancellation counts per country."""
    if not all(col in df.columns for col in ['is_canceled', 'country']):
        print("Warning: 'is_canceled' or 'country' missing. Cannot store cancellation insights.")
        return

    # Normalize country codes *before* grouping
    df['country_normalized'] = df['country'].apply(normalize_country_code)

    # Filter and group
    grouped = df[(df["is_canceled"] == 1) & (df['country_normalized'].notna())]\
                .groupby("country_normalized")\
                .size()\
                .reset_index(name="cancellations")
    grouped.columns = ["location", "cancellations"]
    grouped = grouped.sort_values("cancellations", ascending=False) # Sort here

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    table_name = "cancellation_insights"
    create_sql = "CREATE TABLE cancellation_insights (location TEXT, cancellations INTEGER)"
    _create_and_clear_table(cursor, table_name, create_sql)

    insert_sql = "INSERT INTO cancellation_insights (location, cancellations) VALUES (?, ?)"
    execute_many_query(cursor, insert_sql, grouped.to_records(index=False))

    conn.commit()
    conn.close()
    print(f"Stored {len(grouped)} cancellation insights by location.")


def store_cancellations_by_date_insights(df, db_path): # Renamed function
    """Calculates and stores cancellation counts per date."""
    if not all(col in df.columns for col in ['is_canceled', 'reservation_status_date']):
        print("Warning: 'is_canceled' or 'reservation_status_date' missing. Cannot store cancellations by date.")
        return

    canceled_by_date = df[(df["is_canceled"] == 1) & (df['reservation_status_date'].notna())]\
                        .groupby("reservation_status_date")\
                        .size()\
                        .reset_index(name="count")
    canceled_by_date = canceled_by_date.sort_values("count", ascending=False) # Sort here

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    table_name = "cancellation_by_date"
    create_sql = "CREATE TABLE cancellation_by_date (date TEXT, count INTEGER)"
    _create_and_clear_table(cursor, table_name, create_sql)

    insert_sql = "INSERT INTO cancellation_by_date (date, count) VALUES (?, ?)"
    execute_many_query(cursor, insert_sql, canceled_by_date.to_records(index=False))

    conn.commit()
    conn.close()
    print(f"Stored {len(canceled_by_date)} cancellation insights by date.")


def store_average_adr(df, db_path):
    """Calculates and stores the overall average ADR."""
    if 'adr' not in df.columns:
        print("Warning: 'adr' column missing. Cannot store average ADR.")
        return

    avg_adr = df['adr'].mean()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    table_name = "average_adr_insight"
    create_sql = "CREATE TABLE average_adr_insight (average_adr REAL)"
    _create_and_clear_table(cursor, table_name, create_sql)

    insert_sql = "INSERT INTO average_adr_insight (average_adr) VALUES (?)"
    execute_query(cursor, insert_sql, (avg_adr,))

    conn.commit()
    conn.close()
    print(f"Stored average ADR: {avg_adr:.2f}")


def store_top_booking_countries(df, db_path, top_n=10):
    """Calculates and stores the top N booking countries."""
    if 'country' not in df.columns:
        print("Warning: 'country' column missing. Cannot store top booking countries.")
        return

    # Normalize country codes *before* grouping
    df['country_normalized'] = df['country'].apply(normalize_country_code)

    top_countries = df['country_normalized'].value_counts().head(top_n).reset_index()
    top_countries.columns = ['country', 'count']

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    table_name = "top_booking_countries"
    create_sql = "CREATE TABLE top_booking_countries (country TEXT, count INTEGER)"
    _create_and_clear_table(cursor, table_name, create_sql)

    insert_sql = "INSERT INTO top_booking_countries (country, count) VALUES (?, ?)"
    execute_many_query(cursor, insert_sql, top_countries.to_records(index=False))

    conn.commit()
    conn.close()
    print(f"Stored top {len(top_countries)} booking countries.")


def generate_and_store_analytics(df, db_path):
    """Generates and stores all pre-calculated analytics."""
    print("Starting analytics generation and storage...")
    store_revenue_insights(df, db_path)
    store_cancellation_insights(df, db_path)
    store_cancellations_by_date_insights(df, db_path) # Use renamed function
    store_average_adr(df, db_path)
    store_top_booking_countries(df, db_path)
    print("Finished storing analytics.")