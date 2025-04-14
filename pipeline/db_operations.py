import sqlite3
import pandas as pd
from utils import get_db_connection, normalize_country_code

def _create_and_clear_table(cursor, table_name, create_sql):
    """Helper to create table if not exists and clear existing data."""
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}") # Drop instead of delete for schema changes
    cursor.execute(create_sql)

def _execute_and_store(conn, cursor, table_name, create_sql, insert_sql, data):
    """Helper to create and store data in a table."""
    _create_and_clear_table(cursor, table_name, create_sql)
    cursor.executemany(insert_sql, data)
    conn.commit()
    print(f"Stored {cursor.rowcount} records in {table_name}.")

def generate_and_store_analytics(df, db_path):
    """Generates and stores pre-calculated analytics using SQL queries."""
    print("Starting analytics generation and storage using SQL...")
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Revenue Trends: Monthly revenue over time
    print("Calculating and storing revenue trends...")
    _create_and_clear_table(cursor, "revenue_insights", "CREATE TABLE revenue_insights (month TEXT, year INTEGER, revenue REAL)")
    cursor.execute("""
        INSERT INTO revenue_insights (month, year, revenue)
        SELECT arrival_date_month, arrival_date_year, SUM(adr * (stays_in_weekend_nights + stays_in_week_nights))
        FROM hotel_bookings
        GROUP BY arrival_date_year, arrival_date_month
        ORDER BY arrival_date_year,
                CASE arrival_date_month
                    WHEN 'January' THEN 1 WHEN 'February' THEN 2 WHEN 'March' THEN 3 WHEN 'April' THEN 4
                    WHEN 'May' THEN 5 WHEN 'June' THEN 6 WHEN 'July' THEN 7 WHEN 'August' THEN 8
                    WHEN 'September' THEN 9 WHEN 'October' THEN 10 WHEN 'November' THEN 11 WHEN 'December' THEN 12
                    ELSE 0 END
    """)
    conn.commit()
    print(f"Stored {cursor.rowcount} revenue insights.")

    # 2. Top Cancellation Locations: Locations with the highest number of cancellations
    print("Calculating and storing top cancellation locations...")
    _create_and_clear_table(cursor, "cancellation_insights", "CREATE TABLE cancellation_insights (location TEXT, cancellations INTEGER)")
    cursor.execute("""
        INSERT INTO cancellation_insights (location, cancellations)
        SELECT country, COUNT(*)
        FROM hotel_bookings
        WHERE is_canceled = 1
        GROUP BY country
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    conn.commit()
    print(f"Stored {cursor.rowcount} top cancellation locations.")

    # 3. Average Daily Rate (ADR)
    print("Calculating and storing average ADR...")
    _create_and_clear_table(cursor, "average_adr_insight", "CREATE TABLE average_adr_insight (average_adr REAL)")
    cursor.execute("""
        INSERT INTO average_adr_insight (average_adr)
        SELECT AVG(adr)
        FROM hotel_bookings
    """)
    conn.commit()
    print(f"Stored average ADR: {cursor.fetchone()[0]:.2f}")

    # 4. Top Booking Countries: Countries with the most bookings
    print("Calculating and storing top booking countries...")
    _create_and_clear_table(cursor, "top_booking_countries", "CREATE TABLE top_booking_countries (country TEXT, count INTEGER)")
    cursor.execute("""
        INSERT INTO top_booking_countries (country, count)
        SELECT country, COUNT(*)
        FROM hotel_bookings
        GROUP BY country
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    conn.commit()
    print(f"Stored {cursor.rowcount} top booking countries.")

    # 5. Cancellation by Date: Number of cancellations per reservation status date
    print("Calculating and storing cancellations by date...")
    _create_and_clear_table(cursor, "cancellation_by_date", "CREATE TABLE cancellation_by_date (date TEXT, count INTEGER)")
    cursor.execute("""
        INSERT INTO cancellation_by_date (date, count)
        SELECT reservation_status_date, COUNT(*)
        FROM hotel_bookings
        WHERE is_canceled = 1
        GROUP BY reservation_status_date
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    conn.commit()
    print(f"Stored {cursor.rowcount} cancellation insights by date.")

    conn.close()
    print("Finished storing analytics using SQL.")