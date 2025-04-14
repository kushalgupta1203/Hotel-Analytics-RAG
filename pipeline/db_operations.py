import sqlite3
import pandas as pd

def get_db_connection():
    return sqlite3.connect(r"D:\Projects\Hotel-Analytics-RAG\dataset\analytics.db")

def fetch_all_results(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def store_revenue_insights(df):
    df["arrival_month"] = df["arrival_date_month"]
    df["arrival_year"] = df["arrival_date_year"]
    df["revenue"] = df["adr"] * df["total_nights"]

    grouped = df.groupby(["arrival_month", "arrival_year"])["revenue"].sum().reset_index()
    grouped.columns = ["month", "year", "revenue"]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS revenue_insights (month TEXT, year INTEGER, revenue REAL)")
    cursor.execute("DELETE FROM revenue_insights")

    for _, row in grouped.iterrows():
        cursor.execute("INSERT INTO revenue_insights (month, year, revenue) VALUES (?, ?, ?)",
                       (row["month"], row["year"], row["revenue"]))
    conn.commit()
    conn.close()

def store_cancellation_insights(df):
    grouped = df[df["is_canceled"] == 1].groupby("country").size().reset_index(name="cancellations")
    grouped.columns = ["location", "cancellations"]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS cancellation_insights (location TEXT, cancellations INTEGER)")
    cursor.execute("DELETE FROM cancellation_insights")

    for _, row in grouped.iterrows():
        cursor.execute("INSERT INTO cancellation_insights (location, cancellations) VALUES (?, ?)",
                       (row["location"], row["cancellations"]))
    conn.commit()
    conn.close()

def store_non_mathematical_insights(df):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS cancellation_by_date (date TEXT, count INTEGER)")
    cursor.execute("DELETE FROM cancellation_by_date")

    canceled_by_date = df[df["is_canceled"] == 1].groupby("reservation_status_date").size().reset_index(name="count")
    for _, row in canceled_by_date.iterrows():
        cursor.execute("INSERT INTO cancellation_by_date (date, count) VALUES (?, ?)",
                       (row["reservation_status_date"], row["count"]))
    conn.commit()
    conn.close()

def generate_and_store_analytics(df):
    store_revenue_insights(df)
    store_cancellation_insights(df)
    store_non_mathematical_insights(df)
