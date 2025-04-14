import sqlite3
import pandas as pd

# Aggregate and store precomputed insights

def store_revenue_insights(df):
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['month'] = df['arrival_date'].dt.month_name()
    df['year'] = df['arrival_date'].dt.year
    monthly_revenue = df.groupby(['month', 'year'])['revenue'].sum().reset_index()

    conn = sqlite3.connect("D:/Projects/Hotel-Analytics-RAG/analytics.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS revenue_insights (
            month TEXT,
            year INTEGER,
            revenue REAL
        )
    """)

    for _, row in monthly_revenue.iterrows():
        cursor.execute("INSERT INTO revenue_insights (month, year, revenue) VALUES (?, ?, ?)",
                       (row['month'], row['year'], row['revenue']))

    conn.commit()
    conn.close()

def store_cancellation_insights(df):
    location_cancellations = df.groupby('country')['is_canceled'].sum().reset_index()

    conn = sqlite3.connect("D:/Projects/Hotel-Analytics-RAG/analytics.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cancellation_insights (
            location TEXT,
            cancellations INTEGER
        )
    """)

    for _, row in location_cancellations.iterrows():
        cursor.execute("INSERT INTO cancellation_insights (location, cancellations) VALUES (?, ?)",
                       (row['country'], row['is_canceled']))

    conn.commit()
    conn.close()
