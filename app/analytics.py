import pandas as pd
import numpy as np

# Load data
csv_path = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv"
df = pd.read_csv(csv_path)

def get_analytics_report(metrics):
    report = {}
    for metric in metrics:
        if metric == "cancellation_rate":
            cancellation_rate = df["is_canceled"].mean() * 100
            report["cancellation_rate"] = cancellation_rate
        elif metric == "average_revenue":
            avg_revenue = df["revenue"].mean()
            report["average_revenue"] = avg_revenue
        elif metric == "lead_time":
            avg_lead_time = df["lead_time"].mean()
            report["lead_time"] = avg_lead_time
        elif metric == "booking_by_month":
            bookings_by_month = df.groupby('arrival_date_month').size()
            report["booking_by_month"] = bookings_by_month.to_dict()
        elif metric == "booking_by_week":
            bookings_by_week = df.groupby('arrival_date_week_number').size()
            report["booking_by_week"] = bookings_by_week.to_dict()
        # Add more metrics as needed

    return report
