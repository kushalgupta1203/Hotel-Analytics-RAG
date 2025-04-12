import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv")

# Preprocessing
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['arrival_month_name'] = df['arrival_date'].dt.month_name()

# --------------------------
# 🔹 Stay Patterns
# --------------------------

# 1. Average Stay Duration
avg_stay = df['total_nights'].mean()
print(f"Average Stay Duration: {avg_stay:.2f} nights")

# 2. Weekend vs Weekday Stay
df_melted = df.melt(id_vars='hotel', value_vars=['stays_in_weekend_nights', 'stays_in_week_nights'],
                    var_name='stay_type', value_name='nights')

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_melted, x='stay_type', y='nights')
plt.title("Weekend vs Weekday Stay Duration")
plt.ylabel("Nights")
plt.xlabel("Stay Type")
plt.tight_layout()
plt.show()

# 3. Room Type Preferences
room_type_ct = pd.crosstab(df['reserved_room_type'], df['assigned_room_type'])

plt.figure(figsize=(10, 6))
sns.heatmap(room_type_ct, annot=True, fmt="d", cmap="Purples")
plt.title("Reserved vs Assigned Room Types")
plt.tight_layout()
plt.show()

# 4. Impact of Room Changes
room_change_stats = df.groupby('booking_changes')[['is_canceled', 'revenue']].mean().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=room_change_stats, x='booking_changes', y='is_canceled', label="Cancellation Rate")
sns.lineplot(data=room_change_stats, x='booking_changes', y='revenue', label="Average Revenue")
plt.title("Impact of Room Changes on Cancellation & Revenue")
plt.xlabel("Booking Changes")
plt.ylabel("Rate / Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Seasonal Trends
seasonal_avg = df.groupby('arrival_month_name')['revenue'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(12, 5))
seasonal_avg.plot(kind='bar', color='skyblue')
plt.title("Average Revenue by Arrival Month")
plt.ylabel("Revenue (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------
# 🔹 Agent/Company Analytics
# --------------------------

# 1. Top Booking Agents
top_agents = df['agent'].value_counts().head(10).reset_index()
top_agents.columns = ['agent', 'booking_count']

plt.figure(figsize=(10, 5))
sns.barplot(data=top_agents, x='agent', y='booking_count')
plt.title("Top 10 Booking Agents")
plt.xlabel("Agent ID")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

# 2. Revenue and Cancellations by Agent
agent_stats = df.groupby('agent').agg({
    'revenue': 'sum',
    'is_canceled': 'mean'
}).reset_index()
agent_stats['is_canceled'] *= 100
top_agents_data = agent_stats.sort_values('revenue', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_agents_data, x='agent', y='revenue')
plt.title("Top 10 Agents by Revenue")
plt.tight_layout()
plt.show()

# 3. Company-Based Booking Trends
company_stats = df['company'].value_counts().head(10).reset_index()
company_stats.columns = ['company', 'bookings']

plt.figure(figsize=(10, 5))
sns.barplot(data=company_stats, x='company', y='bookings')
plt.title("Top 10 Companies by Booking Volume")
plt.tight_layout()
plt.show()

# --------------------------
# 🔹 Customer Segmentation
# --------------------------

# 1. Bookings by Customer Type
cust_type_counts = df['customer_type'].value_counts().reset_index()
cust_type_counts.columns = ['customer_type', 'count']

plt.figure(figsize=(8, 5))
sns.barplot(data=cust_type_counts, x='customer_type', y='count')
plt.title("Bookings by Customer Type")
plt.tight_layout()
plt.show()

# 2. ADR by Customer Type
adr_cust = df.groupby('customer_type')['adr'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=adr_cust, x='customer_type', y='adr')
plt.title("Average ADR by Customer Type")
plt.tight_layout()
plt.show()

# 3. Special Requests by Customer Type
req_cust = df.groupby('customer_type')['total_of_special_requests'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=req_cust, x='customer_type', y='total_of_special_requests')
plt.title("Average Special Requests by Customer Type")
plt.tight_layout()
plt.show()

# --------------------------
# 🔹 Waitlist & Status Analysis
# --------------------------

# 1. Waiting List Trends
plt.figure(figsize=(10, 5))
sns.histplot(df['days_in_waiting_list'], bins=50, kde=True)
plt.title("Days in Waiting List Distribution")
plt.tight_layout()
plt.show()

# 2. Reservation Status Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='reservation_status', order=df['reservation_status'].value_counts().index)
plt.title("Reservation Status Distribution")
plt.tight_layout()
plt.show()

# 3. Waitlist vs Booking Success
waitlist_status = df.groupby('days_in_waiting_list')['is_canceled'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=waitlist_status, x='days_in_waiting_list', y='is_canceled')
plt.title("Waitlist Days vs Cancellation Rate")
plt.ylabel("Cancellation Rate")
plt.tight_layout()
plt.show()

# --------------------------
# 🔹 Advanced Insights
# --------------------------

# 1. Correlation Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 2. Feature Importance for Cancellation
features = ['lead_time', 'adr', 'booking_changes', 'total_of_special_requests', 'total_nights', 
            'previous_cancellations', 'previous_bookings_not_canceled', 'days_in_waiting_list']
df_ml = df[features + ['is_canceled']].dropna()

X = df_ml[features]
y = df_ml['is_canceled']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance for Cancellation Prediction")
plt.tight_layout()
plt.show()

# 3. Time Between Booking and Arrival vs Cancellation
plt.figure(figsize=(10, 5))
sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.title("Lead Time vs Cancellation Status")
plt.tight_layout()
plt.show()
