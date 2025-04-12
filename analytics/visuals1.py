import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv")

# Preprocessing
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['month_year'] = df['arrival_date'].dt.to_period('M').astype(str)
df['year'] = df['arrival_date'].dt.year
df['quarter'] = df['arrival_date'].dt.to_period('Q').astype(str)
df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)

# 1. Revenue Trends Over Time (Monthly / Quarterly / Yearly)
monthly_revenue = df.groupby('month_year')['revenue'].sum().reset_index()
quarterly_revenue = df.groupby('quarter')['revenue'].sum().reset_index()
yearly_revenue = df.groupby('year')['revenue'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_revenue, x='month_year', y='revenue', marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue (€)')
plt.xlabel('Month-Year')
plt.tight_layout()
plt.show()

# 2. Total Revenue by Hotel Type
hotel_revenue = df.groupby('hotel')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=hotel_revenue, x='hotel', y='revenue')
plt.title('Total Revenue by Hotel Type')
plt.ylabel('Revenue (€)')
plt.xlabel('Hotel Type')
plt.tight_layout()
plt.show()

# 3. Average Daily Rate (ADR) Trend Over Time
adr_monthly = df.groupby('month_year')['adr'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=adr_monthly, x='month_year', y='adr', marker='o', color='green')
plt.xticks(rotation=45)
plt.title('Average Daily Rate (ADR) Over Time')
plt.ylabel('ADR (€)')
plt.xlabel('Month-Year')
plt.tight_layout()
plt.show()

# 4. Revenue by Market Segment & Distribution Channel
segment_revenue = df.groupby(['market_segment', 'distribution_channel'])['revenue'].sum().reset_index()

plt.figure(figsize=(14, 6))
pivot_table = segment_revenue.pivot("market_segment", "distribution_channel", "revenue")
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Blues")
plt.title("Revenue by Market Segment & Distribution Channel")
plt.tight_layout()
plt.show()

# 5. Revenue per Booking and per Guest
df['revenue_per_booking'] = df['revenue']
df['revenue_per_guest'] = df['revenue'] / df['total_guests'].replace(0, 1)  # avoid div/0

avg_revenue_booking = df['revenue_per_booking'].mean()
avg_revenue_guest = df['revenue_per_guest'].mean()

print(f"Average Revenue per Booking: €{avg_revenue_booking:.2f}")
print(f"Average Revenue per Guest: €{avg_revenue_guest:.2f}")
