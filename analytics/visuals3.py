import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
df = pd.read_csv(r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv")

# Fill missing country codes (if any) with "Unknown"
df['country'] = df['country'].fillna("Unknown")

# 1. Top Countries by Booking Volume
country_counts = df['country'].value_counts().reset_index()
country_counts.columns = ['country', 'booking_count']
top_countries = country_counts.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_countries, x='country', y='booking_count')
plt.title("Top 10 Countries by Booking Volume")
plt.xlabel("Country")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

# 2. Cancellation Rate by Country (for countries with > 100 bookings)
country_cancel = df.groupby('country').agg(
    bookings=('is_canceled', 'count'),
    cancel_rate=('is_canceled', 'mean')
).reset_index()
country_cancel = country_cancel[country_cancel['bookings'] > 100]
country_cancel['cancel_rate'] *= 100

top_cancel_countries = country_cancel.sort_values('cancel_rate', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_cancel_countries, x='country', y='cancel_rate')
plt.title("Top 10 Countries by Cancellation Rate (min 100 bookings)")
plt.xlabel("Country")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

# 3. Average Revenue by Country (filter for top 10 countries by booking count)
country_revenue = df.groupby('country')['revenue'].mean().reset_index()
top_revenue_countries = country_revenue[country_revenue['country'].isin(top_countries['country'])]

plt.figure(figsize=(10, 6))
sns.barplot(data=top_revenue_countries, x='country', y='revenue')
plt.title("Average Revenue by Country (Top 10 by bookings)")
plt.xlabel("Country")
plt.ylabel("Average Revenue (€)")
plt.tight_layout()
plt.show()

# 4. Map Visualization of Booking Origins using Plotly
country_summary = df.groupby('country').agg(
    bookings=('is_canceled', 'count'),
    revenue=('revenue', 'sum')
).reset_index()

fig = px.choropleth(
    country_summary,
    locations='country',
    locationmode='ISO-3',  # Use 3-letter ISO codes if available
    color='bookings',
    hover_name='country',
    color_continuous_scale='Viridis',
    title='Global Booking Volume by Country'
)
fig.show()
