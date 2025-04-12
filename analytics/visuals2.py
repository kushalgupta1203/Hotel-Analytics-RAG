import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv")

# 1. Cancellation Rate (% of total bookings)
cancellation_rate = df['is_canceled'].mean() * 100
print(f"Overall Cancellation Rate: {cancellation_rate:.2f}%")

# 2. Cancellation Rate by Hotel Type
hotel_cancel_rate = df.groupby('hotel')['is_canceled'].mean().reset_index()
hotel_cancel_rate['is_canceled'] *= 100

plt.figure(figsize=(7, 5))
sns.barplot(data=hotel_cancel_rate, x='hotel', y='is_canceled')
plt.title("Cancellation Rate by Hotel Type")
plt.ylabel("Cancellation Rate (%)")
plt.xlabel("Hotel")
plt.tight_layout()
plt.show()

# 2b. Cancellation Rate by Country (Top 10 only)
top_countries = df['country'].value_counts().head(10).index
country_cancel_rate = df[df['country'].isin(top_countries)].groupby('country')['is_canceled'].mean().reset_index()
country_cancel_rate['is_canceled'] *= 100

plt.figure(figsize=(10, 5))
sns.barplot(data=country_cancel_rate, x='country', y='is_canceled')
plt.title("Cancellation Rate by Top 10 Countries")
plt.ylabel("Cancellation Rate (%)")
plt.xlabel("Country")
plt.tight_layout()
plt.show()

# 2c. Cancellation Rate by Customer Type
cust_cancel_rate = df.groupby('customer_type')['is_canceled'].mean().reset_index()
cust_cancel_rate['is_canceled'] *= 100

plt.figure(figsize=(8, 5))
sns.barplot(data=cust_cancel_rate, x='customer_type', y='is_canceled')
plt.title("Cancellation Rate by Customer Type")
plt.ylabel("Cancellation Rate (%)")
plt.xlabel("Customer Type")
plt.tight_layout()
plt.show()

# 3. Lead Time Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.title("Booking Lead Time Distribution")
plt.xlabel("Lead Time (days)")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

# 4. Cancellation vs Lead Time
plt.figure(figsize=(10, 5))
sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.title("Lead Time vs Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Lead Time (days)")
plt.tight_layout()
plt.show()

# 5. Cancellation vs Deposit Type
deposit_cancel_rate = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
deposit_cancel_rate['is_canceled'] *= 100

plt.figure(figsize=(8, 5))
sns.barplot(data=deposit_cancel_rate, x='deposit_type', y='is_canceled')
plt.title("Cancellation Rate by Deposit Type")
plt.ylabel("Cancellation Rate (%)")
plt.xlabel("Deposit Type")
plt.tight_layout()
plt.show()