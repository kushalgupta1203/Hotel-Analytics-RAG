import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Hotel Analytics Dashboard",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #3B82F6;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏨 Hotel Analytics Dashboard</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Overview", "Revenue Analysis", "Cancellation Analysis", "Geographic Analysis", 
     "Stay Patterns", "Customer Segmentation", "Advanced Insights"]
)

# Data loading function
@st.cache_data
def load_data():
    # Update the path to where your data is stored
    df = pd.read_csv("hotel_bookings_dataset.csv")
    
    # Data preprocessing
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['month_year'] = df['arrival_date'].dt.to_period('M').astype(str)
    df['year'] = df['arrival_date'].dt.year
    df['quarter'] = df['arrival_date'].dt.to_period('Q').astype(str)
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['arrival_month_name'] = df['arrival_date'].dt.month_name()
    df['country'] = df['country'].fillna("Unknown")
    df['revenue_per_booking'] = df['revenue']
    df['revenue_per_guest'] = df['revenue'] / df['total_guests'].replace(0, 1)  # avoid div/0
    
    return df

# Load data
try:
    df = load_data()
    success_load = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    success_load = False
    df = None

if success_load:
    # Set up filter sidebar
    st.sidebar.markdown("## Filters")
    
    # Date range filter
    min_date = df['arrival_date'].min().date()
    max_date = df['arrival_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['arrival_date'].dt.date >= start_date) & 
                         (df['arrival_date'].dt.date <= end_date)]
    else:
        filtered_df = df.copy()
    
    # Hotel type filter
    hotel_types = st.sidebar.multiselect(
        "Hotel Type",
        options=df['hotel'].unique(),
        default=df['hotel'].unique()
    )
    
    if hotel_types:
        filtered_df = filtered_df[filtered_df['hotel'].isin(hotel_types)]
    
    # Customer type filter
    customer_types = st.sidebar.multiselect(
        "Customer Type",
        options=df['customer_type'].unique(),
        default=df['customer_type'].unique()
    )
    
    if customer_types:
        filtered_df = filtered_df[filtered_df['customer_type'].isin(customer_types)]
    
    # Apply filters
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
    else:
        # ===== PAGES =====
        
        # ===== OVERVIEW PAGE =====
        if page == "Overview":
            st.markdown('<div class="section-header">📊 Hotel Booking Overview</div>', unsafe_allow_html=True)
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Bookings", f"{len(filtered_df):,}")
            col2.metric("Total Revenue", f"€{filtered_df['revenue'].sum():,.2f}")
            col3.metric("Average Daily Rate", f"€{filtered_df['adr'].mean():.2f}")
            col4.metric("Cancellation Rate", f"{(filtered_df['is_canceled'].mean() * 100):.1f}%")
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Bookings by Hotel Type</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                hotel_counts = filtered_df['hotel'].value_counts()
                sns.barplot(x=hotel_counts.index, y=hotel_counts.values, ax=ax)
                plt.ylabel("Number of Bookings")
                plt.xlabel("Hotel Type")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Booking Status</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                status_counts = filtered_df['reservation_status'].value_counts()
                plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Monthly Booking Trends</div>', unsafe_allow_html=True)
            monthly_bookings = filtered_df.groupby('month_year').size().reset_index(name='booking_count')
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.lineplot(data=monthly_bookings, x='month_year', y='booking_count', marker='o')
            plt.xticks(rotation=45)
            plt.ylabel("Number of Bookings")
            plt.xlabel("Month-Year")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Customer Type Distribution
            st.markdown('<div class="subsection-header">Customer Type Distribution</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                cust_type_counts = filtered_df['customer_type'].value_counts().reset_index()
                cust_type_counts.columns = ['customer_type', 'count']
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=cust_type_counts, x='customer_type', y='count')
                plt.title("Bookings by Customer Type")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Room Type Distribution
                room_type_counts = filtered_df['reserved_room_type'].value_counts().reset_index()
                room_type_counts.columns = ['room_type', 'count']
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=room_type_counts, x='room_type', y='count')
                plt.title("Room Type Distribution")
                plt.tight_layout()
                st.pyplot(fig)
        
        # ===== REVENUE ANALYSIS PAGE =====
        elif page == "Revenue Analysis":
            st.markdown('<div class="section-header">💰 Revenue Analysis</div>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Total Revenue", f"€{filtered_df['revenue'].sum():,.2f}")
            col2.metric("Average Revenue per Booking", f"€{filtered_df['revenue_per_booking'].mean():.2f}")
            col3.metric("Average Revenue per Guest", f"€{filtered_df['revenue_per_guest'].mean():.2f}")
            
            # Revenue Trends
            st.markdown('<div class="subsection-header">Revenue Trends Over Time</div>', unsafe_allow_html=True)
            
            time_period = st.radio(
                "Select Time Period:",
                ["Monthly", "Quarterly", "Yearly"]
            )
            
            if time_period == "Monthly":
                monthly_revenue = filtered_df.groupby('month_year')['revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(15, 6))
                sns.lineplot(data=monthly_revenue, x='month_year', y='revenue', marker='o')
                plt.xticks(rotation=45)
                plt.title('Monthly Revenue Trend')
                plt.ylabel('Revenue (€)')
                plt.xlabel('Month-Year')
                plt.tight_layout()
                st.pyplot(fig)
            
            elif time_period == "Quarterly":
                quarterly_revenue = filtered_df.groupby('quarter')['revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=quarterly_revenue, x='quarter', y='revenue')
                plt.title('Quarterly Revenue Trend')
                plt.ylabel('Revenue (€)')
                plt.xlabel('Quarter')
                plt.tight_layout()
                st.pyplot(fig)
            
            else:  # Yearly
                yearly_revenue = filtered_df.groupby('year')['revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=yearly_revenue, x='year', y='revenue')
                plt.title('Yearly Revenue Trend')
                plt.ylabel('Revenue (€)')
                plt.xlabel('Year')
                plt.tight_layout()
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Revenue by Hotel Type</div>', unsafe_allow_html=True)
                hotel_revenue = filtered_df.groupby('hotel')['revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=hotel_revenue, x='hotel', y='revenue')
                plt.title('Total Revenue by Hotel Type')
                plt.ylabel('Revenue (€)')
                plt.xlabel('Hotel Type')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Average Daily Rate (ADR) Trend</div>', unsafe_allow_html=True)
                adr_monthly = filtered_df.groupby('month_year')['adr'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.lineplot(data=adr_monthly, x='month_year', y='adr', marker='o', color='green')
                plt.xticks(rotation=45)
                plt.title('Average Daily Rate (ADR) Over Time')
                plt.ylabel('ADR (€)')
                plt.xlabel('Month-Year')
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Revenue by Market Segment & Distribution Channel</div>', unsafe_allow_html=True)
            segment_revenue = filtered_df.groupby(['market_segment', 'distribution_channel'])['revenue'].sum().reset_index()
            
            pivot_table = segment_revenue.pivot("market_segment", "distribution_channel", "revenue")
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Blues")
            plt.title("Revenue by Market Segment & Distribution Channel")
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Seasonal Revenue Analysis</div>', unsafe_allow_html=True)
            seasonal_avg = filtered_df.groupby('arrival_month_name')['revenue'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            fig, ax = plt.subplots(figsize=(12, 5))
            seasonal_avg.plot(kind='bar', color='skyblue', ax=ax)
            plt.title("Average Revenue by Arrival Month")
            plt.ylabel("Revenue (€)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        # ===== CANCELLATION ANALYSIS PAGE =====
        elif page == "Cancellation Analysis":
            st.markdown('<div class="section-header">❌ Cancellation Analysis</div>', unsafe_allow_html=True)
            
            # Metrics
            cancellation_rate = filtered_df['is_canceled'].mean() * 100
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Overall Cancellation Rate", f"{cancellation_rate:.2f}%")
            col2.metric("Total Cancellations", f"{filtered_df['is_canceled'].sum():,}")
            col3.metric("Average Lead Time for Canceled Bookings", 
                       f"{filtered_df[filtered_df['is_canceled']==1]['lead_time'].mean():.1f} days")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Cancellation Rate by Hotel Type</div>', unsafe_allow_html=True)
                hotel_cancel_rate = filtered_df.groupby('hotel')['is_canceled'].mean().reset_index()
                hotel_cancel_rate['is_canceled'] *= 100
                
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(data=hotel_cancel_rate, x='hotel', y='is_canceled')
                plt.title("Cancellation Rate by Hotel Type")
                plt.ylabel("Cancellation Rate (%)")
                plt.xlabel("Hotel")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Cancellation Rate by Customer Type</div>', unsafe_allow_html=True)
                cust_cancel_rate = filtered_df.groupby('customer_type')['is_canceled'].mean().reset_index()
                cust_cancel_rate['is_canceled'] *= 100
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=cust_cancel_rate, x='customer_type', y='is_canceled')
                plt.title("Cancellation Rate by Customer Type")
                plt.ylabel("Cancellation Rate (%)")
                plt.xlabel("Customer Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Cancellation Rate by Country (Top 10)</div>', unsafe_allow_html=True)
            # Filter countries with at least 20 bookings for better visualization
            country_bookings = filtered_df['country'].value_counts()
            valid_countries = country_bookings[country_bookings > 20].index
            
            if len(valid_countries) > 0:
                country_data = filtered_df[filtered_df['country'].isin(valid_countries)]
                country_cancel_rate = country_data.groupby('country')['is_canceled'].mean().reset_index()
                country_cancel_rate['is_canceled'] *= 100
                
                # Get top 10 by cancellation rate
                top_cancel_countries = country_cancel_rate.sort_values('is_canceled', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=top_cancel_countries, x='country', y='is_canceled')
                plt.title("Top 10 Countries by Cancellation Rate")
                plt.ylabel("Cancellation Rate (%)")
                plt.xlabel("Country")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Lead Time Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(filtered_df['lead_time'], bins=50, kde=True, ax=ax)
                plt.title("Booking Lead Time Distribution")
                plt.xlabel("Lead Time (days)")
                plt.ylabel("Number of Bookings")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Cancellation vs Lead Time</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x='is_canceled', y='lead_time', data=filtered_df, ax=ax)
                plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
                plt.title("Lead Time vs Cancellation")
                plt.xlabel("Booking Status")
                plt.ylabel("Lead Time (days)")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Cancellation vs Deposit Type</div>', unsafe_allow_html=True)
            deposit_cancel_rate = filtered_df.groupby('deposit_type')['is_canceled'].mean().reset_index()
            deposit_cancel_rate['is_canceled'] *= 100
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=deposit_cancel_rate, x='deposit_type', y='is_canceled')
            plt.title("Cancellation Rate by Deposit Type")
            plt.ylabel("Cancellation Rate (%)")
            plt.xlabel("Deposit Type")
            plt.tight_layout()
            st.pyplot(fig)
        
        # ===== GEOGRAPHIC ANALYSIS PAGE =====
        elif page == "Geographic Analysis":
            st.markdown('<div class="section-header">🌎 Geographic Analysis</div>', unsafe_allow_html=True)
            
            # Top Countries by Bookings
            country_counts = filtered_df['country'].value_counts().reset_index()
            country_counts.columns = ['country', 'booking_count']
            top_countries = country_counts.head(10)
            
            st.markdown('<div class="subsection-header">Top 10 Countries by Booking Volume</div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=top_countries, x='country', y='booking_count')
            plt.title("Top 10 Countries by Booking Volume")
            plt.xlabel("Country")
            plt.ylabel("Number of Bookings")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Average Revenue by Country</div>', unsafe_allow_html=True)
                country_revenue = filtered_df.groupby('country')['revenue'].mean().reset_index()
                top_revenue_countries = country_revenue[country_revenue['country'].isin(top_countries['country'])]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_revenue_countries, x='country', y='revenue')
                plt.title("Average Revenue by Country (Top 10 by bookings)")
                plt.xlabel("Country")
                plt.ylabel("Average Revenue (€)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Cancellation Rate by Country</div>', unsafe_allow_html=True)
                top_countries_data = filtered_df[filtered_df['country'].isin(top_countries['country'])]
                country_cancel = top_countries_data.groupby('country')['is_canceled'].mean().reset_index()
                country_cancel['is_canceled'] *= 100
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=country_cancel, x='country', y='is_canceled')
                plt.title("Cancellation Rate by Country (Top 10 by bookings)")
                plt.xlabel("Country")
                plt.ylabel("Cancellation Rate (%)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Prepare data for the map
            country_summary = filtered_df.groupby('country').agg(
                bookings=('is_canceled', 'count'),
                revenue=('revenue', 'sum'),
                cancel_rate=('is_canceled', 'mean')
            ).reset_index()
            country_summary['cancel_rate'] *= 100
            
            st.markdown('<div class="subsection-header">Global Booking Distribution</div>', unsafe_allow_html=True)
            metric = st.radio("Select Metric for Map:", ["bookings", "revenue", "cancel_rate"])
            
            if metric == "bookings":
                title = "Global Booking Volume by Country"
                color_scale = "Viridis"
            elif metric == "revenue":
                title = "Total Revenue by Country"
                color_scale = "Greens"
            else:
                title = "Cancellation Rate by Country"
                color_scale = "Reds"
            
            # Create Plotly map
            fig = px.choropleth(
                country_summary,
                locations='country',
                locationmode='ISO-3',
                color=metric,
                hover_name='country',
                color_continuous_scale=color_scale,
                title=title
            )
            
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, b=0, t=30),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ===== STAY PATTERNS PAGE =====
        elif page == "Stay Patterns":
            st.markdown('<div class="section-header">🛏️ Stay Patterns & Preferences</div>', unsafe_allow_html=True)
            
            # Metrics
            avg_stay = filtered_df['total_nights'].mean()
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Average Stay Duration", f"{avg_stay:.2f} nights")
            col2.metric("Average Weekend Nights", f"{filtered_df['stays_in_weekend_nights'].mean():.2f}")
            col3.metric("Average Weekday Nights", f"{filtered_df['stays_in_week_nights'].mean():.2f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Stay Duration Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(filtered_df['total_nights'], bins=20, kde=True, ax=ax)
                plt.title("Stay Duration Distribution")
                plt.xlabel("Total Nights")
                plt.ylabel("Number of Bookings")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Weekend vs Weekday Stay</div>', unsafe_allow_html=True)
                df_melted = filtered_df.melt(id_vars='hotel', value_vars=['stays_in_weekend_nights', 'stays_in_week_nights'],
                                   var_name='stay_type', value_name='nights')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=df_melted, x='stay_type', y='nights', ax=ax)
                plt.title("Weekend vs Weekday Stay Duration")
                plt.ylabel("Nights")
                plt.xlabel("Stay Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Room Type Preferences</div>', unsafe_allow_html=True)
            room_type_ct = pd.crosstab(filtered_df['reserved_room_type'], filtered_df['assigned_room_type'])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(room_type_ct, annot=True, fmt="d", cmap="Purples", ax=ax)
            plt.title("Reserved vs Assigned Room Types")
            plt.tight_layout()
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Impact of Room Changes</div>', unsafe_allow_html=True)
                room_change_stats = filtered_df.groupby('booking_changes')[['is_canceled', 'revenue']].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.figure(figsize=(12, 5))
                sns.lineplot(data=room_change_stats, x='booking_changes', y='is_canceled', label="Cancellation Rate")
                sns.lineplot(data=room_change_stats, x='booking_changes', y='revenue', label="Average Revenue")
                plt.title("Impact of Room Changes on Cancellation & Revenue")
                plt.xlabel("Booking Changes")
                plt.ylabel("Rate / Revenue")
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Seasonal Booking Trends</div>', unsafe_allow_html=True)
                monthly_bookings = filtered_df.groupby('arrival_month_name').size().reindex([
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'
                ])
                
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_bookings.plot(kind='bar', color='teal', ax=ax)
                plt.title("Bookings by Month")
                plt.ylabel("Number of Bookings")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Waiting List Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(filtered_df['days_in_waiting_list'], bins=20, kde=True, ax=ax)
                plt.title("Days in Waiting List Distribution")
                plt.xlabel("Days in Waiting List")
                plt.ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                waitlist_status = filtered_df.groupby('days_in_waiting_list')['is_canceled'].mean().reset_index()
                waitlist_status = waitlist_status[waitlist_status['days_in_waiting_list'] <= 30]  # Limit to 30 days for better visualization
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=waitlist_status, x='days_in_waiting_list', y='is_canceled', ax=ax)
                plt.title("Waitlist Days vs Cancellation Rate")
                plt.ylabel("Cancellation Rate")
                plt.xlabel("Days in Waiting List")
                plt.tight_layout()
                st.pyplot(fig)
        
        # ===== CUSTOMER SEGMENTATION PAGE =====
        elif page == "Customer Segmentation":
            st.markdown('<div class="section-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Bookings by Customer Type</div>', unsafe_allow_html=True)
                cust_type_counts = filtered_df['customer_type'].value_counts().reset_index()
                cust_type_counts.columns = ['customer_type', 'count']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=cust_type_counts, x='customer_type', y='count', ax=ax)
                plt.title("Bookings by Customer Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">ADR by Customer Type</div>', unsafe_allow_html=True)
                adr_cust = filtered_df.groupby('customer_type')['adr'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=adr_cust, x='customer_type', y='adr', ax=ax)
                plt.title("Average ADR by Customer Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Special Requests by Customer Type</div>', unsafe_allow_html=True)
                req_cust = filtered_df.groupby('customer_type')['total_of_special_requests'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=req_cust, x='customer_type', y='total_of_special_requests', ax=ax)
                plt.title("Average Special Requests by Customer Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="subsection-header">Lead Time by Customer Type</div>', unsafe_allow_html=True)
                lead_cust = filtered_df.groupby('customer_type')['lead_time'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=lead_cust, x='customer_type', y='lead_time', ax=ax)
                plt.title("Average Lead Time by Customer Type")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Market Segment Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = filtered_df['market_segment'].value_counts().reset_index()
                segment_counts.columns = ['market_segment', 'count']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=segment_counts, x='market_segment', y='count', ax=ax)
                plt.title("Bookings by Market Segment")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                segment_revenue = filtered_df.groupby('market_segment')['revenue'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=segment_revenue, x='market_segment', y='revenue', ax=ax)
                plt.title("Average Revenue by Market Segment")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('<div class="subsection-header">Distribution Channel Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                channel_counts = filtered_df['distribution_channel'].value_counts().reset_index()
                channel_counts.columns = ['distribution_channel', 'count']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=channel_counts, x='distribution_channel', y='count', ax=ax)
                plt.title("Bookings by Distribution Channel")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                channel_revenue = filtered_df.groupby('distribution_channel')['revenue'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=channel_revenue, x='distribution_channel', y='revenue', ax=ax)
                plt.title("Average Revenue by Distribution Channel")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Agent/Company Analysis
            st.markdown('<div class="subsection-header">Agent & Company Analysis</div>', unsafe_allow_html=True)
            
            top_agents = filtered_df['agent'].value_counts().head(10).reset_index()
            top_agents.columns = ['agent', 'booking_count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=top_agents, x='agent', y='booking_count', ax=ax)
                plt.title("Top 10 Booking Agents")
                plt.xlabel("Agent ID")
                plt.ylabel("Number of Bookings")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                agent_stats = filtered_df.groupby('agent').agg({
                    'revenue': 'sum',
                    'is_canceled': 'mean'
                }).reset_index()
                agent_stats['is_canceled'] *= 100
                top_agents_data = agent_stats.sort_values('revenue', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=top_agents_data, x='agent', y='revenue', ax=ax)
                plt.title("Top 10 Agents by Revenue")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        # ===== ADVANCED INSIGHTS PAGE =====
        elif page == "Advanced Insights":
            st.markdown('<div class="section-header">🔍 Advanced Insights</div>', unsafe_allow_html=True)
            
            # Correlation Analysis
            st.markdown('<div class="subsection-header">Correlation Matrix</div>', unsafe_allow_html=True)
            
            # Select only numeric columns for correlation
            numeric_cols = ['lead_time', 'adr', 'booking_changes', 'total_of_special_requests', 
                            'total_nights', 'previous_cancellations', 'previous_bookings_not_canceled', 
                            'days_in_waiting_list', 'is_canceled', 'revenue']
            
            # Filter out only columns that exist in the dataframe
            available_cols = [col for col in numeric_cols if col in filtered_df.columns]
            corr_df = filtered_df[available_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature Importance for Cancellation Prediction
            st.markdown('<div class="subsection-header">Feature Importance for Cancellation Prediction</div>', unsafe_allow_html=True)
            
            features = ['lead_time', 'adr', 'booking_changes', 'total_of_special_requests', 
                        'total_nights', 'previous_cancellations', 'previous_bookings_not_canceled', 
                        'days_in_waiting_list']
            
            # Filter out only features that exist in the dataframe
            available_features = [col for col in features if col in filtered_df.columns]
            
            if len(available_features) > 0 and 'is_canceled' in filtered_df.columns:
                df_ml = filtered_df[available_features + ['is_canceled']].dropna()
                
                if not df_ml.empty:
                    X = df_ml[available_features]
                    y = df_ml['is_canceled']
                    
                    # Check for sufficient samples
                    if len(df_ml) > 50:  # Require at least 50 samples for modeling
                        try:
                            model = RandomForestClassifier(random_state=42, n_estimators=50)
                            model.fit(X, y)
                            
                            feature_imp = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax)
                            plt.title("Feature Importance for Cancellation Prediction")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.markdown("""
                            This chart shows the relative importance of different features in predicting cancellations. 
                            Higher values indicate more influential factors in determining whether a booking will be canceled.
                            """)
                        except Exception as e:
                            st.warning(f"Could not build prediction model: {e}")
                    else:
                        st.warning("Not enough data for cancellation prediction model. Try expanding your filter criteria.")
                else:
                    st.warning("Insufficient data for modeling after filtering. Try adjusting your filters.")
            else:
                st.warning("Required columns for cancellation prediction model are not available in the dataset.")
            
            # Price Sensitivity Analysis
            st.markdown('<div class="subsection-header">Price Sensitivity Analysis</div>', unsafe_allow_html=True)
            
            if 'adr' in filtered_df.columns and 'is_canceled' in filtered_df.columns:
                # Create ADR bins
                filtered_df['adr_bin'] = pd.qcut(filtered_df['adr'], q=10, labels=False)
                adr_cancel = filtered_df.groupby('adr_bin').agg({
                    'adr': 'mean',
                    'is_canceled': 'mean'
                }).reset_index()
                adr_cancel['is_canceled'] *= 100
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=adr_cancel, x='adr', y='is_canceled', marker='o', ax=ax)
                plt.title("Price Sensitivity vs Cancellation Rate")
                plt.xlabel("Average Daily Rate (€)")
                plt.ylabel("Cancellation Rate (%)")
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                This chart shows how cancellation rates change with different pricing levels,
                helping identify optimal price points to minimize cancellations.
                """)
            
            # Length of Stay vs Revenue
            st.markdown('<div class="subsection-header">Length of Stay vs Revenue Analysis</div>', unsafe_allow_html=True)
            
            if 'total_nights' in filtered_df.columns and 'revenue' in filtered_df.columns:
                stay_revenue = filtered_df.groupby('total_nights')['revenue'].mean().reset_index()
                # Limit to reasonable stay length for visualization
                stay_revenue = stay_revenue[stay_revenue['total_nights'] <= 15]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=stay_revenue, x='total_nights', y='revenue', marker='o', ax=ax)
                plt.title("Average Revenue by Length of Stay")
                plt.xlabel("Total Nights")
                plt.ylabel("Average Revenue (€)")
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                This analysis shows how revenue changes with different lengths of stay,
                helping identify the most profitable stay durations to promote.
                """)
            
            # Time Between Booking and Arrival vs Revenue
            st.markdown('<div class="subsection-header">Lead Time vs Revenue</div>', unsafe_allow_html=True)
            
            if 'lead_time' in filtered_df.columns and 'revenue' in filtered_df.columns:
                # Create lead time bins (0-7, 8-14, 15-30, 31-60, 61-90, 91-180, 181+)
                lead_time_labels = ['0-7 days', '8-14 days', '15-30 days', '31-60 days', 
                                   '61-90 days', '91-180 days', '180+ days']
                
                filtered_df['lead_time_bin'] = pd.cut(
                    filtered_df['lead_time'], 
                    bins=[0, 7, 14, 30, 60, 90, 180, float('inf')],
                    labels=lead_time_labels
                )
                
                lead_revenue = filtered_df.groupby('lead_time_bin').agg({
                    'revenue': 'mean',
                    'is_canceled': 'mean'
                }).reset_index()
                lead_revenue['is_canceled'] *= 100
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create twin axis
                ax2 = ax.twinx()
                
                # Plot revenue bars
                sns.barplot(data=lead_revenue, x='lead_time_bin', y='revenue', color='blue', alpha=0.6, ax=ax)
                
                # Plot cancellation line
                sns.lineplot(data=lead_revenue, x=lead_revenue.index, y='is_canceled', marker='o', color='red', ax=ax2)
                
                # Labels and legend
                ax.set_xlabel('Lead Time')
                ax.set_ylabel('Average Revenue (€)', color='blue')
                ax2.set_ylabel('Cancellation Rate (%)', color='red')
                ax.tick_params(axis='y', colors='blue')
                ax2.tick_params(axis='y', colors='red')
                plt.title('Average Revenue and Cancellation Rate by Lead Time')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                This visualization shows both average revenue and cancellation rates by booking lead time.
                This can help identify optimal booking windows that balance revenue with lower cancellation risk.
                """)
        else:
            st.error("Please upload the hotel bookings dataset or adjust the file path in the code.")
            st.markdown("""
            ## How to use this dashboard
            
            1. Make sure the hotel bookings dataset CSV file is available at the path specified in the code 
            or update the path in the `load_data()` function.
            2. The dashboard will automatically load and visualize the data once the file is available.
            3. Use the sidebar filters to analyze specific segments of your hotel data.
            4. Navigate through different pages of the dashboard using the page selector in the sidebar.
            """)

# Add footer
st.markdown("---")
st.markdown("Hotel Analytics Dashboard | Created with Streamlit")