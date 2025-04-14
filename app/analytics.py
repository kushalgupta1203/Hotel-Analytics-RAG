import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class HotelAnalytics:
    def __init__(self, csv_path):
        """Initialize the HotelAnalytics class with data from CSV file"""
        self.csv_path = csv_path
        self.df = self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the hotel bookings data"""
        df = pd.read_csv(self.csv_path)
        
        # Data preprocessing
        # Convert dates if date columns are available as strings
        if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns and 'arrival_date_day_of_month' in df.columns:
            # Map month names to numbers if necessary
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            
            # Check if month is string or already numeric
            if df['arrival_date_month'].iloc[0] in month_map:
                df['arrival_date_month'] = df['arrival_date_month'].map(month_map)
            
            # Create datetime column
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' + 
                df['arrival_date_month'].astype(str) + '-' + 
                df['arrival_date_day_of_month'].astype(str)
            )
        
        # Calculate derived metrics
        df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        
        # Add month and year columns for easier analysis
        if 'arrival_date' in df.columns:
            df['month'] = df['arrival_date'].dt.month
            df['year'] = df['arrival_date'].dt.year
            df['month_year'] = df['arrival_date'].dt.strftime('%Y-%m')
            df['quarter'] = df['arrival_date'].dt.quarter
            df['arrival_month_name'] = df['arrival_date'].dt.month_name()
        
        # Handle missing values in important columns
        df['country'] = df['country'].fillna("Unknown")
        df['agent'] = df['agent'].fillna(0)
        
        # Ensure revenue column exists or calculate it
        if 'revenue' not in df.columns and 'adr' in df.columns:
            df['revenue'] = df['adr'] * df['total_nights']
        
        # Calculate revenue per guest
        if 'revenue' in df.columns:
            df['revenue_per_booking'] = df['revenue']
            df['revenue_per_guest'] = df['revenue'] / df['total_guests'].replace(0, 1)  # avoid div/0
        
        return df
    
    def filter_data(self, start_date=None, end_date=None, hotel_types=None, customer_types=None):
        """Filter data based on provided criteria"""
        filtered_df = self.df.copy()
        
        # Date filtering
        if start_date and end_date and 'arrival_date' in filtered_df.columns:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[(filtered_df['arrival_date'] >= start_date) & 
                                      (filtered_df['arrival_date'] <= end_date)]
        
        # Hotel type filtering
        if hotel_types and 'hotel' in filtered_df.columns:
            if isinstance(hotel_types, str):
                hotel_types = [hotel_types]
            filtered_df = filtered_df[filtered_df['hotel'].isin(hotel_types)]
        
        # Customer type filtering
        if customer_types and 'customer_type' in filtered_df.columns:
            if isinstance(customer_types, str):
                customer_types = [customer_types]
            filtered_df = filtered_df[filtered_df['customer_type'].isin(customer_types)]
        
        return filtered_df
    
    def get_overview_metrics(self, filtered_df=None):
        """Calculate overview metrics"""
        if filtered_df is None:
            filtered_df = self.df
        
        metrics = {
            "total_bookings": len(filtered_df),
            "cancellation_rate": filtered_df['is_canceled'].mean() * 100,
            "total_revenue": filtered_df['revenue'].sum() if 'revenue' in filtered_df.columns else None,
            "average_daily_rate": filtered_df['adr'].mean() if 'adr' in filtered_df.columns else None,
            "average_lead_time": filtered_df['lead_time'].mean() if 'lead_time' in filtered_df.columns else None
        }
        
        # Add hotel type distribution if available
        if 'hotel' in filtered_df.columns:
            metrics["hotel_type_distribution"] = filtered_df['hotel'].value_counts().to_dict()
        
        # Add reservation status distribution if available
        if 'reservation_status' in filtered_df.columns:
            metrics["reservation_status_distribution"] = filtered_df['reservation_status'].value_counts().to_dict()
        
        return metrics
    
    def get_revenue_analysis(self, filtered_df=None, time_period="monthly"):
        """Analyze revenue data by different time periods"""
        if filtered_df is None:
            filtered_df = self.df
        
        if 'revenue' not in filtered_df.columns:
            return {"error": "Revenue data not available"}
        
        analysis = {
            "total_revenue": filtered_df['revenue'].sum(),
            "average_revenue_per_booking": filtered_df['revenue_per_booking'].mean() if 'revenue_per_booking' in filtered_df.columns else None,
            "average_revenue_per_guest": filtered_df['revenue_per_guest'].mean() if 'revenue_per_guest' in filtered_df.columns else None
        }
        
        # Revenue by time period
        if time_period == "monthly" and 'month_year' in filtered_df.columns:
            monthly_revenue = filtered_df.groupby('month_year')['revenue'].sum().to_dict()
            analysis["monthly_revenue"] = monthly_revenue
        elif time_period == "quarterly" and 'quarter' in filtered_df.columns:
            quarterly_revenue = filtered_df.groupby(['year', 'quarter'])['revenue'].sum().to_dict()
            analysis["quarterly_revenue"] = quarterly_revenue
        elif time_period == "yearly" and 'year' in filtered_df.columns:
            yearly_revenue = filtered_df.groupby('year')['revenue'].sum().to_dict()
            analysis["yearly_revenue"] = yearly_revenue
        
        # Revenue by hotel type
        if 'hotel' in filtered_df.columns:
            hotel_revenue = filtered_df.groupby('hotel')['revenue'].sum().to_dict()
            analysis["revenue_by_hotel_type"] = hotel_revenue
        
        # ADR trend
        if 'adr' in filtered_df.columns and 'month_year' in filtered_df.columns:
            adr_monthly = filtered_df.groupby('month_year')['adr'].mean().to_dict()
            analysis["adr_monthly"] = adr_monthly
        
        return analysis
    
    def get_cancellation_analysis(self, filtered_df=None):
        """Analyze cancellation patterns"""
        if filtered_df is None:
            filtered_df = self.df
        
        analysis = {
            "overall_cancellation_rate": filtered_df['is_canceled'].mean() * 100,
            "total_cancellations": filtered_df['is_canceled'].sum(),
        }
        
        # Average lead time for canceled bookings
        if 'lead_time' in filtered_df.columns:
            analysis["avg_lead_time_canceled"] = filtered_df[filtered_df['is_canceled']==1]['lead_time'].mean()
        
        # Cancellation rate by hotel type
        if 'hotel' in filtered_df.columns:
            hotel_cancel_rate = filtered_df.groupby('hotel')['is_canceled'].mean().multiply(100).to_dict()
            analysis["cancellation_rate_by_hotel"] = hotel_cancel_rate
        
        # Cancellation rate by customer type
        if 'customer_type' in filtered_df.columns:
            cust_cancel_rate = filtered_df.groupby('customer_type')['is_canceled'].mean().multiply(100).to_dict()
            analysis["cancellation_rate_by_customer_type"] = cust_cancel_rate
        
        # Cancellation rate by country (top 10)
        if 'country' in filtered_df.columns:
            country_bookings = filtered_df['country'].value_counts()
            valid_countries = country_bookings[country_bookings > 20].index.tolist()
            
            if valid_countries:
                country_data = filtered_df[filtered_df['country'].isin(valid_countries)]
                country_cancel_rate = country_data.groupby('country')['is_canceled'].mean().multiply(100)
                top_cancel_countries = country_cancel_rate.sort_values(ascending=False).head(10).to_dict()
                analysis["top_cancellation_rate_by_country"] = top_cancel_countries
        
        # Cancellation by deposit type
        if 'deposit_type' in filtered_df.columns:
            deposit_cancel_rate = filtered_df.groupby('deposit_type')['is_canceled'].mean().multiply(100).to_dict()
            analysis["cancellation_rate_by_deposit"] = deposit_cancel_rate
        
        return analysis

    def get_booking_by_month(self, filtered_df=None):
        """Get booking distribution by month"""
        if filtered_df is None:
            filtered_df = self.df
            
        if 'arrival_date_month' in filtered_df.columns:
            return filtered_df.groupby('arrival_date_month').size().to_dict()
        elif 'arrival_month_name' in filtered_df.columns:
            return filtered_df.groupby('arrival_month_name').size().to_dict()
        elif 'month' in filtered_df.columns:
            return filtered_df.groupby('month').size().to_dict()
        else:
            return {}
            
    def get_booking_by_week(self, filtered_df=None):
        """Get booking distribution by week number"""
        if filtered_df is None:
            filtered_df = self.df
            
        if 'arrival_date_week_number' in filtered_df.columns:
            return filtered_df.groupby('arrival_date_week_number').size().to_dict()
        elif 'arrival_date' in filtered_df.columns:
            # Calculate week numbers if not already present
            filtered_df['week_number'] = filtered_df['arrival_date'].dt.isocalendar().week
            return filtered_df.groupby('week_number').size().to_dict()
        else:
            return {}
    
    def get_geographic_analysis(self, filtered_df=None):
        """Analyze booking patterns by geography"""
        if filtered_df is None:
            filtered_df = self.df
        
        analysis = {}
        
        # Top countries by bookings
        if 'country' in filtered_df.columns:
            country_counts = filtered_df['country'].value_counts().head(10).to_dict()
            analysis["top_countries_by_bookings"] = country_counts
        
            # Average revenue by country
            if 'revenue' in filtered_df.columns:
                top_countries = list(country_counts.keys())
                country_revenue = filtered_df[filtered_df['country'].isin(top_countries)].groupby('country')['revenue'].mean().to_dict()
                analysis["avg_revenue_by_top_country"] = country_revenue
            
            # Cancellation rate by country
            top_countries_data = filtered_df[filtered_df['country'].isin(list(country_counts.keys()))]
            country_cancel = top_countries_data.groupby('country')['is_canceled'].mean().multiply(100).to_dict()
            analysis["cancellation_rate_by_top_country"] = country_cancel
        
        return analysis
    
    def get_customer_segmentation(self, filtered_df=None):
        """Analyze customer segments"""
        if filtered_df is None:
            filtered_df = self.df
        
        analysis = {}
        
        # Bookings by customer type
        if 'customer_type' in filtered_df.columns:
            cust_type_counts = filtered_df['customer_type'].value_counts().to_dict()
            analysis["bookings_by_customer_type"] = cust_type_counts
        
            # ADR by customer type
            if 'adr' in filtered_df.columns:
                adr_cust = filtered_df.groupby('customer_type')['adr'].mean().to_dict()
                analysis["adr_by_customer_type"] = adr_cust
            
            # Special requests by customer type
            if 'total_of_special_requests' in filtered_df.columns:
                req_cust = filtered_df.groupby('customer_type')['total_of_special_requests'].mean().to_dict()
                analysis["special_requests_by_customer_type"] = req_cust
            
            # Lead time by customer type
            if 'lead_time' in filtered_df.columns:
                lead_cust = filtered_df.groupby('customer_type')['lead_time'].mean().to_dict()
                analysis["lead_time_by_customer_type"] = lead_cust
        
        # Market segment analysis
        if 'market_segment' in filtered_df.columns:
            segment_counts = filtered_df['market_segment'].value_counts().to_dict()
            analysis["bookings_by_market_segment"] = segment_counts
            
            # Revenue by market segment
            if 'revenue' in filtered_df.columns:
                segment_revenue = filtered_df.groupby('market_segment')['revenue'].mean().to_dict()
                analysis["avg_revenue_by_market_segment"] = segment_revenue
        
        # Distribution channel analysis
        if 'distribution_channel' in filtered_df.columns:
            channel_counts = filtered_df['distribution_channel'].value_counts().to_dict()
            analysis["bookings_by_distribution_channel"] = channel_counts
            
            # Revenue by distribution channel
            if 'revenue' in filtered_df.columns:
                channel_revenue = filtered_df.groupby('distribution_channel')['revenue'].mean().to_dict()
                analysis["avg_revenue_by_distribution_channel"] = channel_revenue
        
        # Agent analysis
        if 'agent' in filtered_df.columns:
            top_agents = filtered_df['agent'].value_counts().head(10).to_dict()
            analysis["top_booking_agents"] = top_agents
            
            # Revenue by top agents
            if 'revenue' in filtered_df.columns:
                agent_stats = filtered_df.groupby('agent').agg({
                    'revenue': 'sum',
                    'is_canceled': 'mean'
                })
                agent_stats['is_canceled'] = agent_stats['is_canceled'] * 100
                top_agents_data = agent_stats.sort_values('revenue', ascending=False).head(10).to_dict()
                analysis["top_agents_by_revenue"] = top_agents_data
        
        return analysis
    
    def get_advanced_insights(self, filtered_df=None):
        """Generate advanced analytics insights"""
        if filtered_df is None:
            filtered_df = self.df
        
        insights = {}
        
        # Correlation analysis for numeric columns
        numeric_cols = ['lead_time', 'adr', 'booking_changes', 'total_of_special_requests', 
                        'total_nights', 'previous_cancellations', 'previous_bookings_not_canceled', 
                        'days_in_waiting_list', 'is_canceled']
        
        if 'revenue' in filtered_df.columns:
            numeric_cols.append('revenue')
        
        # Filter out only columns that exist in the dataframe
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        
        if available_cols:
            corr_df = filtered_df[available_cols].corr().round(2).to_dict()
            insights["correlation_matrix"] = corr_df
        
        # Price sensitivity analysis
        if 'adr' in filtered_df.columns and 'is_canceled' in filtered_df.columns:
            # Create ADR bins
            filtered_df['adr_bin'] = pd.qcut(filtered_df['adr'], q=10, labels=False, duplicates='drop')
            adr_cancel = filtered_df.groupby('adr_bin').agg({
                'adr': 'mean',
                'is_canceled': 'mean'
            })
            adr_cancel['is_canceled'] *= 100
            insights["price_sensitivity"] = adr_cancel.to_dict()
        
        # Length of stay vs Revenue
        if 'total_nights' in filtered_df.columns and 'revenue' in filtered_df.columns:
            stay_revenue = filtered_df.groupby('total_nights')['revenue'].mean()
            # Limit to reasonable stay length for visualization
            stay_revenue = stay_revenue[stay_revenue.index <= 15].to_dict()
            insights["revenue_by_stay_length"] = stay_revenue
        
        # Lead time analysis
        if 'lead_time' in filtered_df.columns:
            # Create lead time bins
            lead_time_labels = ['0-7 days', '8-14 days', '15-30 days', '31-60 days', 
                              '61-90 days', '91-180 days', '180+ days']
            
            filtered_df['lead_time_bin'] = pd.cut(
                filtered_df['lead_time'], 
                bins=[0, 7, 14, 30, 60, 90, 180, float('inf')],
                labels=lead_time_labels
            )
            
            lead_time_analysis = filtered_df.groupby('lead_time_bin').agg({
                'lead_time': 'count',
                'is_canceled': 'mean'
            })
            
            if 'revenue' in filtered_df.columns:
                lead_time_revenue = filtered_df.groupby('lead_time_bin')['revenue'].mean()
                lead_time_analysis['avg_revenue'] = lead_time_revenue
            
            lead_time_analysis['is_canceled'] *= 100
            insights["lead_time_analysis"] = lead_time_analysis.to_dict()
        
        return insights
    
    def plot_monthly_bookings(self, filtered_df=None, save_path=None):
        """Plot monthly booking trends"""
        if filtered_df is None:
            filtered_df = self.df
            
        if 'month_year' not in filtered_df.columns:
            return None
            
        plt.figure(figsize=(15, 6))
        monthly_bookings = filtered_df.groupby('month_year').size()
        
        sns.lineplot(x=monthly_bookings.index, y=monthly_bookings.values, marker='o')
        plt.title('Monthly Booking Trends')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of Bookings')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt
    
    def plot_cancellation_by_hotel_type(self, filtered_df=None, save_path=None):
        """Plot cancellation rates by hotel type"""
        if filtered_df is None:
            filtered_df = self.df
            
        if 'hotel' not in filtered_df.columns or 'is_canceled' not in filtered_df.columns:
            return None
            
        plt.figure(figsize=(10, 6))
        hotel_cancel_rate = filtered_df.groupby('hotel')['is_canceled'].mean().multiply(100)
        
        sns.barplot(x=hotel_cancel_rate.index, y=hotel_cancel_rate.values)
        plt.title('Cancellation Rate by Hotel Type')
        plt.xlabel('Hotel Type')
        plt.ylabel('Cancellation Rate (%)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt


# Keep the original get_analytics_report function for compatibility with main.py
def get_analytics_report(metrics):
    """Original function to get hotel analytics report for specified metrics
    
    Parameters:
    metrics (list): List of metrics to include in the report
    
    Returns:
    dict: Report containing requested metrics
    """
    # Initialize the analytics engine with the CSV path
    csv_path = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv"
    analytics = HotelAnalytics(csv_path)
    
    report = {}
    
    for metric in metrics:
        if metric == "cancellation_rate":
            cancellation_rate = analytics.df["is_canceled"].mean() * 100
            report["cancellation_rate"] = cancellation_rate
            
        elif metric == "average_revenue":
            if 'revenue' in analytics.df.columns:
                avg_revenue = analytics.df["revenue"].mean()
                report["average_revenue"] = avg_revenue
            elif 'adr' in analytics.df.columns:
                # Calculate from ADR if revenue not directly available
                avg_revenue = analytics.df["adr"].mean() * analytics.df["total_nights"].mean()
                report["average_revenue"] = avg_revenue
                
        elif metric == "lead_time":
            avg_lead_time = analytics.df["lead_time"].mean()
            report["lead_time"] = avg_lead_time
            
        elif metric == "booking_by_month":
            bookings_by_month = analytics.get_booking_by_month()
            report["booking_by_month"] = bookings_by_month
            
        elif metric == "booking_by_week":
            bookings_by_week = analytics.get_booking_by_week()
            report["booking_by_week"] = bookings_by_week
            
        elif metric == "top_countries":
            if 'country' in analytics.df.columns:
                top_countries = analytics.df['country'].value_counts().head(10).to_dict()
                report["top_countries"] = top_countries
                
        elif metric == "hotel_distribution":
            if 'hotel' in analytics.df.columns:
                hotel_distribution = analytics.df['hotel'].value_counts().to_dict()
                report["hotel_distribution"] = hotel_distribution
                
        elif metric == "customer_type_distribution":
            if 'customer_type' in analytics.df.columns:
                customer_distribution = analytics.df['customer_type'].value_counts().to_dict()
                report["customer_type_distribution"] = customer_distribution
                
        elif metric == "adr_by_month":
            if 'adr' in analytics.df.columns and 'arrival_month_name' in analytics.df.columns:
                adr_monthly = analytics.df.groupby('arrival_month_name')['adr'].mean().to_dict()
                report["adr_by_month"] = adr_monthly
                
        elif metric == "total_revenue":
            if 'revenue' in analytics.df.columns:
                total_revenue = analytics.df['revenue'].sum()
                report["total_revenue"] = total_revenue
            elif 'adr' in analytics.df.columns and 'total_nights' in analytics.df.columns:
                # Calculate if not directly available
                total_revenue = (analytics.df['adr'] * analytics.df['total_nights']).sum()
                report["total_revenue"] = total_revenue
                
        elif metric == "cancellation_by_deposit":
            if 'deposit_type' in analytics.df.columns and 'is_canceled' in analytics.df.columns:
                cancel_by_deposit = analytics.df.groupby('deposit_type')['is_canceled'].mean().multiply(100).to_dict()
                report["cancellation_by_deposit"] = cancel_by_deposit
                
        elif metric == "advanced_insights":
            # Get selected advanced insights
            advanced = analytics.get_advanced_insights()
            if "price_sensitivity" in advanced:
                report["price_sensitivity"] = advanced["price_sensitivity"]
            if "lead_time_analysis" in advanced:
                report["lead_time_cancellation"] = advanced["lead_time_analysis"]
    
    return report


# Example usage
if __name__ == "__main__":
    # Method 1: Using the HotelAnalytics class directly
    csv_path = r"D:\Projects\Hotel-Analytics-RAG\dataset\hotel_bookings_dataset.csv"
    hotel_analytics = HotelAnalytics(csv_path)
    
    # Generate comprehensive report
    comprehensive_report = hotel_analytics.get_overview_metrics()
    print("Comprehensive Report:")
    print(f"Total Bookings: {comprehensive_report['total_bookings']}")
    print(f"Cancellation Rate: {comprehensive_report['cancellation_rate']:.2f}%")
    
    # Method 2: Using the original get_analytics_report function
    metrics_to_analyze = ["cancellation_rate", "average_revenue", "lead_time", 
                         "booking_by_month", "booking_by_week"]
    
    report = get_analytics_report(metrics_to_analyze)
    print("\nOriginal get_analytics_report function:")
    for metric, value in report.items():
        if isinstance(value, dict):
            print(f"{metric}: {list(value.keys())[:3]}...")  # Show just first 3 keys
        else:
            print(f"{metric}: {value}")