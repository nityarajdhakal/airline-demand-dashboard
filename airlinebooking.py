# %% 
# Cell 1: Dataset creation
import pandas as pd
import numpy as np
import streamlit as st


np.random.seed(42)
n = 1000

airlines = ["Qantas", "Virgin Australia", "Jetstar", "Singapore Airlines", "Emirates"]
routes = [
    ("Sydney", "Melbourne"),
    ("Sydney", "Brisbane"),
    ("Melbourne", "Perth"),
    ("Brisbane", "Adelaide"),
    ("Sydney", "Singapore"),
    ("Melbourne", "Dubai"),
]
cabins = ["Economy", "Business", "First"]

# Choose random routes properly
chosen_routes = np.random.choice(len(routes), n)
origins = [routes[i][0] for i in chosen_routes]
destinations = [routes[i][1] for i in chosen_routes]

data = {
    "BookingID": np.arange(1, n + 1),
    "Airline": np.random.choice(airlines, n),
    "Origin": origins,
    "Destination": destinations,
    "Cabin": np.random.choice(cabins, n, p=[0.7, 0.25, 0.05]),
    "Price": np.random.randint(100, 2000, n),
    "BookingDate": pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
    "TravelDate": pd.to_datetime("2023-01-15") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
}

df = pd.DataFrame(data)
df.to_csv("airline_bookings.csv", index=False)

print("✅ Dataset created with", len(df), "rows")
print(df.head())



# %% 
# Cell 2: Load dataset
import pandas as pd

# Load dataset
df = pd.read_csv("airline_bookings.csv", parse_dates=["BookingDate","TravelDate"])

# Quick look
print("Dataset shape:", df.shape)
df.head()



# %%
# Cell 3: KPIs
# Total passengers booked
total_bookings = len(df)

# Total revenue
total_revenue = df["Price"].sum()

# Average ticket price
avg_price = df["Price"].mean()

print("Total Bookings:", total_bookings)
print("Total Revenue (AUD):", round(total_revenue, 2))
print("Average Price per Ticket (AUD):", round(avg_price, 2))



# %%
# Cell 4: Popular routes
# (groupby OD and plot)
# Count passengers per Origin-Destination pair
route_counts = df.groupby(["Origin","Destination"]).size().sort_values(ascending=False).head(10)
print("Top 10 Routes:\n", route_counts)

# Optional: plot
import matplotlib.pyplot as plt

route_counts.plot(kind="bar", figsize=(10,5), title="Top 10 Routes by Number of Bookings")
plt.ylabel("Number of Bookings")
plt.show()

# %%
# Cell 5: Price trends
# (line chart of average price per month)
# Average price by travel month
df['TravelMonth'] = df['TravelDate'].dt.to_period('M').dt.to_timestamp()
price_trend = df.groupby("TravelMonth")["Price"].mean()

price_trend.plot(figsize=(10,5), title="Average Ticket Price per Month")
plt.ylabel("Average Price (AUD)")
plt.show()

# %%
# Cell 6: High-demand periods
# (bookings per travel month)
# Bookings by travel month
bookings_per_month = df.groupby("TravelMonth").size()
bookings_per_month.plot(figsize=(10,5), title="Bookings per Month")
plt.ylabel("Number of Bookings")
plt.show()

# %% Lead-Time Analysis
df['LeadDays'] = (df['TravelDate'] - df['BookingDate']).dt.days

# Group by lead time buckets
bins = [0, 7, 14, 30, 60, 90, 120, 365]
labels = ["0-6","7-13","14-29","30-59","60-89","90-119","120+"]
df['LeadBucket'] = pd.cut(df['LeadDays'], bins=bins, labels=labels, right=False)

lead_dist = df.groupby('LeadBucket').size()
print("Booking Lead-Time Distribution:\n", lead_dist)

# Plot
import matplotlib.pyplot as plt
lead_dist.plot(kind='bar', figsize=(10,5), title='Bookings by Lead-Time Bucket')
plt.ylabel("Number of Bookings")
plt.show()


# %% Cabin Type Analysis
cabin_dist = df.groupby('Cabin').size()
print("Cabin Type Distribution:\n", cabin_dist)

# Plot
cabin_dist.plot(kind='bar', figsize=(6,4), title='Cabin Type Distribution')
plt.ylabel("Number of Bookings")
plt.show()


# %% Simple Forecast
monthly_bookings = df.groupby(df['TravelMonth']).size()

# Rolling mean for trend
rolling_window = 3
rolling_mean = monthly_bookings.rolling(window=rolling_window).mean()

# Forecast next 3 months using last rolling mean
last_val = rolling_mean.dropna().iloc[-1]
future_months = pd.date_range(start=monthly_bookings.index.max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
forecast = pd.Series([last_val]*3, index=future_months)

# Plot
plt.figure(figsize=(10,5))
plt.plot(monthly_bookings.index, monthly_bookings.values, label='Actual Bookings')
plt.plot(rolling_mean.index, rolling_mean.values, label=f'{rolling_window}-month Rolling Mean')
plt.plot(forecast.index, forecast.values, label='3-Month Forecast')
plt.title("Monthly Bookings with Rolling Mean & Forecast")
plt.ylabel("Number of Bookings")
plt.legend()
plt.show()


# %% Interactive Dashboard

from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("airline_bookings.csv", parse_dates=["BookingDate","TravelDate"])
df['TravelMonth'] = df['TravelDate'].dt.to_period('M').dt.to_timestamp()
df['LeadDays'] = (df['TravelDate'] - df['BookingDate']).dt.days
bins = [0, 7, 14, 30, 60, 90, 120, 365]
labels = ["0-6","7-13","14-29","30-59","60-89","90-119","120+"]
df['LeadBucket'] = pd.cut(df['LeadDays'], bins=bins, labels=labels, right=False)

# Widget dropdowns
airline_dropdown = st.selectbox(
    options=['All'] + sorted(df['Airline'].unique().tolist()),
    description='Airline:'
)
origin_dropdown = st.selectbox(
    options=['All'] + sorted(df['Origin'].unique().tolist()),
    description='Origin:'
)
destination_dropdown = st.selectbox(
    options=['All'] + sorted(df['Destination'].unique().tolist()),
    description='Destination:'
)
cabin_dropdown = st.selectbox(
    options=['All'] + sorted(df['Cabin'].unique().tolist()),
    description='Cabin:'
)

# Output area
output = st.Output()

# Function to update charts
def update_dashboard(change=None):
    with output:
        clear_output(wait=True)
        
        # Apply filters
        filtered = df.copy()
        if airline_dropdown.value != 'All':
            filtered = filtered[filtered['Airline']==airline_dropdown.value]
        if origin_dropdown.value != 'All':
            filtered = filtered[filtered['Origin']==origin_dropdown.value]
        if destination_dropdown.value != 'All':
            filtered = filtered[filtered['Destination']==destination_dropdown.value]
        if cabin_dropdown.value != 'All':
            filtered = filtered[filtered['Cabin']==cabin_dropdown.value]
        
        # KPIs
        total_bookings = len(filtered)
        total_revenue = filtered['Price'].sum()
        avg_price = filtered['Price'].mean() if total_bookings>0 else 0
        
        print(f"Total Bookings: {total_bookings}")
        print(f"Total Revenue (AUD): {total_revenue:.2f}")
        print(f"Average Price per Ticket (AUD): {avg_price:.2f}")
        
        if total_bookings==0:
            print("No data for this filter combination.")
            return
        
        # Top Routes
        top_routes = filtered.groupby(['Origin','Destination']).size().sort_values(ascending=False).head(5)
        top_routes.plot(kind='bar', figsize=(8,4), title='Top Routes')
        plt.ylabel("Number of Bookings")
        plt.show()
        
        # Monthly Price Trend
        price_trend = filtered.groupby("TravelMonth")["Price"].mean()
        price_trend.plot(figsize=(8,4), title="Average Price per Month")
        plt.ylabel("Average Price (AUD)")
        plt.show()
        
        # Lead-Time Distribution
        lead_dist = filtered.groupby('LeadBucket').size()
        lead_dist.plot(kind='bar', figsize=(8,4), title="Bookings by Lead-Time")
        plt.ylabel("Number of Bookings")
        plt.show()

# Observe widget changes
for w in [airline_dropdown, origin_dropdown, destination_dropdown, cabin_dropdown]:
    w.observe(update_dashboard, names='value')

# Display widgets and output
display(st.HBox([airline_dropdown, origin_dropdown, destination_dropdown, cabin_dropdown]))
display(output)

# Initial dashboard
update_dashboard()
