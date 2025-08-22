# app.py
"""
Airline Booking Market Demand Dashboard (Streamlit)
- Filters, KPIs, charts, rolling-mean forecast
- Optional live signals: OpenSky (recent arrivals), BITRE CSV discovery
- Optional AI summary using OpenAI (set OPENAI_API_KEY)
"""

import os
import time
import math
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(page_title="Airline Booking Demand", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Utilities & caching
# -----------------------------
@st.cache_data(ttl=60 * 60)
def load_or_create_csv(csv_path: str = "airline_bookings.csv", n_synth: int = 2000) -> pd.DataFrame:
    """Load CSV if exists, otherwise create a synthetic dataset and save it."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=["BookingDate", "TravelDate"])
            return df
        except Exception as e:
            st.warning(f"Failed to read {csv_path}: {e}. Will generate synthetic data.")
    # generate synthetic data
    np.random.seed(42)
    n = n_synth
    airlines = ["Qantas", "Virgin Australia", "Jetstar", "Singapore Airlines", "Emirates"]
    routes = [
        ("Sydney", "Melbourne"),
        ("Sydney", "Brisbane"),
        ("Melbourne", "Perth"),
        ("Brisbane", "Adelaide"),
        ("Sydney", "Singapore"),
        ("Melbourne", "Dubai"),
        ("Sydney", "Cairns"),
        ("Brisbane", "Gold Coast"),
    ]
    cabins = ["Economy", "Business", "First"]
    chosen_routes = np.random.choice(len(routes), n)
    origins = [routes[i][0] for i in chosen_routes]
    destinations = [routes[i][1] for i in chosen_routes]

    data = {
        "BookingID": np.arange(1, n + 1),
        "Airline": np.random.choice(airlines, n),
        "Origin": origins,
        "Destination": destinations,
        "Cabin": np.random.choice(cabins, n, p=[0.78, 0.18, 0.04]),
        "Price": np.random.randint(100, 2000, n),
        "BookingDate": pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
        "TravelDate": pd.to_datetime("2023-01-15") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return df

def to_csv_bytes(df_input: pd.DataFrame) -> bytes:
    return df_input.to_csv(index=False).encode("utf-8")

# -----------------------------
# Optional: OpenSky arrivals helper
# -----------------------------
@st.cache_data(ttl=15 * 60)
def fetch_opensky_arrival_count(icao: str, days: int = 3) -> int | None:
    """Return number of arrival records in last `days` days via OpenSky public API """
    try:
        end = int(time.time())
        start = end - days * 24 * 3600
        url = f"https://opensky-network.org/api/flights/arrival?airport={icao}&begin={start}&end={end}"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            try:
                j = r.json()
                return len(j)
            except Exception:
                return None
        return None
    except Exception:
        return None

# Small IATA -> ICAO map for AU airports (extend as needed)
IATA_TO_ICAO = {
    "Sydney": "YSSY",
    "Melbourne": "YMML",
    "Brisbane": "YBBN",
    "Perth": "YPPH",
    "Adelaide": "YPAD",
    "Cairns": "YBCS",
    "Gold Coast": "YBCG",
}

# -----------------------------
# Optional: BITRE CSV discovery 
# -----------------------------
@st.cache_data(ttl=6 * 60 * 60)
def discover_bitre_csvs() -> list:
    """Try to discover some BITRE CSV links from public pages."""
    roots = [
        "https://www.bitre.gov.au/publications/ongoing/international_airline_activity-monthly_publications",
        "https://www.bitre.gov.au/publications/ongoing/airport_traffic_data",
    ]
    found = []
    for root in roots:
        try:
            r = requests.get(root, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(" ", strip=True).lower()
                if href.lower().endswith(".csv") and ("airport" in text or "passenger" in text or "route" in text or "city" in text):
                    url = href if href.startswith("http") else requests.compat.urljoin(root, href)
                    found.append({"title": a.get_text(strip=True), "url": url})
        except Exception:
            continue
    return found

# -----------------------------
# Load dataset & preprocess
# -----------------------------
df = load_or_create_csv()

# Basic derived columns
df["TravelMonth"] = pd.to_datetime(df["TravelDate"]).dt.to_period("M").dt.to_timestamp()
df["LeadDays"] = (pd.to_datetime(df["TravelDate"]) - pd.to_datetime(df["BookingDate"])).dt.days
lead_bins = [0, 7, 14, 30, 60, 90, 120, 365]
lead_labels = ["0-6", "7-13", "14-29", "30-59", "60-89", "90-119", "120+"]
df["LeadBucket"] = pd.cut(df["LeadDays"], bins=lead_bins, labels=lead_labels, right=False)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Filters & Controls")
st.sidebar.caption("Filter data; small caption text shows guidance.")

airline = st.sidebar.selectbox("Airline", ["All"] + sorted(df["Airline"].unique().tolist()))
origin = st.sidebar.selectbox("Origin", ["All"] + sorted(df["Origin"].unique().tolist()))
destination = st.sidebar.selectbox("Destination", ["All"] + sorted(df["Destination"].unique().tolist()))
cabin = st.sidebar.selectbox("Cabin", ["All"] + sorted(df["Cabin"].unique().tolist()))

st.sidebar.markdown("---")
st.sidebar.header("Forecast settings")
rolling_window = st.sidebar.slider("Rolling window (months)", min_value=3, max_value=12, value=3, step=1)
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3, step=1)
st.sidebar.markdown("---")

# Optional API tools
st.sidebar.header("Optional live data")
use_opensky = st.sidebar.checkbox("Show OpenSky recent arrivals", value=False)
show_bitre = st.sidebar.checkbox("Discover BITRE CSVs", value=False)

# -----------------------------
# Apply filters
# -----------------------------
filtered = df.copy()
if airline != "All":
    filtered = filtered[filtered["Airline"] == airline]
if origin != "All":
    filtered = filtered[filtered["Origin"] == origin]
if destination != "All":
    filtered = filtered[filtered["Destination"] == destination]
if cabin != "All":
    filtered = filtered[filtered["Cabin"] == cabin]

# Header & KPIs
st.title("Airline Booking Market Demand Dashboard")
st.caption("Interactive dashboard — use sidebar filters. Data is local synthetic CSV by default.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bookings", f"{len(filtered):,}")
col2.metric("Total Revenue (AUD)", f"{filtered['Price'].sum():,.0f}")
col3.metric("Average Price (AUD)", f"{filtered['Price'].mean():,.2f}" if len(filtered) else "—")
col4.metric("Unique Routes", f"{filtered[['Origin','Destination']].drop_duplicates().shape[0]}")

st.divider()

# -----------------------------
# Optional live signals
# -----------------------------
if use_opensky:
    # If a single origin is selected, show arrivals for it
    chosen = origin if origin != "All" else None
    if chosen and chosen in IATA_TO_ICAO:
        with st.spinner(f"Fetching OpenSky arrivals for {chosen}..."):
            count = fetch_opensky_arrival_count(IATA_TO_ICAO[chosen], days=3)
            if count is None:
                st.sidebar.info("OpenSky: data unavailable or rate-limited.")
            else:
                st.sidebar.success(f"Arrivals in last 72h to {chosen}: {count}")
    else:
        st.sidebar.info("Select a specific origin (not 'All') for OpenSky arrivals.")

if show_bitre:
    with st.spinner("Searching BITRE pages for CSVs..."):
        bitre = discover_bitre_csvs()
        if bitre:
            st.sidebar.write("Discovered BITRE CSVs :")
            for b in bitre[:8]:
                st.sidebar.write(f"- {b['title']}")
        else:
            st.sidebar.info("No BITRE CSVs discovered (site structure may have changed or network blocked).")

# -----------------------------
# Top routes chart
# -----------------------------
st.subheader("Top Routes (by bookings)")
top_routes = filtered.groupby(["Origin","Destination"]).size().sort_values(ascending=False).head(12)
if top_routes.empty:
    st.info("No route data for the current filter.")
else:
    fig, ax = plt.subplots(figsize=(10,4))
    top_routes.plot(kind="bar", ax=ax)
    ax.set_ylabel("Bookings")
    ax.set_xlabel("Route")
    # nicer xlabels:
    ax.set_xticklabels([f"{o}-{d}" for (o,d) in top_routes.index], rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# Price trend chart
# -----------------------------
st.subheader("Average Price by Travel Month")
price_trend = filtered.groupby("TravelMonth")["Price"].mean().sort_index()
if price_trend.empty:
    st.info("No price data for current filters.")
else:
    fig, ax = plt.subplots(figsize=(10,4))
    price_trend.plot(ax=ax, marker="o")
    ax.set_ylabel("Average Price (AUD)")
    ax.set_xlabel("Travel Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# Bookings time series + forecast
# -----------------------------
st.subheader("Bookings by Travel Month & Forecast")
monthly_bookings = filtered.groupby("TravelMonth").size().sort_index()

if monthly_bookings.empty:
    st.info("No time-series data for current filters.")
else:
    roll = monthly_bookings.rolling(window=rolling_window, center=True, min_periods=max(1, rolling_window//2)).mean()
    last_val = roll.dropna().iloc[-1] if not roll.dropna().empty else monthly_bookings.iloc[-1]
    future_idx = pd.date_range(monthly_bookings.index.max() + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq="MS")
    forecast = pd.Series([last_val] * forecast_horizon, index=future_idx)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_bookings.index, monthly_bookings.values, label="Actual", marker="o")
    ax.plot(roll.index, roll.values, label=f"{rolling_window}-mo Rolling Mean", linestyle="--")
    ax.plot(forecast.index, forecast.values, label=f"Forecast ({forecast_horizon} mo)", marker="x")
    ax.set_ylabel("Bookings")
    ax.set_xlabel("Travel Month")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    fc_df = pd.DataFrame({"month": forecast.index.astype(str), "forecast_bookings": forecast.values.astype(int)})
    st.markdown("**Forecast (table)**")
    st.dataframe(fc_df, use_container_width=True)

# -----------------------------
# Lead-time & cabin breakdown
# -----------------------------
left, right = st.columns(2)
with left:
    st.subheader("Lead-Time Distribution")
    lead_dist = filtered.groupby("LeadBucket").size().reindex(lead_labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7,3))
    lead_dist.plot(kind="bar", ax=ax)
    ax.set_ylabel("Bookings")
    ax.set_xlabel("Lead bucket (days)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Cabin Distribution")
    cabin_dist = filtered["Cabin"].value_counts()
    fig, ax = plt.subplots(figsize=(6,3))
    cabin_dist.plot(kind="bar", ax=ax)
    ax.set_ylabel("Bookings")
    ax.set_xlabel("Cabin")
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# AI Insights (Optional - OpenAI)
# -----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY

        # Prepare a compact prompt
        top_routes_list = [f"{o}-{d} ({c})" for (o, d), c in top_routes.items()] if not top_routes.empty else []
        last_price_vals = price_trend.tail(3).tolist() if not price_trend.empty else []
        prompt = (
            "You are an airline market analyst. Write 4 short actionable bullet points (one line each) "
            "for hostel owners about the following dataset summary:\n\n"
            f"Total bookings: {len(filtered)}, total revenue: {int(filtered['Price'].sum()) if len(filtered) else 0}, "
            f"avg price: {filtered['Price'].mean():.1f if len(filtered) else 'N/A'}\n"
            f"Top routes: {', '.join(top_routes_list[:5])}\n"
            f"Recent avg price (last 3 months): {', '.join([str(int(x)) for x in last_price_vals]) if last_price_vals else 'N/A'}\n\n"
            "Keep it short and actionable."
        )
        # Call ChatCompletion
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.2,
        )
        summary_text = resp.choices[0].message.content.strip()
        st.subheader("AI Insights")
        st.write(summary_text)
    except Exception as e:
        st.sidebar.warning("OpenAI call failed: " + str(e))
else:
    st.sidebar.caption("Set OPENAI_API_KEY (env) to enable AI summaries.")


# -----------------------------
# Download filtered CSV
# -----------------------------
st.markdown("---")
st.download_button("Download filtered data (CSV)", data=to_csv_bytes(filtered), file_name="filtered_airline_bookings.csv", mime="text/csv")

st.caption("Built with Streamlit. Use OpenSky/BITRE hints for live signals; AI summary is optional. Replace forecast with ML model for production.")
