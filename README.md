# Airline Booking Market Demand Dashboard

A Streamlit web app to analyze airline booking market demand (filters, KPIs, charts, rolling-mean forecast).
Designed as a test task for hostel group market insights.

## Features
- Loads local CSV (`airline_bookings.csv`) or creates a synthetic dataset if missing.
- Filters: Airline, Origin, Destination, Cabin.
- KPIs: Total bookings, Total revenue, Average price, Unique routes.
- Charts: Top routes, Average price trend, Bookings by month, Lead-time distribution, Cabin breakdown.
- Simple rolling-mean baseline forecast (configurable).
- Optional live signals: OpenSky recent arrivals (best-effort) & BITRE CSV discovery (best-effort).
- Optional AI summary using OpenAI (set `OPENAI_API_KEY`).

## Setup (local)
1. Create & activate venv:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate
     ```
   

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
