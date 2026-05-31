import streamlit as st
import requests

@st.cache_data(ttl=3600)
def get_usd_to_inr():
    """
    Fetches the live USD to INR exchange rate from open.er-api.com.
    Result is cached for 1 hour to avoid hitting the API on every render.
    Returns (rate, last_updated_string).
    """
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        data = response.json()
        rate = float(data['rates']['INR'])
        updated = data.get('time_last_update_utc', 'Unknown')
        return rate, updated
    except Exception:
        # Fallback to a reasonable approximate rate if the API is unreachable
        return 84.0, "Fallback rate (live fetch failed)"


def format_inr(usd_price, rate):
    """
    Converts a USD price value to a formatted INR string.
    Returns 'Price unavailable' if the value is zero, negative, or non-numeric.
    """
    try:
        val = float(str(usd_price).replace('$', '').replace(',', '').strip())
        if val <= 0:
            return "Price unavailable"
        inr = val * rate
        if inr >= 1_00_000:
            return f"\u20b9{inr:,.0f}"
        return f"\u20b9{inr:,.0f}"
    except Exception:
        return "Price unavailable"
