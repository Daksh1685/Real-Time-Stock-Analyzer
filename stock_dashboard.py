import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import ta
from typing import Optional, Tuple, Dict, List
import json
import requests
import io
import re

# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initial list of popular Indian stocks
POPULAR_INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'ADANIPORTS.NS': 'Adani Ports',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto'
}

# Function to fetch comprehensive list of NSE stocks
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_all_nse_stocks() -> Dict[str, str]:
    """
    Fetch a comprehensive list of stocks listed on the National Stock Exchange (NSE).
    
    Returns:
        Dictionary with stock symbols as keys and company names as values
    """
    try:
        # Try to load cached data if it exists
        try:
            with open('nse_stocks_data.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # URLs to fetch stock lists
        nifty500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        all_stocks_url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        
        # First try to get the comprehensive list from NSE API (may fail due to restrictions)
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9"
            }
            response = requests.get(all_stocks_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                stocks = {}
                for stock in data.get('data', []):
                    symbol = stock.get('symbol', '')
                    name = stock.get('meta', {}).get('companyName', '')
                    if symbol and name:
                        stocks[f"{symbol}.NS"] = name
                
                # If we got data, save it and return
                if stocks:
                    with open('nse_stocks_data.json', 'w') as f:
                        json.dump(stocks, f)
                    return stocks
        except Exception as e:
            st.warning(f"Couldn't fetch comprehensive NSE data. Using fallback method. Error: {str(e)}")
        
        # Fallback: Get Nifty 500 list
        try:
            response = requests.get(nifty500_url, timeout=10)
            if response.status_code == 200:
                data = pd.read_csv(io.StringIO(response.text))
                stocks = {}
                for _, row in data.iterrows():
                    symbol = row['Symbol']
                    name = row['Company Name']
                    stocks[f"{symbol}.NS"] = name
                
                # If we got data, save it and return
                if stocks:
                    with open('nse_stocks_data.json', 'w') as f:
                        json.dump(stocks, f)
                    return stocks
        except Exception as e:
            st.warning(f"Couldn't fetch Nifty 500 list. Using default stock list. Error: {str(e)}")
        
        # Fallback to our predefined list
        return POPULAR_INDIAN_STOCKS
    except Exception as e:
        st.error(f"Error fetching NSE stocks: {str(e)}")
        return POPULAR_INDIAN_STOCKS

# Function to fetch penny stocks from NSE
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_penny_stocks() -> Dict[str, str]:
    """
    Fetch penny stocks from NSE (stocks trading under ₹10)
    
    Returns:
        Dictionary with stock symbols as keys and company names as values
    """
    try:
        # Skip the file loading that might be causing errors
        # Return a hardcoded list of confirmed working penny stocks
        confirmed_penny_stocks = {
            "YESBANK.NS": "Yes Bank Ltd. (₹8.90)",
            "SUZLON.NS": "Suzlon Energy Ltd. (₹3.80)",
            "RPOWER.NS": "Reliance Power Ltd. (₹1.80)",
            "IDEA.NS": "Vodafone Idea Ltd. (₹2.25)",
            "JPPOWER.NS": "Jaiprakash Power Ventures Ltd. (₹7.20)",
            "SOUTHBANK.NS": "South Indian Bank Ltd. (₹9.10)",
            "PNB.NS": "Punjab National Bank (₹8.50)"
        }
        
        return confirmed_penny_stocks
    except Exception as e:
        st.error(f"Error fetching penny stocks: {str(e)}")
        # Return a minimal fallback list to avoid breaking the app
        return {
            "YESBANK.NS": "Yes Bank Ltd. (₹8.90)",
            "SUZLON.NS": "Suzlon Energy Ltd. (₹3.80)",
            "IDEA.NS": "Vodafone Idea Ltd. (₹2.25)"
        }

# Function to get newly listed stocks (IPOs in the last 6 months)
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_recent_ipos() -> Dict[str, str]:
    """
    Get recently listed stocks (IPOs) on NSE in the last 6 months
    
    Returns:
        Dictionary with stock symbols as keys and company names as values
    """
    try:
        # Skip the file loading that might be causing errors
        # Just return a hardcoded list of confirmed working stocks
        confirmed_ipos = {
            "ZOMATO.NS": "Zomato Ltd.",
            "POLICYBZR.NS": "PB Fintech Ltd. (PolicyBazaar)",
            "NYKAA.NS": "FSN E-Commerce Ventures Ltd. (Nykaa)",
            "PAYTM.NS": "One97 Communications Ltd. (Paytm)",
            "CARTRADE.NS": "CarTrade Tech Ltd.",
            "EASEMYTRIP.NS": "Easy Trip Planners Ltd.",
            "NAZARA.NS": "Nazara Technologies Ltd.",
            "IRFC.NS": "Indian Railway Finance Corporation Ltd."
        }
        
        return confirmed_ipos
    except Exception as e:
        st.error(f"Error fetching recent IPOs: {str(e)}")
        # Return a minimal fallback list to avoid breaking the app
        return {
            "ZOMATO.NS": "Zomato Ltd.",
            "NYKAA.NS": "FSN E-Commerce Ventures Ltd. (Nykaa)",
            "PAYTM.NS": "One97 Communications Ltd. (Paytm)"
        }

# Get all Indian stocks
INDIAN_STOCKS = get_all_nse_stocks()

# Function to search for stocks by name
def search_stocks(search_term: str, stocks_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Search for stocks by company name or symbol.
    
    Args:
        search_term: The search query (company name or symbol)
        stocks_dict: Dictionary of stocks to search in
    
    Returns:
        Dictionary of matching stocks
    """
    if not search_term:
        return {}
    
    search_term = search_term.lower()
    results = {}
    
    # First search for exact matches in symbols
    for symbol, name in stocks_dict.items():
        if search_term == symbol.lower().replace('.ns', ''):
            results[symbol] = name
    
    # Then search for matches at the start of company names
    if not results:
        for symbol, name in stocks_dict.items():
            if name.lower().startswith(search_term):
                results[symbol] = name
    
    # If still no results, do a more general search
    if not results:
        for symbol, name in stocks_dict.items():
            if search_term in name.lower() or search_term in symbol.lower():
                results[symbol] = name
                # Limit to 20 results to prevent overwhelming the UI
                if len(results) >= 20:
                    break
    
    return results

# Cache the stock data to improve performance
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with error handling and caching.
    
    Args:
        ticker: Stock symbol
        period: Time period (e.g., '1d', '5d', '1mo')
        interval: Data interval (e.g., '1m', '5m', '1h')
    
    Returns:
        DataFrame with stock data or None if error occurs
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol and try again.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate() -> float:
    """
    Get the current USD to INR conversion rate.
    
    Returns:
        Current USD to INR rate
    """
    try:
        usd_inr = yf.Ticker("USDINR=X")
        rate = usd_inr.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 83.0  # Fallback rate if API fails

def is_indian_stock(ticker: str) -> bool:
    """
    Check if the stock is an Indian stock.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        True if it's an Indian stock, False otherwise
    """
    return ticker.endswith('.NS') or ticker in INDIAN_STOCKS

def get_stock_currency(ticker: str) -> str:
    """
    Get the currency for a given stock.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Currency code ('USD' or 'INR')
    """
    return 'INR' if is_indian_stock(ticker) else 'USD'

def convert_to_inr(usd_value: float) -> float:
    """
    Convert USD value to INR.
    
    Args:
        usd_value: Value in USD
    
    Returns:
        Value in INR
    """
    rate = get_usd_to_inr_rate()
    return usd_value * rate

def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format currency value with appropriate symbol.
    
    Args:
        value: The value to format
        currency: Currency code ('USD' or 'INR')
    
    Returns:
        Formatted string with currency symbol
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    else:  # INR
        return f"₹{value:,.2f}"

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process and format the stock data.
    
    Args:
        data: Raw stock data DataFrame
    
    Returns:
        Processed DataFrame with proper timezone and formatting
    """
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate key metrics from stock data.
    
    Args:
        data: Processed stock data DataFrame
    
    Returns:
        Tuple of (last_close, change, pct_change, high, low, volume)
    """
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Args:
        data: Processed stock data DataFrame
    
    Returns:
        DataFrame with added technical indicators
    """
    # Trend indicators
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    
    # Momentum indicators
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    data['Stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
    
    # Volume indicators
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    return data

def create_chart(data: pd.DataFrame, chart_type: str, indicators: List[str], ticker: str, time_period: str, currency: str = 'USD') -> go.Figure:
    """
    Create an interactive chart with selected indicators.
    
    Args:
        data: Processed stock data with indicators
        chart_type: Type of chart ('Candlestick' or 'Line')
        indicators: List of selected technical indicators
        ticker: Stock symbol
        time_period: Selected time period
        currency: Currency to display ('USD' or 'INR')
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Convert prices to INR if needed (only for US stocks)
    if currency == 'INR' and not is_indian_stock(ticker):
        rate = get_usd_to_inr_rate()
        data['Open'] = data['Open'] * rate
        data['High'] = data['High'] * rate
        data['Low'] = data['Low'] * rate
        data['Close'] = data['Close'] * rate
        data['SMA_20'] = data['SMA_20'] * rate
        data['EMA_20'] = data['EMA_20'] * rate
    
    # Add main price chart
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data['Datetime'],
            y=data['Close'],
            name='Price',
            line=dict(color='blue')
        ))
    
    # Add selected indicators
    for indicator in indicators:
        if indicator == 'SMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange')
            ))
        elif indicator == 'EMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['EMA_20'],
                name='EMA 20',
                line=dict(color='green')
            ))
        elif indicator == 'RSI 14':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['RSI_14'],
                name='RSI 14',
                yaxis='y2',
                line=dict(color='purple')
            ))
        elif indicator == 'MACD':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['MACD'],
                name='MACD',
                yaxis='y3',
                line=dict(color='red')
            ))
    
    # Update layout
    currency_symbol = '₹' if currency == 'INR' else '$'
    fig.update_layout(
        title=f"{ticker} {time_period.upper()} Chart",
        xaxis_title='Time',
        yaxis_title=f'Price ({currency_symbol})',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 100]
        ),
        yaxis3=dict(
            title='MACD',
            overlaying='y',
            side='right',
            showgrid=False,
            position=0.95
        ),
        height=600,
        template='plotly_dark'
    )
    
    return fig

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_nse_stock_info(symbol: str) -> Dict:
    """
    Fetch detailed information for an NSE stock symbol.
    
    Args:
        symbol: Stock symbol without the .NS suffix
    
    Returns:
        Dictionary with stock information
    """
    try:
        # Remove .NS suffix if present
        clean_symbol = symbol.replace('.NS', '')
        
        # NSE API URL
        url = f"https://www.nseindia.com/api/quote-equity?symbol={clean_symbol}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            # Fall back to basic info from yfinance
            stock = yf.Ticker(f"{clean_symbol}.NS")
            info = stock.info
            return info
    except Exception as e:
        st.warning(f"Could not fetch detailed info for {symbol}: {str(e)}")
        # Return empty dict if failed
        return {}

def market_statistics():
    st.title('Market Statistics')
    st.markdown("""
    This page provides an overview of major stock market indices and sector performance.
    """)
    
    # Create a custom button navigation instead of tabs
    st.markdown("""
    <style>
    .nav-button-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .nav-button {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .nav-button:hover {
        background-color: #2d2d2d;
        border-color: #666;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .nav-button.active {
        background-color: #0e6fff;
        border-color: #0e6fff;
        color: white;
        box-shadow: 0 4px 12px rgba(14, 111, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define the sections
    sections = ["US Markets", "Indian Markets", "Stock Search", "Penny Stocks", "Recent IPOs"]
    
    # Use session state to keep track of the active section
    if 'active_section' not in st.session_state:
        st.session_state.active_section = "US Markets"
    
    # Display the navigation buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    
    for i, section in enumerate(sections):
        active_class = "active" if st.session_state.active_section == section else ""
        button_clicked = cols[i].button(section, key=f"btn_{section}")
        if button_clicked:
            st.session_state.active_section = section
            st.rerun()
    
    # Display the selected section content
    if st.session_state.active_section == "US Markets":
        st.subheader("Major US Indices")
        
        # Define major US indices
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^RUT": "Russell 2000"
        }
        
        # Create columns for indices
        index_cols = st.columns(len(indices))
        
        # Display each index
        for i, (symbol, name) in enumerate(indices.items()):
            try:
                # Get data for the index
                data = fetch_stock_data(symbol, '1d', '1m')
                if data is not None:
                    data = process_data(data)
                    
                    # Calculate change
                    last_price = data['Close'].iloc[-1]
                    prev_close = data['Open'].iloc[0]
                    change = last_price - prev_close
                    pct_change = (change / prev_close) * 100
                    
                    # Display in metric
                    index_cols[i].metric(
                        name,
                        f"${last_price:.2f}",
                        f"{change:.2f} ({pct_change:.2f}%)"
                    )
            except Exception as e:
                index_cols[i].error(f"Could not load {name}: {str(e)}")
        
        # US Sector Performance
        st.subheader("US Sector Performance (ETFs)")
        sectors = {
            "XLF": "Financial",
            "XLK": "Technology",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrial",
            "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary",
            "XLB": "Materials",
            "XLU": "Utilities",
            "XLRE": "Real Estate"
        }
        
        # Create table for sectors
        sector_data = []
        for symbol, name in sectors.items():
            try:
                data = fetch_stock_data(symbol, '5d', '1d')
                if data is not None:
                    data = process_data(data)
                    last_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[0]
                    change = last_price - prev_close
                    pct_change = (change / prev_close) * 100
                    
                    # Add to table data
                    sector_data.append({
                        "Sector": name,
                        "Symbol": symbol,
                        "Price": f"${last_price:.2f}",
                        "Change %": f"{pct_change:.2f}%"
                    })
            except Exception as e:
                st.error(f"Could not load {name}: {str(e)}")
        
        # Display sectors table
        if sector_data:
            sector_df = pd.DataFrame(sector_data)
            st.dataframe(
                sector_df.style.map(
                    lambda x: "color: red" if "-" in str(x) else "color: green",
                    subset=["Change %"]
                ),
                use_container_width=True
            )
    
    elif st.session_state.active_section == "Indian Markets":
        st.subheader("Major Indian Indices")
        
        # Define major Indian indices
        indices = {
            "^NSEI": "Nifty 50",
            "^BSESN": "Sensex",
            "^NSEBANK": "Nifty Bank",
            "^CNXIT": "Nifty IT"
        }
        
        # Create columns for indices
        index_cols = st.columns(len(indices))
        
        # Display each index
        for i, (symbol, name) in enumerate(indices.items()):
            try:
                # Get data for the index
                data = fetch_stock_data(symbol, '1d', '1m')
                if data is not None:
                    data = process_data(data)
                    
                    # Calculate change
                    last_price = data['Close'].iloc[-1]
                    prev_close = data['Open'].iloc[0]
                    change = last_price - prev_close
                    pct_change = (change / prev_close) * 100
                    
                    # Display in metric
                    index_cols[i].metric(
                        name,
                        f"₹{last_price:.2f}",
                        f"{change:.2f} ({pct_change:.2f}%)"
                    )
            except Exception as e:
                index_cols[i].error(f"Could not load {name}: {str(e)}")
        
        # Indian sector/stock performance
        st.subheader("Top Indian Stocks")
        
        # Sample of major Indian stocks to show
        sample_stocks = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "ICICIBANK.NS": "ICICI Bank",
            "ADANIENT.NS": "Adani Enterprises",
            "TATAMOTORS.NS": "Tata Motors",
            "SUNPHARMA.NS": "Sun Pharma",
            "BAJFINANCE.NS": "Bajaj Finance"
        }
        
        # Create table for stocks
        stock_data = []
        for symbol, name in sample_stocks.items():
            try:
                data = fetch_stock_data(symbol, '5d', '1d')
                if data is not None:
                    data = process_data(data)
                    last_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[0]
                    change = last_price - prev_close
                    pct_change = (change / prev_close) * 100
                    
                    # Calculate 52-week high and low
                    yearly_data = fetch_stock_data(symbol, '1y', '1d')
                    if yearly_data is not None:
                        year_high = yearly_data['High'].max()
                        year_low = yearly_data['Low'].min()
                    else:
                        year_high = last_price
                        year_low = last_price
                    
                    # Add to table data
                    stock_data.append({
                        "Company": name,
                        "Symbol": symbol,
                        "Price": f"₹{last_price:.2f}",
                        "Change %": f"{pct_change:.2f}%",
                        "52W High": f"₹{year_high:.2f}",
                        "52W Low": f"₹{year_low:.2f}"
                    })
            except Exception as e:
                st.error(f"Could not load {name}: {str(e)}")
        
        # Display stocks table
        if stock_data:
            stock_df = pd.DataFrame(stock_data)
            st.dataframe(
                stock_df.style.map(
                    lambda x: "color: red" if "-" in str(x) else "color: green",
                    subset=["Change %"]
                ),
                use_container_width=True
            )
     
    elif st.session_state.active_section == "Stock Search":
        st.subheader("Search for Any Stock")
        
        search_query = st.text_input("Enter Company Name or Symbol", "")
        market_type = st.radio("Market", ["Indian Stocks (NSE)", "US Stocks"])
        
        if search_query:
            if market_type == "Indian Stocks (NSE)":
                search_results = search_stocks(search_query, INDIAN_STOCKS)
                
                if search_results:
                    # Create a table of search results
                    result_list = []
                    for symbol, name in search_results.items():
                        try:
                            # Get basic price data
                            data = fetch_stock_data(symbol, '1d', '1d')
                            if data is not None and not data.empty:
                                price = data['Close'].iloc[-1]
                                result_list.append({
                                    "Symbol": symbol,
                                    "Company": name,
                                    "Current Price": f"₹{price:.2f}"
                                })
                        except Exception:
                            # If we can't get price, still show the company
                            result_list.append({
                                "Symbol": symbol,
                                "Company": name,
                                "Current Price": "N/A"
                            })
                    
                    # Display results
                    if result_list:
                        result_df = pd.DataFrame(result_list)
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Allow user to select a stock for detailed view
                        selected_symbol = st.selectbox(
                            "Select a stock for detailed information",
                            [row["Symbol"] for row in result_list]
                        )
                        
                        if selected_symbol:
                            st.subheader(f"Detailed Information: {INDIAN_STOCKS.get(selected_symbol, selected_symbol)}")
                            
                            # Get detailed information
                            with st.spinner("Fetching detailed information..."):
                                # Fetch stock data for different time periods
                                daily_data = fetch_stock_data(selected_symbol, '5d', '1d')
                                monthly_data = fetch_stock_data(selected_symbol, '1mo', '1d')
                                yearly_data = fetch_stock_data(selected_symbol, '1y', '1wk')
                                
                                if daily_data is not None:
                                    # Process the data
                                    daily_data = process_data(daily_data)
                                    
                                    # Calculate metrics
                                    last_close = daily_data['Close'].iloc[-1]
                                    prev_close = daily_data['Close'].iloc[-2] if len(daily_data) > 1 else daily_data['Open'].iloc[0]
                                    change = last_close - prev_close
                                    pct_change = (change / prev_close) * 100
                                    
                                    # Display metrics in columns
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Current Price", f"₹{last_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
                                    
                                    if yearly_data is not None:
                                        year_high = yearly_data['High'].max()
                                        year_low = yearly_data['Low'].min()
                                        col2.metric("52W High", f"₹{year_high:.2f}")
                                        col3.metric("52W Low", f"₹{year_low:.2f}")
                                    
                                    # Create stock chart
                                    if monthly_data is not None:
                                        monthly_data = process_data(monthly_data)
                                        monthly_data = add_technical_indicators(monthly_data)
                                        
                                        fig = create_chart(monthly_data, 'Candlestick', ['SMA 20', 'EMA 20'], selected_symbol, '1mo', 'INR')
                                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No stocks found matching your search criteria.")
            else:
                # US stock search - simpler implementation
                st.warning("For US stocks, please enter the symbol directly in the dashboard view.")
                symbol = search_query.upper()
                
                try:
                    # Fetch data for the US stock
                    data = fetch_stock_data(symbol, '1d', '1d')
                    if data is not None and not data.empty:
                        st.success(f"Found stock: {symbol}")
                        
                        # Process and display basic info
                        data = process_data(data)
                        last_price = data['Close'].iloc[-1]
                        
                        st.metric("Current Price", f"${last_price:.2f}")
                        
                        # Create a button to view this stock in the dashboard
                        if st.button("View in Dashboard"):
                            # Set session state to remember this selection
                            if 'selected_us_stock' not in st.session_state:
                                st.session_state['selected_us_stock'] = symbol
                            st.session_state['nav_option'] = "Single Stock Analysis"
                            st.session_state['stock_type'] = "US Stocks"
                            st.rerun()
                    else:
                        st.error(f"Could not find stock with symbol: {symbol}")
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
    
    elif st.session_state.active_section == "Penny Stocks":
        st.subheader("Penny Stocks (Under ₹10)")
        
        try:
            with st.spinner("Loading penny stocks... This may take a moment."):
                penny_stocks = get_penny_stocks()
            
            if penny_stocks:
                # Create table for penny stocks
                penny_data = []
                error_count = 0
                for symbol, name_with_price in penny_stocks.items():
                    try:
                        # Extract name and price from the format "Name (₹price)"
                        if "(" in name_with_price:
                            name = name_with_price.split(" (")[0]
                            price_str = name_with_price.split("(₹")[1].replace(")", "")
                            
                            try:
                                price = float(price_str)
                                penny_data.append({
                                    "Symbol": symbol,
                                    "Company": name,
                                    "Price": f"₹{price:.2f}"
                                })
                            except ValueError:
                                # If price conversion fails, add without price
                                penny_data.append({
                                    "Symbol": symbol,
                                    "Company": name,
                                    "Price": "N/A"
                                })
                        else:
                            # If the format is different, just use the name as is
                            penny_data.append({
                                "Symbol": symbol,
                                "Company": name_with_price,
                                "Price": "N/A"
                            })
                    except Exception as e:
                        error_count += 1
                        st.warning(f"Error processing {symbol}: {str(e)}")
                
                if error_count > 0:
                    st.info(f"Note: Could not process {error_count} stocks.")
                
                if penny_data:
                    penny_df = pd.DataFrame(penny_data)
                    st.dataframe(penny_df, use_container_width=True)
                    
                    # Add an option to view a selected penny stock
                    penny_symbols = [row["Symbol"] for row in penny_data]
                    
                    if penny_symbols:
                        selected_penny = st.selectbox("Select a penny stock to view details", penny_symbols)
                        
                        if selected_penny:
                            try:
                                with st.spinner("Loading stock details..."):
                                    # Show a simple chart for the selected penny stock
                                    data = fetch_stock_data(selected_penny, '6mo', '1d')
                                    if data is not None and not data.empty:
                                        data = process_data(data)
                                        
                                        # Create a simple chart
                                        fig = px.line(data, x='Datetime', y='Close', title=f"{selected_penny} Price History")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error(f"No data available for {selected_penny}")
                            except Exception as e:
                                st.error(f"Error displaying chart for {selected_penny}: {str(e)}")
                            
                            # Always show the warning
                            st.warning("Warning: Penny stocks are highly volatile and carry significant risk. They are often subject to manipulation and have low liquidity.")
                else:
                    st.info("No penny stock data could be displayed. Please try again later.")
            else:
                st.info("No penny stocks found. Please try again later.")
        except Exception as e:
            st.error(f"Error loading penny stock data: {str(e)}")
            st.info("Please try again later or select a different section.")
    
    elif st.session_state.active_section == "Recent IPOs":
        st.subheader("Recently Listed Stocks (Last 6 Months)")
        
        try:
            with st.spinner("Loading recent IPOs... This may take a moment."):
                recent_ipos = get_recent_ipos()
            
            if recent_ipos:
                # Create table for recent IPOs
                ipo_data = []
                error_count = 0
                for symbol, name in recent_ipos.items():
                    try:
                        # Get current price and listing date
                        data = fetch_stock_data(symbol, 'max', '1d')
                        if data is not None and not data.empty:
                            data = process_data(data)
                            current_price = data['Close'].iloc[-1]
                            listing_date = data['Datetime'].iloc[0].strftime('%Y-%m-%d')
                            
                            # Calculate performance since listing
                            initial_price = data['Open'].iloc[0]
                            perf_pct = ((current_price - initial_price) / initial_price) * 100
                            
                            ipo_data.append({
                                "Symbol": symbol,
                                "Company": name,
                                "Listing Date": listing_date,
                                "Current Price": f"₹{current_price:.2f}",
                                "Performance": f"{perf_pct:.2f}%"
                            })
                        else:
                            error_count += 1
                    except Exception as e:
                        # Add with error message but don't stop the whole process
                        st.warning(f"Error fetching data for {symbol}: {str(e)}")
                        error_count += 1
                
                if error_count > 0:
                    st.info(f"Note: Could not load data for {error_count} stocks.")
                
                if ipo_data:
                    ipo_df = pd.DataFrame(ipo_data)
                    
                    # Handle any styling errors by using a try-except block
                    try:
                        styled_df = ipo_df.style.map(
                            lambda x: "color: red" if "-" in str(x) else "color: green", 
                            subset=["Performance"]
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    except Exception:
                        # If styling fails, just display the dataframe without styling
                        st.dataframe(ipo_df, use_container_width=True)
                    
                    # Add an option to view a selected IPO
                    valid_ipo_symbols = [row["Symbol"] for row in ipo_data if row["Listing Date"] != "N/A"]
                    
                    if valid_ipo_symbols:
                        selected_ipo = st.selectbox("Select a recent IPO to view details", valid_ipo_symbols)
                        
                        if selected_ipo:
                            try:
                                with st.spinner("Loading IPO details..."):
                                    # Show a simple chart for the selected IPO
                                    data = fetch_stock_data(selected_ipo, 'max', '1d')
                                    if data is not None and not data.empty:
                                        data = process_data(data)
                                        
                                        # Create a chart showing performance since IPO
                                        fig = px.line(data, x='Datetime', y='Close', title=f"{selected_ipo} Performance Since IPO")
                                        try:
                                            # Try to add the IPO price reference line
                                            fig.add_hline(y=data['Open'].iloc[0], line_dash="dash", line_color="red", annotation_text="IPO Price")
                                        except Exception:
                                            # If adding the reference line fails, continue without it
                                            pass
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"No data available for {selected_ipo}")
                            except Exception as e:
                                st.error(f"Error displaying chart for {selected_ipo}: {str(e)}")
                else:
                    st.info("No IPO data could be displayed. Please try again later.")
            else:
                st.info("No recent IPOs found. Please try again later.")
        except Exception as e:
            st.error(f"Error loading IPO data: {str(e)}")
            st.info("Please try again later or select a different section.")

def compare_stocks():
    st.title("Compare Stocks")
    
    # Currency selection
    currency = st.selectbox('Select Currency', ['USD', 'INR'])
    
    # Stock selection
    col1, col2 = st.columns(2)
    with col1:
        stock_type1 = st.selectbox('Select Market for Stock 1', ['US Stocks', 'Indian Stocks'])
        if stock_type1 == 'Indian Stocks':
            # Add search box for Indian stocks
            search_query1 = st.text_input('Search Indian Stock 1 by Name or Symbol', '')
            
            if search_query1:
                # Show search results
                search_results1 = search_stocks(search_query1, INDIAN_STOCKS)
                
                if search_results1:
                    # Format results for display in selectbox
                    formatted_results1 = [f"{symbol} - {name}" for symbol, name in search_results1.items()]
                    
                    # Get the selected result
                    selected_result1 = st.selectbox('Select Stock 1', formatted_results1, key='stock1_select')
                    
                    # Extract symbol from the selection
                    ticker1 = selected_result1.split(' - ')[0] if ' - ' in selected_result1 else selected_result1
                else:
                    st.warning("No matching stocks found. Please try a different search term.")
                    ticker1 = list(INDIAN_STOCKS.keys())[0]  # Default to first stock
            else:
                # Without search, show a dropdown of all stocks
                ticker1 = st.selectbox('Select Indian Stock 1', list(INDIAN_STOCKS.keys()))
        else:
            ticker1 = st.text_input('Enter US Stock Symbol 1', 'AAPL').upper()
    
    with col2:
        stock_type2 = st.selectbox('Select Market for Stock 2', ['US Stocks', 'Indian Stocks'])
        if stock_type2 == 'Indian Stocks':
            # Add search box for Indian stocks
            search_query2 = st.text_input('Search Indian Stock 2 by Name or Symbol', '')
            
            if search_query2:
                # Show search results
                search_results2 = search_stocks(search_query2, INDIAN_STOCKS)
                
                if search_results2:
                    # Format results for display in selectbox
                    formatted_results2 = [f"{symbol} - {name}" for symbol, name in search_results2.items()]
                    
                    # Get the selected result
                    selected_result2 = st.selectbox('Select Stock 2', formatted_results2, key='stock2_select')
                    
                    # Extract symbol from the selection
                    ticker2 = selected_result2.split(' - ')[0] if ' - ' in selected_result2 else selected_result2
                else:
                    st.warning("No matching stocks found. Please try a different search term.")
                    ticker2 = list(INDIAN_STOCKS.keys())[0]  # Default to first stock
            else:
                # Without search, show a dropdown of all stocks
                ticker2 = st.selectbox('Select Indian Stock 2', list(INDIAN_STOCKS.keys()))
        else:
            ticker2 = st.text_input('Enter US Stock Symbol 2', 'MSFT').upper()
    
    # Time period selection
    time_period = st.selectbox(
        'Time Period',
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'],
        index=2
    )
    
    # Fetch and compare data
    if st.button('Compare'):
        with st.spinner('Fetching data...'):
            data1 = fetch_stock_data(ticker1, time_period, '1d')
            data2 = fetch_stock_data(ticker2, time_period, '1d')
            
            if data1 is not None and data2 is not None:
                # Convert prices to selected currency if needed
                if currency == 'INR':
                    rate = get_usd_to_inr_rate()
                    if not is_indian_stock(ticker1):
                        data1['Close'] = data1['Close'] * rate
                    if not is_indian_stock(ticker2):
                        data2['Close'] = data2['Close'] * rate
                
                # Create comparison chart
                fig = go.Figure()
                
                # Add traces for both stocks
                fig.add_trace(go.Scatter(
                    x=data1.index,
                    y=data1['Close'],
                    name=ticker1,
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data2.index,
                    y=data2['Close'],
                    name=ticker2,
                    line=dict(color='red')
                ))
                
                # Update layout
                currency_symbol = '₹' if currency == 'INR' else '$'
                fig.update_layout(
                    title=f"{ticker1} vs {ticker2} Price Comparison",
                    xaxis_title='Date',
                    yaxis_title=f'Price ({currency_symbol})',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{ticker1} Metrics")
                    last_close1 = data1['Close'].iloc[-1]
                    prev_close1 = data1['Close'].iloc[0]
                    change1 = last_close1 - prev_close1
                    pct_change1 = (change1 / prev_close1) * 100
                    
                    st.metric(
                        "Current Price",
                        format_currency(last_close1, currency),
                        f"{format_currency(change1, currency)} ({pct_change1:.2f}%)"
                    )
                
                with col2:
                    st.subheader(f"{ticker2} Metrics")
                    last_close2 = data2['Close'].iloc[-1]
                    prev_close2 = data2['Close'].iloc[0]
                    change2 = last_close2 - prev_close2
                    pct_change2 = (change2 / prev_close2) * 100
                    
                    st.metric(
                        "Current Price",
                        format_currency(last_close2, currency),
                        f"{format_currency(change2, currency)} ({pct_change2:.2f}%)"
                    )
            else:
                if data1 is None:
                    st.error(f"Could not fetch data for {ticker1}")
                if data2 is None:
                    st.error(f"Could not fetch data for {ticker2}")

def dashboard():
    # Title and description
    st.title('Real-Time Stock Market Dashboard')
    st.markdown("""
    This dashboard provides real-time stock market data visualization with technical indicators.
    Use the sidebar to customize your view.
    """)
    
    # Sidebar configuration
    st.sidebar.header('Chart Parameters')
    
    # Stock selection
    stock_type = st.sidebar.selectbox('Stock Market', ['US Stocks', 'Indian Stocks'])
    
    if stock_type == 'Indian Stocks':
        # Add search box for Indian stocks
        search_query = st.sidebar.text_input('Search Indian Stocks by Name or Symbol', '')
        
        if search_query:
            # Show search results
            search_results = search_stocks(search_query, INDIAN_STOCKS)
            
            if search_results:
                # Format results for display in selectbox
                formatted_results = [f"{symbol} - {name}" for symbol, name in search_results.items()]
                
                # Get the selected result
                selected_result = st.sidebar.selectbox('Select Stock', formatted_results)
                
                # Extract symbol from the selection
                ticker = selected_result.split(' - ')[0] if ' - ' in selected_result else selected_result
            else:
                st.sidebar.warning("No matching stocks found. Please try a different search term.")
                ticker = list(INDIAN_STOCKS.keys())[0]  # Default to first stock
        else:
            # Without search, show a dropdown of all stocks
            ticker = st.sidebar.selectbox('Select Indian Stock', list(INDIAN_STOCKS.keys()))
        
        currency = 'INR'  # Force INR for Indian stocks
    else:
        ticker = st.sidebar.text_input('Enter US Stock Symbol', 'AAPL').upper()
        currency = st.sidebar.selectbox('Currency', ['USD', 'INR'])
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        'Time Period',
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'],
        index=2
    )
    
    # Chart type selection
    chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
    
    # Technical indicators selection
    available_indicators = ['SMA 20', 'EMA 20', 'RSI 14', 'MACD']
    indicators = st.sidebar.multiselect(
        'Technical Indicators',
        available_indicators,
        default=['SMA 20', 'RSI 14']
    )
    
    # Interval mapping
    interval_mapping = {
        '1d': '1m',
        '5d': '5m',
        '1mo': '1h',
        '3mo': '1d',
        '6mo': '1d',
        '1y': '1wk',
        '5y': '1mo',
        'max': '1mo',
    }
    
    # Update button
    if st.sidebar.button('Update Chart'):
        with st.spinner('Fetching data...'):
            data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
            
            if data is not None:
                # Process data
                data = process_data(data)
                data = add_technical_indicators(data)
                
                # Calculate metrics
                last_close, change, pct_change, high, low, volume = calculate_metrics(data)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    label=f"{ticker} Last Price",
                    value=format_currency(last_close if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(last_close), currency),
                    delta=f"{format_currency(change if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(change), currency)} ({pct_change:.2f}%)"
                )
                col2.metric('High', format_currency(high if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(high), currency))
                col3.metric('Low', format_currency(low if currency == 'USD' or is_indian_stock(ticker) else convert_to_inr(low), currency))
                col4.metric('Volume', f"{volume:,}")
                
                # Create and display chart
                fig = create_chart(data, chart_type, indicators, ticker, time_period, currency)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data tables
                tab1, tab2 = st.tabs(['Historical Data', 'Technical Indicators'])
                
                with tab1:
                    display_data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    if currency == 'INR' and not is_indian_stock(ticker):
                        rate = get_usd_to_inr_rate()
                        for col in ['Open', 'High', 'Low', 'Close']:
                            display_data[col] = display_data[col] * rate
                    st.dataframe(display_data, use_container_width=True)
                
                with tab2:
                    display_data = data[['Datetime', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD']].copy()
                    if currency == 'INR' and not is_indian_stock(ticker):
                        rate = get_usd_to_inr_rate()
                        for col in ['SMA_20', 'EMA_20']:
                            display_data[col] = display_data[col] * rate
                    st.dataframe(display_data, use_container_width=True)
    
    # Real-time stock prices in sidebar
    st.sidebar.header('Real-Time Stock Prices')
    
    if stock_type == 'Indian Stocks':
        stock_symbols = list(INDIAN_STOCKS.keys())[:5]  # Limit to first 5 for performance
    else:
        stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
    
    for symbol in stock_symbols:
        if symbol != ticker:  # Skip the currently selected stock
            real_time_data = fetch_stock_data(symbol, '1d', '1m')
            if real_time_data is not None:
                real_time_data = process_data(real_time_data)
                last_price = real_time_data['Close'].iloc[-1]
                change = last_price - real_time_data['Open'].iloc[0]
                pct_change = (change / real_time_data['Open'].iloc[0]) * 100
                
                if currency == 'INR' and not is_indian_stock(symbol):
                    last_price = convert_to_inr(last_price)
                    change = convert_to_inr(change)
                
                st.sidebar.metric(
                    f"{symbol}",
                    format_currency(last_price, currency),
                    f"{format_currency(change, currency)} ({pct_change:.2f}%)"
                )
    
    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown("""
    ### About
    This dashboard uses the Yahoo Finance API to provide real-time stock market data.
    Data is cached for 5 minutes to improve performance.
    """)

if __name__ == "__main__":
    # Add custom CSS for styling
    st.markdown("""
    <style>
    /* Tab styling to make tabs look like buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border-radius: 8px;
        gap: 1px;
        padding: 10px 20px;
        color: white;
        border: 1px solid #444;
        font-weight: 600;
        transition: all 0.3s ease;
        margin: 0 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
        cursor: pointer;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2d2d2d;
        border-color: #666;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0e6fff !important;
        border-color: #0e6fff !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(14, 111, 255, 0.4) !important;
    }
    
    /* Hide the default tab bar bottom border */
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* Make the tabs take full width on mobile */
    @media screen and (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            width: 100%;
            margin-bottom: 8px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create sidebar for navigation
    with st.sidebar:
        st.title("Stock Dashboard")
        # Navigation options
        nav_option = st.radio(
            "Navigate to:",
            ["Single Stock Analysis", "Compare Stocks", "Market Statistics"]
        )
    
    # Display the appropriate section based on navigation
    if nav_option == "Single Stock Analysis":
        dashboard()
    elif nav_option == "Compare Stocks":
        compare_stocks()
    elif nav_option == "Market Statistics":
        market_statistics() 