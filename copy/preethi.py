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


# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom horizontal navbar using HTML/CSS
st.markdown("""
    <style>
        .navbar {
            background-color: #f8f9fa;
            padding: 10px 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-around;
            font-weight: 600;
        }
        .navbar a {
            color: #495057;
            text-decoration: none;
            padding: 10px 16px;
            border-radius: 6px;
        }
        .navbar a:hover {
            background-color: #dee2e6;
        }
        .active {
            background-color: #ced4da;
        }
    </style>
    <div class="navbar">
        <a href="/?nav=Dashboard" class="%s">📈 Dashboard</a>
        <a href="/?nav=Compare" class="%s">🔍 Compare Stocks</a>
        <a href="/?nav=About" class="%s">ℹ️ About</a>
    </div>
""" % (
    "active" if st.query_params.get("nav", ["Dashboard"])[0] == "Dashboard" else "",
    "active" if st.query_params.get("nav", ["Dashboard"])[0] == "Compare" else "",
    "active" if st.query_params.get("nav", ["Dashboard"])[0] == "About" else ""
), unsafe_allow_html=True)
# Read the nav query parameter
nav = st.query_params.get("nav", ["Dashboard"])[0]


# Define Indian stock symbols (you can add more)
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro'
}

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

def compare_stocks(stock1: str, stock2: str, time_period: str, currency: str):
    """
    Compare two stocks side by side.
    
    Args:
        stock1: First stock symbol
        stock2: Second stock symbol
        time_period: Time period for comparison
        currency: Currency to display ('USD' or 'INR')
    """
    # Fetch data for both stocks
    data1 = fetch_stock_data(stock1, time_period, interval_mapping[time_period])
    data2 = fetch_stock_data(stock2, time_period, interval_mapping[time_period])
    
    if data1 is None or data2 is None:
        st.error("Failed to fetch data for one or both stocks. Please check the ticker symbols.")
        return
    
    # Process data
    data1 = process_data(data1)
    data2 = process_data(data2)
    
    # Add technical indicators
    data1 = add_technical_indicators(data1)
    data2 = add_technical_indicators(data2)
    
    # Create two columns for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{stock1} Analysis")
        # Calculate and display metrics
        last_close1, change1, pct_change1, high1, low1, volume1 = calculate_metrics(data1)
        st.metric(
            label="Last Price",
            value=format_currency(last_close1 if currency == 'USD' or is_indian_stock(stock1) else convert_to_inr(last_close1), currency),
            delta=f"{format_currency(change1 if currency == 'USD' or is_indian_stock(stock1) else convert_to_inr(change1), currency)} ({pct_change1:.2f}%)"
        )
        st.metric("High", format_currency(high1 if currency == 'USD' or is_indian_stock(stock1) else convert_to_inr(high1), currency))
        st.metric("Low", format_currency(low1 if currency == 'USD' or is_indian_stock(stock1) else convert_to_inr(low1), currency))
        st.metric("Volume", f"{volume1:,}")
        
        # Create chart for stock1
        fig1 = create_chart(data1, 'Line', ['SMA 20', 'RSI 14'], stock1, time_period, currency)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader(f"{stock2} Analysis")
        # Calculate and display metrics
        last_close2, change2, pct_change2, high2, low2, volume2 = calculate_metrics(data2)
        st.metric(
            label="Last Price",
            value=format_currency(last_close2 if currency == 'USD' or is_indian_stock(stock2) else convert_to_inr(last_close2), currency),
            delta=f"{format_currency(change2 if currency == 'USD' or is_indian_stock(stock2) else convert_to_inr(change2), currency)} ({pct_change2:.2f}%)"
        )
        st.metric("High", format_currency(high2 if currency == 'USD' or is_indian_stock(stock2) else convert_to_inr(high2), currency))
        st.metric("Low", format_currency(low2 if currency == 'USD' or is_indian_stock(stock2) else convert_to_inr(low2), currency))
        st.metric("Volume", f"{volume2:,}")
        
        # Create chart for stock2
        fig2 = create_chart(data2, 'Line', ['SMA 20', 'RSI 14'], stock2, time_period, currency)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Combined performance comparison
    st.subheader("Performance Comparison")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write(f"{stock1} vs {stock2} - Price Change")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=data1['Datetime'],
            y=data1['Close'],
            name=stock1,
            line=dict(color='blue')
        ))
        fig_compare.add_trace(go.Scatter(
            x=data2['Datetime'],
            y=data2['Close'],
            name=stock2,
            line=dict(color='red')
        ))
        fig_compare.update_layout(
            title="Price Comparison",
            xaxis_title="Date",
            yaxis_title=f"Price ({'₹' if currency == 'INR' else '$'})",
            template='plotly_dark'
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    
    with col4:
        st.write(f"{stock1} vs {stock2} - RSI Comparison")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data1['Datetime'],
            y=data1['RSI_14'],
            name=f"{stock1} RSI",
            line=dict(color='blue')
        ))
        fig_rsi.add_trace(go.Scatter(
            x=data2['Datetime'],
            y=data2['RSI_14'],
            name=f"{stock2} RSI",
            line=dict(color='red')
        ))
        fig_rsi.update_layout(
            title="RSI Comparison",
            xaxis_title="Date",
            yaxis_title="RSI",
            template='plotly_dark'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

def main():
    # Title and description
    st.title('📈 Real-Time Stock Market Dashboard')
    st.markdown("""
    This dashboard provides real-time stock market data visualization with technical indicators.
    Use the sidebar to customize your view.
    """)
    
    # Sidebar configuration
    st.sidebar.header('Chart Parameters')
    
    # Stock selection
    stock_type = st.sidebar.selectbox('Stock Market', ['US Stocks', 'Indian Stocks'])
    
    if stock_type == 'Indian Stocks':
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
        stock_symbols = list(INDIAN_STOCKS.keys())
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
    nav = st.sidebar.radio("Navigation", ["Dashboard", "Compare", "About"])

    if nav == "Dashboard":
        main()

    elif nav == "Compare":
        st.subheader("🔍 Stock Comparison")
        
        # Stock selection
        stock_type = st.selectbox('Stock Market', ['US Stocks', 'Indian Stocks'], key='compare_stock_type')
        
        if stock_type == 'Indian Stocks':
            stock1 = st.selectbox('Select First Indian Stock', list(INDIAN_STOCKS.keys()), key='compare_stock1')
            stock2 = st.selectbox('Select Second Indian Stock', list(INDIAN_STOCKS.keys()), key='compare_stock2')
            currency = 'INR'
        else:
            stock1 = st.text_input('Enter First US Stock Symbol', 'AAPL', key='compare_stock1').upper()
            stock2 = st.text_input('Enter Second US Stock Symbol', 'MSFT', key='compare_stock2').upper()
            currency = st.selectbox('Currency', ['USD', 'INR'], key='compare_currency')
        
        # Time period selection
        time_period = st.selectbox(
            'Time Period',
            ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'],
            index=2,
            key='compare_time_period'
        )
        
        if st.button('Compare Stocks'):
            compare_stocks(stock1, stock2, time_period, currency)

    elif nav == "About":
        st.subheader("ℹ️ About This App")
        st.markdown("""
            This is a real-time stock dashboard built with **Streamlit**, using data from **Yahoo Finance**.
            
            **Features:**
            - Live charts with technical indicators
            - USD to INR conversion
            - Real-time side panel stock prices
            - Supports US & Indian Stocks
            
            Created by [Your Name].
        """)
