import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    # Download stock data
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(data, weights):
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Portfolio metrics
    portfolio_return = np.dot(returns.mean(), weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )
    sharpe_ratio = portfolio_return / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to plot stock trends
def plot_stock_trends(data):
    plt.figure(figsize=(10, 6))
    data.plot(title="Stock Price Trends")
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.grid()
    st.pyplot(plt)

# Streamlit Interactive Dashboard
def dashboard():
    st.title("Interactive Stock Portfolio Analysis")

    # User inputs for stock tickers and dates
    tickers = st.text_input("Enter stock symbols (comma-separated):", "AAPL,GOOGL,AMZN")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp("2023-01-01"))

    if st.button("Fetch and Analyze"):
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
        data = fetch_stock_data(tickers_list, start_date, end_date)['Adj Close']

        st.subheader("Stock Price Trends")
        plot_stock_trends(data)

        # Default weights for portfolio (equal weights)
        weights = np.array([1 / len(tickers_list)] * len(tickers_list))
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_metrics(data, weights)

        # Display portfolio metrics
        st.subheader("Portfolio Metrics")
        st.write(f"**Portfolio Return (Annualized):** {portfolio_return:.2f}")
        st.write(f"**Portfolio Volatility (Annualized):** {portfolio_volatility:.2f}")
        st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

# Run the dashboard
if __name__ == "__main__":
    dashboard()
