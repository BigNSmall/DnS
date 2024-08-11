import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sonar_indicator(prices: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Sonar indicator based on price and volume.
    
    Parameters:
    prices (pd.Series): Series of stock prices.
    volume (pd.Series): Series of trading volume.
    window (int): Window size for the moving average.
    
    Returns:
    pd.Series: Sonar indicator values.
    """
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Calculate the moving average of the log returns
    moving_avg_returns = log_returns.rolling(window=window).mean()
    
    # Calculate the volume-weighted moving average of the log returns
    volume_weighted_returns = (log_returns * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    
    # Calculate the Sonar indicator
    sonar = moving_avg_returns - volume_weighted_returns
    
    # Return the Sonar indicator
    return sonar

def plot_sonar_indicator(prices: pd.Series, sonar: pd.Series):
    """
    Plot the Sonar indicator along with the stock prices.
    
    Parameters:
    prices (pd.Series): Series of stock prices.
    sonar (pd.Series): Series of Sonar indicator values.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot stock prices
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Stock Prices')
    plt.title('Stock Prices and Sonar Indicator')
    plt.legend()
    
    # Plot Sonar indicator
    plt.subplot(2, 1, 2)
    plt.plot(sonar, label='Sonar Indicator', color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    
    plt.show()

# Sample data
df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
prices = df["종가"]
volume = df["거래량"]

# Calculate the Sonar indicator
sonar = sonar_indicator(prices, volume)

# Plot the Sonar indicator
plot_sonar_indicator(prices, sonar)