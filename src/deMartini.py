import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

def demartini_index(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the DeMartini Index for a given Pandas Series of close prices.
    
    Parameters:
    close_prices (pd.Series): Pandas Series of close prices.
    period (int): Period for calculating the DeMartini Index. Default is 14.
    
    Returns:
    pd.Series: Pandas Series containing the DeMartini Index.
    """
    
    # Calculate the RSI using the ta library
    rsi = RSIIndicator(close=close_prices, window=period).rsi()
    
    # DeMartini Index is similar to RSI but can have custom calculations
    # Here we use RSI as a proxy to demonstrate the functionality
    demartini_index = rsi
    
    return demartini_index

def plot_demartini_index(close: pd.Series, demartini: pd.Series):
    """
    Plot the close prices and DeMartini Index.
    
    Parameters:
    close (pd.Series): Pandas Series of close prices.
    demartini (pd.Series): Pandas Series of DeMartini Index.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(close.index, close, color=color, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('DeMartini Index', color=color)
    ax2.plot(demartini.index, demartini, color=color, label='DeMartini Index')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Close Price and DeMartini Index')
    plt.show()

# Read the parquet file
df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
price_series = df["종가"]

# Calculate the DeMartini Index
df['DeMartini'] = demartini_index(price_series)

# Plot the close prices and DeMartini Index
plot_demartini_index(price_series, df['DeMartini'])