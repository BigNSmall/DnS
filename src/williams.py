import pandas as pd
import matplotlib.pyplot as plt

def williams_r(high, low, close, lookback_period=14):
    """
    Calculate the Williams %R indicator.

    Parameters:
    high (pd.Series): High prices
    low (pd.Series): Low prices
    close (pd.Series): Close prices
    lookback_period (int): The lookback period for the indicator

    Returns:
    pd.Series: Williams %R values
    """
    highest_high = high.rolling(window=lookback_period).max()
    lowest_low = low.rolling(window=lookback_period).min()

    # Calculate Williams %R
    will_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

    return will_r

def plot_williams_r(df, lookback_period=14):
    """
    Plot the Williams %R indicator.

    Parameters:
    df (pd.DataFrame): DataFrame containing high, low, close prices
    lookback_period (int): The lookback period for the indicator
    """
    will_r = williams_r(df['고가'], df['저가'], df['종가'], lookback_period)

    plt.figure(figsize=(14, 7))
    
    # Plotting the closing prices
    plt.subplot(2, 1, 1)
    plt.plot(df['종가'], label='Close Price')
    plt.title('Close Price')
    plt.legend()

    # Plotting the Williams %R
    plt.subplot(2, 1, 2)
    plt.plot(will_r, label=f'Williams %R (lookback period={lookback_period})', color='orange')
    plt.axhline(-20, linestyle='--', alpha=0.5, color='red')
    plt.axhline(-80, linestyle='--', alpha=0.5, color='green')
    plt.title('Williams %R')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# data = {
#     'high': [127.01, 127.62, 128.06, 127.53, 128.06, 128.77, 128.73, 129.29, 129.29, 128.47, 128.09, 128.18, 127.77, 128.09, 128.06, 128.20, 128.29, 128.13, 128.36, 128.29],
#     'low': [125.36, 126.38, 126.98, 126.39, 126.92, 127.65, 127.35, 128.10, 127.49, 127.00, 126.60, 126.63, 126.32, 127.22, 127.34, 127.63, 127.86, 127.33, 127.75, 127.40],
#     'close': [126.15, 127.17, 127.89, 126.88, 127.95, 128.56, 128.42, 128.80, 128.73, 127.80, 127.63, 127.90, 127.45, 127.95, 127.80, 127.96, 128.05, 127.67, 128.15, 127.88]
# }

# df = pd.DataFrame(data)

df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
# Plot Williams %R
plot_williams_r(df)