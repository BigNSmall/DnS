import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

def calculate_lrl(prices: pd.Series, period: int) -> pd.Series:
    lrl = np.zeros_like(prices)
    for i in range(period, len(prices)):
        y = prices[i-period:i]
        x = np.arange(period)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        lrl[i] = slope * (period - 1) + intercept
    return pd.Series(lrl, index=prices.index)

def calculate_lrs(lrl: pd.Series, period: int) -> pd.Series:
    lrs = np.zeros_like(lrl)
    lrs[period:] = (lrl[period:] - lrl[:-period]) / lrl[:-period] * 100
    return pd.Series(lrs, index=lrl.index)

def calculate_lrl_statsmodels(prices: pd.Series, period: int) -> pd.Series:
    lrl = np.zeros_like(prices)
    for i in range(period, len(prices)):
        y = prices[i-period:i]
        x = np.arange(period)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        intercept, slope = model.params
        lrl[i] = slope * (period - 1) + intercept
    return pd.Series(lrl, index=prices.index)

def calculate_lrs_statsmodels(lrl: pd.Series, period: int) -> pd.Series:
    lrs = np.zeros_like(lrl)
    lrs[period:] = (lrl[period:] - lrl[:-period]) / lrl[:-period] * 100
    return pd.Series(lrs, index=lrl.index)

def plot_lrl_lrs(prices: pd.Series, lrl: pd.Series, lrs: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(prices.index, prices, label='Price', color='blue')
    ax1.plot(lrl.index, lrl, label='LRL', color='red')
    ax1.set_title('Price and Linear Regression Line (LRL)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(lrs.index, lrs, label='LRS', color='green')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Linear Regression Slope (LRS)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('LRS')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# 예제 사용
if __name__ == "__main__":
    df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")

    data = df["종가"]
    period = 14

    lrl = calculate_lrl(data, period)
    lrs = calculate_lrs(lrl, period)

    lrl_statsmodels = calculate_lrl_statsmodels(data, period)
    lrs_statsmodels = calculate_lrs_statsmodels(lrl_statsmodels, period)

    plot_lrl_lrs(data, lrl, lrs)
    plot_lrl_lrs(data, lrl_statsmodels, lrs_statsmodels)