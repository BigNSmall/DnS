import pandas as pd
import matplotlib.pyplot as plt

def calculate_pivot_points(high, low, close):
    """
    피봇 포인트와 주요 지지 및 저항 수준을 계산합니다.
    
    Parameters:
    high (pd.Series): 고가를 포함하는 시리즈
    low (pd.Series): 저가를 포함하는 시리즈
    close (pd.Series): 종가를 포함하는 시리즈
    
    Returns:
    pd.DataFrame: 피봇 포인트와 주요 지지 및 저항 수준을 포함하는 DataFrame
    """
    pivot_points = pd.DataFrame(index=high.index)
    pivot_points['Pivot'] = (high + low + close) / 3
    
    pivot_points['R1'] = (2 * pivot_points['Pivot']) - low
    pivot_points['S1'] = (2 * pivot_points['Pivot']) - high
    
    pivot_points['R2'] = pivot_points['Pivot'] + (high - low)
    pivot_points['S2'] = pivot_points['Pivot'] - (high - low)
    
    pivot_points['R3'] = high + 2 * (pivot_points['Pivot'] - low)
    pivot_points['S3'] = low - 2 * (high - pivot_points['Pivot'])
    
    return pivot_points

def plot_pivot_points(close, pivot_points):
    """
    피봇 포인트와 주요 지지 및 저항 수준을 시각화합니다.
    
    Parameters:
    close (pd.Series): 종가를 포함하는 시리즈
    pivot_points (pd.DataFrame): 피봇 포인트와 주요 지지 및 저항 수준을 포함하는 DataFrame
    """
    plt.figure(figsize=(14, 7))
    plt.plot(close.index, close, label='Close Price', marker='o')
    
    plt.plot(pivot_points.index, pivot_points['Pivot'], label='Pivot', linestyle='--', color='blue')
    plt.plot(pivot_points.index, pivot_points['R1'], label='R1', linestyle='--', color='green')
    plt.plot(pivot_points.index, pivot_points['S1'], label='S1', linestyle='--', color='red')
    plt.plot(pivot_points.index, pivot_points['R2'], label='R2', linestyle='--', color='green')
    plt.plot(pivot_points.index, pivot_points['S2'], label='S2', linestyle='--', color='red')
    plt.plot(pivot_points.index, pivot_points['R3'], label='R3', linestyle='--', color='green')
    plt.plot(pivot_points.index, pivot_points['S3'], label='S3', linestyle='--', color='red')
    
    plt.title('Pivot Points and Support/Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# 예시 데이터 생성
high =  df["고가"]
low =  df["저가"]
close =  df["종가"]

# 피봇 포인트 계산
pivot_points = calculate_pivot_points(high, low, close)

# 피봇 포인트 시각화
plot_pivot_points(close, pivot_points)