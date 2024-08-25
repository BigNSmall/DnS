import pandas as pd
import matplotlib.pyplot as plt


def calculate_buffett_index(market_cap_series: pd.Series, country: str) -> pd.Series:
    # GDP 데이터 (단위: 달러, 년도: 2023 기준)
    gdp_data = {
        "USA": 25.5e12,  # 25.5 trillion USD
        "KOR": 1.8e12,  # 1.8 trillion USD
    }

    if country not in gdp_data:
        raise ValueError("Country must be 'USA' or 'KOR'")

    gdp = gdp_data[country]

    # 버핏 지수 계산
    buffett_index = (market_cap_series / gdp) * 100

    return buffett_index


def plot_individual_buffett_index(buffett_index: pd.Series, country: str):
    plt.figure(figsize=(12, 6))

    plt.plot(
        buffett_index.index,
        buffett_index,
        label=f"Buffett Index ({country})",
        marker="o",
    )

    plt.axhline(y=100, color="r", linestyle="--", label="100% Threshold")

    plt.title(f"Buffett Index for {country}")
    plt.xlabel("Date")
    plt.ylabel("Buffett Index (%)")
    plt.legend()
    plt.grid(True)

    plt.show()


# df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
# market_cap_series = df["종가"]
# # 버핏 지수 계산 (미국 기준)
# buffett_index_usa = calculate_buffett_index(market_cap_series, 'USA')

# # 버핏 지수 계산 (한국 기준)
# buffett_index_kor = calculate_buffett_index(market_cap_series, 'KOR')

# # 버핏 지수 시각화 (미국)
# plot_individual_buffett_index(buffett_index_usa, 'USA')

# # 버핏 지수 시각화 (한국)
# plot_individual_buffett_index(buffett_index_kor, 'KOR')
