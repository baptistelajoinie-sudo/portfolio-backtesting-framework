import pandas as pd
import yfinance as yf
import requests
from io import StringIO


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

SP500_BENCHMARK = "^GSPC"

MIN_COVERAGE = 0.95


# ============================================================
# S&P 500 UNIVERSE
# ============================================================

def get_sp500_tickers():
    """
    Retrieve the current S&P 500 constituent list.

    IMPORTANT
    ---------
    This uses the current constituent universe and therefore
    introduces survivorship bias in the historical backtest.
    """

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    constituents = tables[0]

    tickers = (
        constituents["Symbol"]
        .astype(str)
        .tolist()
    )

    # Yahoo Finance convention:
    # BRK.B -> BRK-B
    tickers = [
        ticker.replace(".", "-")
        for ticker in tickers
    ]

    return tickers


# ============================================================
# PRICE DATA
# ============================================================

def download_prices(
    tickers,
    start_date=START_DATE,
    end_date=END_DATE,
):
    """
    Download adjusted historical closing prices.
    """

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices


# ============================================================
# DATA QUALITY
# ============================================================

def select_eligible_assets(
    prices,
    min_coverage=MIN_COVERAGE,
):
    """
    Keep assets with sufficient historical price coverage.
    """

    coverage = (
        prices.notna().mean()
    )

    eligible_assets = (
        coverage[
            coverage >= min_coverage
        ]
        .index
        .tolist()
    )

    return eligible_assets


def clean_prices(
    prices,
    eligible_assets,
):
    """
    Keep eligible assets and forward-fill isolated
    missing observations.
    """

    cleaned = (
        prices[
            eligible_assets
        ]
        .copy()
    )

    cleaned = cleaned.ffill()

    return cleaned


# ============================================================
# RETURNS
# ============================================================

def calculate_returns(
    prices,
):
    """
    Calculate daily simple returns.

    r_t = P_t / P_(t-1) - 1
    """

    returns = prices.pct_change(
        fill_method=None
    )

    return returns


# ============================================================
# S&P 500 BENCHMARK
# ============================================================

def download_benchmark(
    ticker=SP500_BENCHMARK,
    start_date=START_DATE,
    end_date=END_DATE,
):
    """
    Download S&P 500 benchmark prices and daily returns.
    """

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    prices = data["Close"]

    if isinstance(
        prices,
        pd.DataFrame,
    ):
        prices = prices.iloc[:, 0]

    returns = prices.pct_change(
        fill_method=None
    )

    returns.name = "S&P 500"

    return prices, returns