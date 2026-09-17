from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")


def load_returns():

    return pd.read_csv(
        RESULTS_DIR / "strategy_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def cumulative_performance_chart(returns):

    wealth = (
        1 + returns.fillna(0)
    ).cumprod() * 100

    plt.figure(figsize=(12, 7))

    for column in wealth.columns:
        plt.plot(
            wealth.index,
            wealth[column],
            label=column,
            linewidth=2,
        )

    plt.title(
        "Cumulative Portfolio Performance"
    )

    plt.ylabel(
        "Portfolio Value (Initial = 100)"
    )

    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "cumulative_performance.png",
        dpi=200,
    )

    plt.close()


def drawdown_chart(returns):

    wealth = (
        1 + returns.fillna(0)
    ).cumprod()

    running_max = wealth.cummax()

    drawdowns = (
        wealth / running_max - 1
    )

    plt.figure(figsize=(12, 7))

    for column in drawdowns.columns:
        plt.plot(
            drawdowns.index,
            drawdowns[column],
            label=column,
            linewidth=1.8,
        )

    plt.title("Portfolio Drawdowns")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "drawdowns.png",
        dpi=200,
    )

    plt.close()


def rolling_volatility_chart(
    returns,
    window=63,
):

    rolling_vol = (
        returns
        .rolling(window)
        .std()
        * np.sqrt(252)
    )

    plt.figure(figsize=(12, 7))

    for column in rolling_vol.columns:
        plt.plot(
            rolling_vol.index,
            rolling_vol[column],
            label=column,
            linewidth=1.8,
        )

    plt.title(
        "63-Day Rolling Annualized Volatility"
    )

    plt.ylabel(
        "Annualized Volatility"
    )

    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "rolling_volatility.png",
        dpi=200,
    )

    plt.close()


def risk_parity_allocation_chart():

    weights = pd.read_csv(
        RESULTS_DIR
        / "risk_parity_weights.csv",
        index_col=0,
        parse_dates=True,
    )

    # Use the most recent portfolio allocation.
    latest_weights = (
        weights
        .iloc[-1]
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    # Show the 20 largest allocations.
    top_weights = (
        latest_weights
        .head(20)
        * 100
    )

    # Reverse order so largest bar appears on top.
    top_weights = (
        top_weights
        .sort_values()
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_weights.index,
        top_weights.values,
    )

    plt.title(
        "Risk Parity Portfolio - "
        "20 Largest Allocations"
    )

    plt.xlabel(
        "Portfolio Weight (%)"
    )

    plt.ylabel(
        "Ticker"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "risk_parity_allocations.png",
        dpi=200,
    )

    plt.close()


if __name__ == "__main__":

    returns = load_returns()

    cumulative_performance_chart(
        returns
    )

    drawdown_chart(
        returns
    )

    rolling_volatility_chart(
        returns
    )

    risk_parity_allocation_chart()

    print(
        "Charts generated in /results."
    )