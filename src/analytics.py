import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_return(
    returns,
):
    """
    Compound annual growth rate.
    """

    returns = returns.dropna()

    total_growth = (
        1 + returns
    ).prod()

    years = (
        len(returns)
        / TRADING_DAYS
    )

    return (
        total_growth
        ** (1 / years)
        - 1
    )


def annualized_volatility(
    returns,
):
    """
    Annualized standard deviation of daily returns.
    """

    return (
        returns.dropna().std()
        * np.sqrt(
            TRADING_DAYS
        )
    )


def sharpe_ratio(
    returns,
    risk_free_rate=0.0,
):
    """
    Annualized Sharpe ratio.
    """

    ann_return = (
        annualized_return(
            returns
        )
    )

    ann_vol = (
        annualized_volatility(
            returns
        )
    )

    if ann_vol == 0:
        return np.nan

    return (
        ann_return
        - risk_free_rate
    ) / ann_vol


def drawdown_series(
    returns,
):
    """
    Drawdown from previous portfolio peak.
    """

    wealth = (
        1
        + returns.fillna(0)
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth
        / running_max
        - 1
    )

    return drawdown


def maximum_drawdown(
    returns,
):
    """
    Maximum peak-to-trough drawdown.
    """

    return (
        drawdown_series(
            returns
        )
        .min()
    )


def performance_summary(
    returns,
):
    """
    Main strategy performance metrics.
    """

    return pd.Series({
        "Annualized Return":
            annualized_return(
                returns
            ),

        "Annualized Volatility":
            annualized_volatility(
                returns
            ),

        "Sharpe Ratio":
            sharpe_ratio(
                returns
            ),

        "Maximum Drawdown":
            maximum_drawdown(
                returns
            ),
    })


def rolling_volatility(
    returns,
    window=63,
):
    """
    Rolling annualized volatility.
    """

    return (
        returns
        .rolling(window)
        .std()
        * np.sqrt(
            TRADING_DAYS
        )
    )