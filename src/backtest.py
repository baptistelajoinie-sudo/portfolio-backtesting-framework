from pathlib import Path
import sys

import pandas as pd


sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parent
    )
)


from data import (
    START_DATE,
    END_DATE,
    get_sp500_tickers,
    download_prices,
    select_eligible_assets,
    clean_prices,
    calculate_returns,
    download_benchmark,
)

from strategies import (
    equal_weight_weights,
    inverse_volatility_weights,
    risk_parity_weights,
)

from analytics import (
    performance_summary,
)


INITIAL_CAPITAL = 100.0

LOOKBACK_DAYS = 252

TRANSACTION_COST_BPS = 10

TRANSACTION_COST_RATE = (
    TRANSACTION_COST_BPS
    / 10000
)


STRATEGIES = {
    "Equal Weight":
        equal_weight_weights,

    "Inverse Volatility":
        inverse_volatility_weights,

    "Risk Parity":
        risk_parity_weights,
}


def run_strategy_backtest(
    returns,
    strategy_function,
    lookback_days=LOOKBACK_DAYS,
    transaction_cost_rate=TRANSACTION_COST_RATE,
):
    """
    Monthly rebalanced portfolio backtest.

    Portfolio weights are estimated using only historical
    information available before each rebalance date.

    Transaction costs are charged according to portfolio
    turnover.
    """

    returns = (
        returns
        .copy()
        .fillna(0.0)
    )

    dates = returns.index

    portfolio_returns = pd.Series(
        0.0,
        index=dates,
        dtype=float,
    )

    turnover_series = pd.Series(
        0.0,
        index=dates,
        dtype=float,
    )

    weights_history = pd.DataFrame(
        index=dates,
        columns=returns.columns,
        dtype=float,
    )

    current_weights = None

    previous_month = None

    for i, date in enumerate(
        dates
    ):

        current_month = (
            date.to_period("M")
        )

        should_rebalance = (
            previous_month is None
            or current_month
            != previous_month
        )

        # Require a full historical estimation window.
        if (
            should_rebalance
            and i >= lookback_days
        ):

            historical_returns = (
                returns.iloc[
                    i - lookback_days:i
                ]
            )

            target_weights = (
                strategy_function(
                    historical_returns
                )
            )

            target_weights = (
                target_weights
                .reindex(
                    returns.columns
                )
                .fillna(0.0)
            )

            target_weights = (
                target_weights
                / target_weights.sum()
            )

            if current_weights is None:

                turnover = (
                    target_weights
                    .abs()
                    .sum()
                )

            else:

                turnover = (
                    target_weights
                    - current_weights
                ).abs().sum()

            transaction_cost = (
                turnover
                * transaction_cost_rate
            )

            current_weights = (
                target_weights.copy()
            )

            turnover_series.loc[
                date
            ] = turnover

        else:

            transaction_cost = 0.0

        if current_weights is None:

            previous_month = (
                current_month
            )

            continue

        weights_history.loc[
            date
        ] = current_weights

        daily_asset_returns = (
            returns.loc[
                date
            ]
        )

        gross_return = (
            current_weights
            * daily_asset_returns
        ).sum()

        net_return = (
            gross_return
            - transaction_cost
        )

        portfolio_returns.loc[
            date
        ] = net_return

        # Allow weights to drift after market movement.
        post_return_weights = (
            current_weights
            * (
                1
                + daily_asset_returns
            )
        )

        total = (
            post_return_weights
            .sum()
        )

        if total > 0:

            current_weights = (
                post_return_weights
                / total
            )

        previous_month = (
            current_month
        )

    # Remove pre-investment period.
    invested = (
        weights_history
        .notna()
        .any(axis=1)
    )

    portfolio_returns = (
        portfolio_returns[
            invested
        ]
    )

    turnover_series = (
        turnover_series[
            invested
        ]
    )

    weights_history = (
        weights_history.loc[
            invested
        ]
    )

    return {
        "returns":
            portfolio_returns,

        "turnover":
            turnover_series,

        "weights":
            weights_history,
    }


def wealth_index(
    returns,
    initial_capital=INITIAL_CAPITAL,
):
    """
    Convert returns into portfolio value.
    """

    return (
        (
            1
            + returns.fillna(0)
        )
        .cumprod()
        * initial_capital
    )


if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "PORTFOLIO BACKTESTING FRAMEWORK"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print(
        "Loading current S&P 500 universe..."
    )

    tickers = (
        get_sp500_tickers()
    )

    print(
        f"Constituents found: "
        f"{len(tickers)}"
    )

    print()
    print(
        "Downloading historical prices..."
    )

    prices = download_prices(
        tickers,
        START_DATE,
        END_DATE,
    )

    eligible_assets = (
        select_eligible_assets(
            prices
        )
    )

    print(
        f"Eligible assets: "
        f"{len(eligible_assets)} "
        f"/ {len(tickers)}"
    )

    prices = clean_prices(
        prices,
        eligible_assets,
    )

    returns = calculate_returns(
        prices
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print()
    print(
        "Downloading S&P 500 benchmark..."
    )

    (
        benchmark_prices,
        benchmark_returns,
    ) = download_benchmark()

    # --------------------------------------------------------
    # STRATEGIES
    # --------------------------------------------------------

    strategy_results = {}

    summaries = {}

    for (
        strategy_name,
        strategy_function,
    ) in STRATEGIES.items():

        print()
        print(
            f"Running: "
            f"{strategy_name}"
        )

        result = (
            run_strategy_backtest(
                returns,
                strategy_function,
            )
        )

        strategy_results[
            strategy_name
        ] = result

        summaries[
            strategy_name
        ] = (
            performance_summary(
                result["returns"]
            )
        )

    # --------------------------------------------------------
    # ALIGN BENCHMARK
    # --------------------------------------------------------

    first_strategy = (
        next(
            iter(
                strategy_results
                .values()
            )
        )
    )

    strategy_dates = (
        first_strategy[
            "returns"
        ].index
    )

    benchmark_returns = (
        benchmark_returns
        .reindex(
            strategy_dates
        )
        .fillna(0.0)
    )

    strategy_results[
        "S&P 500"
    ] = {
        "returns":
            benchmark_returns
    }

    summaries[
        "S&P 500"
    ] = (
        performance_summary(
            benchmark_returns
        )
    )

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    performance_table = (
        pd.DataFrame(
            summaries
        )
        .T
    )

    print()
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    display_table = (
        performance_table.copy()
    )

    for column in [
        "Annualized Return",
        "Annualized Volatility",
        "Maximum Drawdown",
    ]:

        display_table[
            column
        ] = (
            display_table[
                column
            ]
            * 100
        )

    print()
    print(
        display_table.round(2)
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_directory = Path(
        "results"
    )

    results_directory.mkdir(
        exist_ok=True
    )

    performance_table.to_csv(
        results_directory
        / "performance_summary.csv"
    )

    for (
        strategy_name,
        result,
    ) in strategy_results.items():

        safe_name = (
            strategy_name
            .lower()
            .replace(" ", "_")
            .replace("&", "and")
        )

        wealth = wealth_index(
            result["returns"]
        )

        wealth.to_csv(
            results_directory
            / f"{safe_name}_wealth.csv"
        )
    # --------------------------------------------------------
    # SAVE RISK PARITY WEIGHTS
    # --------------------------------------------------------

    risk_parity_weights_history = (
        strategy_results[
            "Risk Parity"
        ]["weights"]
    )

    risk_parity_weights_history.to_csv(
        results_directory
        / "risk_parity_weights.csv"
    )
    print()
    print(
        "Backtest complete."
    )

    print(
        "Results saved in /results."
    )

    # Make results available to visualization.py
    combined_returns = pd.DataFrame({
        name: result["returns"]
        for name, result
        in strategy_results.items()
    })

    combined_returns.to_csv(
        results_directory
        / "strategy_returns.csv"
    )