import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight_weights(
    historical_returns,
):
    """
    Equal allocation across all available assets.
    """

    assets = historical_returns.columns

    n_assets = len(assets)

    if n_assets == 0:
        raise ValueError(
            "No assets available."
        )

    weights = pd.Series(
        1.0 / n_assets,
        index=assets,
    )

    return weights


def inverse_volatility_weights(
    historical_returns,
):
    """
    Allocate inversely proportional to historical volatility.
    """

    volatility = (
        historical_returns.std()
    )

    volatility = volatility.replace(
        0,
        np.nan,
    )

    inverse_vol = 1.0 / volatility

    inverse_vol = inverse_vol.dropna()

    weights = (
        inverse_vol
        / inverse_vol.sum()
    )

    return weights


def risk_parity_weights(
    historical_returns,
):
    """
    Equal Risk Contribution portfolio.

    Uses the covariance matrix and numerical optimization
    to target equal contributions to portfolio volatility.
    """

    covariance = (
        historical_returns
        .cov()
        .values
    )

    assets = historical_returns.columns

    n_assets = len(assets)

    if n_assets == 0:
        raise ValueError(
            "No assets available."
        )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def risk_contributions(weights):

        portfolio_variance = (
            weights.T
            @ covariance
            @ weights
        )

        portfolio_volatility = np.sqrt(
            portfolio_variance
        )

        if portfolio_volatility == 0:
            return np.zeros(
                n_assets
            )

        marginal_risk = (
            covariance
            @ weights
        ) / portfolio_volatility

        contributions = (
            weights
            * marginal_risk
        )

        return contributions


    def objective(weights):

        contributions = (
            risk_contributions(
                weights
            )
        )

        target = (
            contributions.sum()
            / n_assets
        )

        return np.sum(
            (
                contributions
                - target
            ) ** 2
        )


    constraints = {
        "type": "eq",
        "fun": lambda w:
            np.sum(w) - 1.0,
    }

    bounds = [
        (0.0, 1.0)
        for _ in range(
            n_assets
        )
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Risk Parity optimization failed: "
            + result.message
        )

    weights = pd.Series(
        result.x,
        index=assets,
    )

    weights = (
        weights
        / weights.sum()
    )

    return weights