import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed CAC 40 universe used for the project.
#
# IMPORTANT:
# This is a current fixed universe, not a point-in-time
# historical universe. Therefore, the backtest is subject
# to survivorship bias.
CAC40_TICKERS = [
    "AC.PA",       # Accor
    "AI.PA",       # Air Liquide
    "AIR.PA",      # Airbus
    "MT.AS",       # ArcelorMittal
    "CS.PA",       # AXA
    "BNP.PA",      # BNP Paribas
    "EN.PA",       # Bouygues
    "BVI.PA",      # Bureau Veritas
    "CAP.PA",      # Capgemini
    "CA.PA",       # Carrefour
    "ACA.PA",      # Credit Agricole
    "BN.PA",       # Danone
    "DSY.PA",      # Dassault Systemes
    "FGR.PA",      # Eiffage
    "ENGI.PA",     # Engie
    "EL.PA",       # EssilorLuxottica
    "ERF.PA",      # Eurofins Scientific
    "ENX.PA",      # Euronext
    "RMS.PA",      # Hermes
    "KER.PA",      # Kering
    "OR.PA",       # L'Oreal
    "LR.PA",       # Legrand
    "MC.PA",       # LVMH
    "ML.PA",       # Michelin
    "ORA.PA",      # Orange
    "RI.PA",       # Pernod Ricard
    "PUB.PA",      # Publicis
    "RNO.PA",      # Renault
    "SAF.PA",      # Safran
    "SGO.PA",      # Saint-Gobain
    "SAN.PA",      # Sanofi
    "SU.PA",       # Schneider Electric
    "GLE.PA",      # Societe Generale
    "STLAP.PA",    # Stellantis
    "STMPA.PA",    # STMicroelectronics
    "HO.PA",       # Thales
    "TTE.PA",      # TotalEnergies
    "URW.PA",      # Unibail-Rodamco-Westfield
    "VIE.PA",      # Veolia
    "DG.PA",       # Vinci
]

# CAC 40 official index used as an external benchmark.
CAC40_BENCHMARK = "^FCHI"

# Backtest period.
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

# Minimum proportion of available observations required
# for an asset to be included in the backtest universe.
MIN_COVERAGE = 0.95


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_prices(tickers, start_date, end_date):
    """
    Download historical adjusted closing prices.

    The function first attempts a batch download.
    Tickers for which the entire series is missing are
    then retried individually.

    Parameters
    ----------
    tickers : list
        Yahoo Finance tickers.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Historical adjusted closing prices.
    """

    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Make sure we always work with a DataFrame.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    failed_tickers = []

    # Identify tickers that completely failed in the batch download.
    for ticker in tickers:
        if ticker not in prices.columns or prices[ticker].isna().all():
            failed_tickers.append(ticker)

    # Retry failed tickers individually.
    for ticker in failed_tickers:

        print(f"Retrying {ticker} individually...")

        individual_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]

        if isinstance(individual_data, pd.DataFrame):

            if ticker in individual_data.columns:
                individual_data = individual_data[ticker]
            else:
                individual_data = individual_data.squeeze()

        if individual_data.notna().any():

            prices[ticker] = individual_data

            print(
                f"{ticker}: individual download successful."
            )

        else:

            print(
                f"{ticker}: no usable data found."
            )

    return prices


# ============================================================
# DATA QUALITY
# ============================================================

def check_missing_values(prices):
    """
    Count missing observations for each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.

    Returns
    -------
    pd.Series
        Number of missing observations per ticker.
    """

    return prices.isna().sum()


def data_availability_report(prices):
    """
    Build a data availability report for each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.

    Returns
    -------
    pd.DataFrame
        Report containing:
        - first valid date
        - last valid date
        - missing observations
        - available observations
        - coverage ratio
    """

    total_observations = len(prices)

    report = pd.DataFrame({
        "first_valid_date": prices.apply(
            lambda x: x.first_valid_index()
        ),
        "last_valid_date": prices.apply(
            lambda x: x.last_valid_index()
        ),
        "missing_values": prices.isna().sum(),
        "observations": prices.notna().sum(),
    })

    report["coverage"] = (
        report["observations"]
        / total_observations
    )

    return report


# ============================================================
# ASSET ELIGIBILITY
# ============================================================

def select_eligible_assets(
    prices,
    min_coverage=MIN_COVERAGE
):
    """
    Select assets with sufficient historical coverage.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.
    min_coverage : float
        Minimum required coverage ratio.

    Returns
    -------
    list
        Tickers satisfying the coverage requirement.
    """

    report = data_availability_report(prices)

    eligible_assets = report.index[
        report["coverage"] >= min_coverage
    ].tolist()

    return eligible_assets


# ============================================================
# RETURNS
# ============================================================

def calculate_returns(prices):
    """
    Calculate daily simple returns from adjusted prices.

    Formula:
        r_t = P_t / P_(t-1) - 1

    Parameters
    ----------
    prices : pd.DataFrame
        Historical adjusted closing prices.

    Returns
    -------
    pd.DataFrame
        Daily simple returns.
    """

    returns = prices.pct_change(
        fill_method=None
    )

    return returns


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Download prices
    # --------------------------------------------------------

    prices = download_prices(
        CAC40_TICKERS,
        START_DATE,
        END_DATE,
    )

    print()
    print("=" * 60)
    print("PRICE DATA")
    print("=" * 60)

    print(prices.head())

    print()
    print("Dataset shape:")
    print(prices.shape)

    # --------------------------------------------------------
    # 2. Missing values
    # --------------------------------------------------------

    missing = check_missing_values(prices)

    print()
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(missing)

    # --------------------------------------------------------
    # 3. Data availability
    # --------------------------------------------------------

    availability = data_availability_report(prices)

    print()
    print("=" * 60)
    print("DATA AVAILABILITY REPORT")
    print("=" * 60)

    print(availability.to_string())

    # --------------------------------------------------------
    # 4. Select eligible assets
    # --------------------------------------------------------

    eligible_assets = select_eligible_assets(
        prices,
        min_coverage=MIN_COVERAGE,
    )

    print()
    print("=" * 60)
    print("ELIGIBLE ASSETS")
    print("=" * 60)

    print(eligible_assets)

    print()
    print(
        f"Number of eligible assets: "
        f"{len(eligible_assets)} / "
        f"{len(CAC40_TICKERS)}"
    )

    # --------------------------------------------------------
    # 5. Identify excluded assets
    # --------------------------------------------------------

    excluded_assets = [
        ticker
        for ticker in CAC40_TICKERS
        if ticker not in eligible_assets
    ]

    print()
    print("=" * 60)
    print("EXCLUDED ASSETS")
    print("=" * 60)

    print(excluded_assets)

    # --------------------------------------------------------
    # 6. Keep only eligible assets
    # --------------------------------------------------------

    clean_prices = prices[eligible_assets].copy()

    print()
    print("=" * 60)
    print("CLEAN PRICE DATA")
    print("=" * 60)

    print(clean_prices.head())

    print()
    print("Clean dataset shape:")
    print(clean_prices.shape)

    # --------------------------------------------------------
    # 7. Calculate daily returns
    # --------------------------------------------------------

    returns = calculate_returns(
        clean_prices
    )

    print()
    print("=" * 60)
    print("DAILY RETURNS")
    print("=" * 60)

    print(returns.head())

    print()
    print("Returns shape:")
    print(returns.shape)