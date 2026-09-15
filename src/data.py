import pandas as pd
import yfinance as yf


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
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

def download_prices(tickers, start_date, end_date):
    """
    Download historical adjusted closing prices.

    First attempt: download all tickers in one batch.
    Fallback: retry individually any ticker with no usable data.

    Parameters
    ----------
    tickers : list
        List of Yahoo Finance tickers.
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

    # Make sure we always have a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    failed_tickers = []

    for ticker in tickers:
        if ticker not in prices.columns or prices[ticker].isna().all():
            failed_tickers.append(ticker)

    # Retry failed tickers individually
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
            print(f"{ticker}: individual download successful.")
        else:
            print(f"{ticker}: no usable data found.")

    return prices


def check_missing_values(prices):
    """
    Count missing values for each asset.
    """
    missing = prices.isna().sum()
    return missing



def data_availability_report(prices):
    """
    Summarize historical data availability for each asset.
    """
    report = pd.DataFrame({
        "first_valid_date": prices.apply(lambda x: x.first_valid_index()),
        "last_valid_date": prices.apply(lambda x: x.last_valid_index()),
        "missing_values": prices.isna().sum(),
        "observations": prices.notna().sum(),
    })

    return report


if __name__ == "__main__":
    prices = download_prices(
        CAC40_TICKERS,
        START_DATE,
        END_DATE,
    )

    print(prices.head())
    print()

    missing = check_missing_values(prices)
    print("Missing values:")
    print(missing)

    availability = data_availability_report(prices)

    print()
    print("Data availability report:")
    print(availability.to_string())

    print()
    print("Rows containing missing values:")
    print(prices[prices.isna().any(axis=1)])

    print()
    print("Dataset shape:")
    print(prices.shape)