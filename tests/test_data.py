from pathlib import Path

import pandas as pd

from market_risk.data.tradingview import load_tradingview_csv


def test_load_tradingview_schema():
    path = Path(__file__).parent / "fixtures" / "gld_prices_sample.csv"
    if not path.exists():
        return
    df = load_tradingview_csv(path, "GLD")
    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert (df["ticker"] == "GLD").all()
