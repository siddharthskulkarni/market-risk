# Market Risk Reporting

Measure and backtest **VaR** and **Expected Shortfall (ES)** for multi-asset portfolios (equities, Treasury yields, gold/silver ETFs) using historical and parametric methods, **DCC-GARCH**, and regulatory-style backtests (Kupiec, Christoffersen, Basel traffic light) with rolling out-of-sample evaluation.

## Project layout

| File | Purpose |
|------|---------|
| [`baseline.ipynb`](baseline.ipynb) | Fetch data → ingest aligned panel → portfolio VaR/ES → DCC-GARCH → rolling OOS backtests |
| [`workflow.ipynb`](workflow.ipynb) | Single-asset **GLD** demo (5y TradingView CSV): VaR, ES, Kupiec, Christoffersen, Basel |
| [`risk_utils.py`](risk_utils.py) | Shared backtest and VaR/ES helpers used by both notebooks |
| [`data/`](data/) | Cached market data and processed return panels |

## Setup

```bash
cd market-risk
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add MASSIVE_API_KEY (Polygon/Massive)
```

Optional: `FRED_API_KEY` for Treasury yield backup in `baseline.ipynb`.

## How to run

1. **`baseline.ipynb`** — run top to bottom:
   - **Fetch** (requires `MASSIVE_API_KEY`): equities/ETFs + Treasury yields (~2 years).
   - **Ingest**: writes `data/portfolio_log_returns_2y.csv`, `data/portfolio_pf_log_returns_2y.csv`, `data/data_manifest.json`.
   - **Risk**: in-sample VaR/ES; DCC-GARCH (reduced universe) with GARCH-t portfolio fallback.
   - **Backtest**: rolling 252-day estimation, Kupiec / Christoffersen / Basel / McNeil–Frey ES → `data/portfolio_backtest_summary.csv`.

2. **`workflow.ipynb`** — GLD 5y path using `data/GLD_5yrs_historical_prices.csv` (manual TradingView export is fine).

## Data sources

- **Equities / ETFs**: Polygon.io via `massive` (`adjusted=true` daily bars).
- **Treasury yields**: Polygon `list_treasury_yields`; optional **FRED** (`DGS1`, `DGS5`, `DGS10`) fill for gaps.
- **GLD 5y**: TradingView CSV (`Date`, `Close/Last`, …) for `workflow.ipynb` only.

Deprecated snapshots (not used by ingest): see [`data/DEPRECATED_README.md`](data/DEPRECATED_README.md).

## CSV schemas

**Raw stocks:** `date,ticker,open,high,low,close,volume`  
**Raw bonds:** `date,ticker,rate`  
**Panel:** `date` + 20 asset columns (17 stocks/ETFs + 3 yields)  
**Portfolio returns:** `date,log_return`

## Limitations (document in reports)

- S&P 15-name subset from Wikipedia is **not** point-in-time.
- ~2-year history limits rolling backtest length after a 252-day warmup.
- Full 20-asset DCC may use a **reduced universe**; portfolio **GARCH-t** is the documented fallback.
- `workflow.ipynb` uses simple returns → `log(1+r)`; `baseline.ipynb` uses `log(price ratio)` for the multi-asset panel.

## Outputs

- `images/` — diagnostic and VaR plots
- `data/portfolio_backtest_summary.csv` — OOS test results
- `data/data_manifest.json` — reproducibility metadata
