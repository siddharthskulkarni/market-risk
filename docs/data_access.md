# Data access

## Environment variables

| Variable | Required | Used by |
|----------|----------|---------|
| `MASSIVE_API_KEY` | For live fetch | `PolygonEquitySource`, `PolygonTreasurySource` |
| `FRED_API_KEY` | Optional | `FredTreasurySource` (higher rate limits) |

Copy [`.env.example`](../.env.example) to `.env` at the repo root.

## Polygon / Massive (primary)

- **Equities / ETFs:** daily bars, `adjusted=true`
- **Treasury:** `list_treasury_yields` → `1YR`, `5YR`, `10YR`

CLI:

```bash
ingest-market-data --lookback 720 --fred-backup
# or rebuild panel only:
ingest-market-data --skip-fetch
```

## FRED (backup)

| Label | Series |
|-------|--------|
| 1YR | DGS1 |
| 5YR | DGS5 |
| 10YR | DGS10 |

Rows present only in FRED are appended to the bond CSV; overlapping dates prefer Polygon.

## TradingView (manual)

Used by `examples/single_asset_gld.ipynb` only.

**File:** `data/GLD_5yrs_historical_prices.csv`  
**Columns:** `Date`, `Close/Last`, optional `Open`, `High`, `Low`, `Volume`

Loader: `market_risk.load_tradingview_csv(path, ticker)`

## Committed cache (`data/`)

| File | Schema |
|------|--------|
| `portfolio_stocks_2yrs_hist_dret.csv` | `date,ticker,open,high,low,close,volume` |
| `portfolio_bonds_2yrs_hist_dret.csv` | `date,ticker,rate` |
| `portfolio_log_returns_2y.csv` | `date` + 20 asset columns |
| `portfolio_pf_log_returns_2y.csv` | `date,log_return` |
| `data_manifest.json` | ingest metadata |

Deprecated: see `data/DEPRECATED_README.md`.
