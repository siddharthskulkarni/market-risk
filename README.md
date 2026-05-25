# market-risk

Python library for **multi-asset market risk** (UMass): historical and parametric **VaR** / **ES**, **DCC-GARCH**, and backtests (Kupiec, Christoffersen, Basel traffic light) with rolling out-of-sample evaluation.

**Installable package:** `market_risk` (v0.2.0)

---

## Repository structure

```
market-risk/
├── market_risk/           # Library
│   ├── data/              # Polygon, FRED, TradingView sources
│   ├── models/            # GARCH / DCC
│   ├── portfolio.py       # Return panel ingest
│   ├── var.py             # VaR / ES
│   ├── backtest.py        # Regulatory tests, rolling OOS
│   └── viz.py             # Matplotlib figures
├── data/                  # Committed CSV cache (offline runs)
├── examples/              # Notebooks (orchestration only)
├── scripts/               # ingest_market_data.py
├── docs/data_access.md
└── tests/
```

---

## Quickstart

```bash
cd market-risk
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[data,models,viz,dev]"
cp .env.example .env   # MASSIVE_API_KEY for live fetch
```

**Portfolio workflow** (uses cached `data/` by default):

```bash
jupyter notebook examples/portfolio_risk.ipynb
```

**Refresh market data:**

```bash
ingest-market-data --lookback 720 --fred-backup
```

**Single-asset GLD demo:**

```bash
jupyter notebook examples/single_asset_gld.ipynb
```

**Tests:**

```bash
pytest
```

---

## Public API

```python
from market_risk import (
    ALPHA,
    build_portfolio_panel,
    historical_var_es,
    parametric_var_es_t,
    kupiec_test,
    christoffersen_conditional_coverage,
    basel_traffic_light,
    rolling_var_backtest,
    PolygonEquitySource,
    default_sp500_tickers,
)
```

See [`docs/data_access.md`](docs/data_access.md) for CSV layouts and env vars.

---

## Examples

| Notebook | Description |
|----------|-------------|
| [`examples/portfolio_risk.ipynb`](examples/portfolio_risk.ipynb) | 20-asset panel, VaR/ES, DCC-GARCH, rolling backtests |
| [`examples/single_asset_gld.ipynb`](examples/single_asset_gld.ipynb) | GLD 5y TradingView CSV, VaR/ES, coverage tests |

Legacy root notebooks `baseline.ipynb` / `workflow.ipynb` are removed; use `examples/` instead.

---

## Dependency extras

| Extra | Purpose |
|-------|---------|
| *(core)* | `numpy`, `scipy` |
| `[data]` | Polygon, FRED, pandas |
| `[models]` | `arch` (GARCH) |
| `[viz]` | `matplotlib` |
| `[dev]` | `pytest`, `ruff` |

---

## Limitations

- S&P 15-name subset (Wikipedia) is not point-in-time.
- ~2y history limits rolling backtest length after 252-day warmup.
- DCC uses a **reduced asset universe**; **GARCH-t on the portfolio** is the documented fallback.
- GLD notebook uses `log(1+r)`; the portfolio panel uses `log(price ratio)` per asset.

---

## License

MIT — see [LICENSE](LICENSE).
