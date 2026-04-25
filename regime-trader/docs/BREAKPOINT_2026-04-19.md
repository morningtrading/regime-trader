# Breakpoint — 2026-04-19

Resume from here. Context below is self-contained.

## Current best on indices (full period 2018-2026, 12 folds OOS, 10 bps slippage)

| Strategy                 | Return  | Sharpe | MaxDD   | Vol    |
|--------------------------|---------|--------|---------|--------|
| **SMA-200 (EW) bench**   | +70.55% | **0.543** | -12.58% | 8.84%  |
| Buy & Hold               | +97.90% | 0.470  | -31.54% | 18.16% |
| **HMM + vix_zscore_60**  | **+78.08%** | 0.446 | -22.90% | 13.54% |
| HMM extended (no VIX)    | +56.64% | 0.263  | -31.02% | 16.62% |
| EMA 9/45                 | +52.52% | 0.313  | -13.68% | 9.59%  |

**Gap vs SMA-200: 0.097 Sharpe points.** HMM wins on return, loses on vol/DD.

## What's locked in (committed)

- VIX feature: `vix_zscore_60` only (vix_level was toxic), enabled by default
  - `use_vix_features: true` in settings.yaml
  - yfinance ^VIX primary + Alpaca VXX fallback in `data/vix_fetcher.py`
- Benchmark fairness: period correctly aligned via `.reindex(eq_idx)` before `analyze_equity_curve`
- Slippage (10 bps) applied uniformly to HMM + all benchmarks
- Feature-config bugs fixed: `features_override` + `extended_features` + `use_vix_features` now propagated end-to-end in `main.py` (3 hmm_config build sites) and whitelisted engine kwargs in `backtester.py` (2 call sites)
- Timeframe display label fixed (was showing misleading `broker.timeframe=5Min`, now `1Day bars (backtest)`)
- Selection bias proven: midcap (+411→+34 neutral), stocks (+108→+48 neutral); warnings added to 6 non-indices groups
- Feature ablation (2020+ run): all 3 extended features are additive (no redundancy)

## Opt-in tools (committed, disabled by default)

- `strategy.sma_hard_gate` — force flat when SPY < SMA200. Destroys signal on indices (0.446 → 0.273). Kept for extreme DD aversion.
- `strategy.sma_soft_mult` — scale LONG sizes when < SMA200:
  - 1.0 (default): disabled
  - 0.7: Sharpe 0.442 / MaxDD -19.5% (trade -0.004 Sharpe for -3pp DD)
  - 0.5: Sharpe 0.425 / MaxDD -17.3%

## What failed (dead ends — don't retry)

- `vix_level` feature (non-stationary, toxic when paired with vix_zscore_60)
- `sma_hard_gate` at full effect (whipsaw double-filter)
- `min_confidence` sweep 0.62/0.68/0.75 — plateau, posteriors already high-conviction

## Next steps (not started)

In order of expected ROI for closing the Sharpe gap:

1. **Credit spread z-score (HYG/LQD ratio rolling 60)** — orthogonal regime signal, same template as VIX. ETFs available on Alpaca. Implementation pattern: mirror `vix_fetcher.py` + feature_engineering hook.
2. **Volatility targeting** — scale positions to target fixed annualized vol (e.g. 10%). Most direct path to Sharpe: HMM return is already > SMA, so shrinking vol from 13.5% toward 9% should flip the Sharpe ordering. Implement in `core/regime_strategies.py::generate_signals` just before rebalance filter: `scale = target_vol / realized_vol_20; scale = clip(scale, 0.25, 1.5)`.
3. **Trailing portfolio-level DD stop** — reduce exposure to 30% if equity drops -10% from peak until recovery.
4. **Per-group params** — plan exists at `docs/per_group_params_plan.md`. Likely needed after #1-3 to generalize beyond indices.
5. **Dollar strength (UUP) z-score** — additional cross-asset signal, same pattern as HYG/LQD.

## Files modified this session

- `data/vix_fetcher.py` (new)
- `data/feature_engineering.py` — HMM_EXTENDED_VIX_FEATURES preset, vix_series threading
- `core/regime_strategies.py` — sma_hard_gate, sma_soft_mult overlays
- `core/asset_groups.py` — warning field on AssetGroup
- `config/asset_groups.yaml` — added midcap_neutral, stocks_neutral + warnings on 6 groups
- `config/settings.yaml` — use_vix_features, sma_hard_gate, sma_soft_mult, display fixes
- `config/sets/balanced.yaml` — min_confidence comment
- `main.py` — _maybe_fetch_vix helper, 3 hmm_config propagation fixes, display fixes
- `backtest/backtester.py` — _maybe_fetch_vix_bt helper, engine-kwargs whitelist (x2 sites)
- `backtest/performance.py` — slippage threaded into all 4 benchmark families (bnh/sma/ema_cross/random) × (single/multi)
- `scripts/ablation_features_indices.py` (new)
- `scripts/ablation_vix_indices.py` (new)
- `docs/per_group_params_plan.md` (new, earlier session)
- `questions.md` — answered mandatory files + hardcoded paths audit

## How to resume

Read this file + `questions.md`, then either:
- Pick item 1 (credit spread) — follow the `data/vix_fetcher.py` + `feature_engineering.py` pattern.
- Or pick item 2 (vol targeting) — edit `core/regime_strategies.py::RegimeStrategyOrchestrator.generate_signals`.

Run with `cd regime-trader && PYTHONIOENCODING=utf-8 py -3.12 main.py backtest --asset-group indices --compare`.
Current HEAD: `d5e0c21` (SMA overlay + min_confidence sweep).
