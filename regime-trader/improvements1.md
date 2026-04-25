Based on my review of the regime-trader codebase (including main.py, core/hmm_engine.py, core/risk_manager.py, core/signal_generator.py, README.md, and LESSONS_LEARNED.md), here is a prioritized list of 20 bugs, improvements, and simplifications. No code has been modified.

🐛 Critical Bugs & Edge Cases
1. Daylight Saving Time (DST) Bug in main.py
_next_bar_close_utc hardcodes the NYSE market close as 21:00 UTC (16:00 ET). During Eastern Daylight Time (EDT), the market actually closes at 20:00 UTC. This will cause timing mismatches, delayed orders, and incorrect end-of-day calculations for half the year. Fix: Use pandas_market_calendars or pytz to dynamically calculate market hours.

2. Hardcoded Market Holiday Ignorance
In _next_bar_close_utc, today_close.weekday() >= 5 is used to skip weekends. It completely ignores market holidays (e.g., Christmas, Thanksgiving), which will cause the system to expect bars and potentially fail or timeout on non-trading days.

3. Fractional Shares Truncation in RiskManager
compute_position_size forces float(int(weight_based_shares)) to round down to whole shares. Because Alpaca supports fractional shares for equities and crypto, this arbitrary truncation restricts precision and leaves capital uninvested.

4. String-Parsing Asset Type Guesser (main.py)
_fetch_prices uses a hardcoded _FIAT_CURRENCIES set and checks for a / in the ticker to guess if an asset is Crypto or Forex. This is brittle. If Alpaca adds a new asset class or a ticker happens to have a slash (e.g., warrants/preferred stock like BRK/B), it will route to the wrong API client.

🏗️ Architectural Improvements
5. Massive main.py Monolith
At over 3,200 lines and 143KB, main.py violates the single-responsibility principle. Simplification: Split it into modular components: cli.py, commands/backtest.py, commands/trade.py, and data/data_fetcher.py.

6. Duplicate Risk Logic
RiskManager contains both validate_signal() (for live trading) and check_trade() (for walk-forward backtesting). These two methods duplicate complex logic (exposure caps, concurrent checks, circuit breakers). They should be refactored to use the exact same underlying mathematical functions to prevent live vs. backtest divergence.

7. Unsafe Model Serialization
HMMEngine.save() uses Python's built-in pickle. Pickles can execute arbitrary code upon loading and are often fragile across Python versions. Improvement: Extract the underlying NumPy arrays (means, covariances, transition matrices) and save them via .npz or safetensors.

8. Missing Short Selling / Inverse ETF Support
The strategy is strictly long-only (as noted in the README). Adding the ability to allocate to inverse ETFs (e.g., SH for SPY) or explicitly shorting during CRASH and STRONG_BEAR regimes would massively improve downside protection and edge.

9. Static Risk Circuit Breakers
RiskManager circuit breakers use static percentages (e.g., daily_halt = 3%). Improvement: Make these volatility-adjusted (e.g., 2.5x the portfolio's ATR). A 3% drop in a low-volatility regime is catastrophic, but normal in a high-volatility regime.

10. Config Parsing & Validation
load_config uses a custom recursive dictionary merge (_deep_merge). Improvement: Migrate settings.yaml parsing to Pydantic. This would grant strict schema validation, type checking, and automatic default injection.

11. UI and Display Logic inside Core Files
main.py has embedded Rich console tables (_print_comparison_table, _print_run_config). This UI/terminal code should be decoupled entirely into monitoring/reporting.py.

12. Hardcoded Python 3.12 Constraint
The README relies on the Windows-specific py -3.12 alias to bypass hmmlearn wheel issues on 3.13. Improvement: Migrate from requirements.txt to uv or poetry with a strict requires-python = "==3.12.*" constraint to make cross-platform setup foolproof.

🚀 Code Simplifications & Performance
13. Synthesizing OHLCV from Close (SignalGenerator)
To feed the strategy, the signal generator artificially creates open/high/low columns using close.shift(1). If the underlying strategy logic only requires close prices, passing a simple pd.Series or close-only DataFrame would reduce memory overhead and simplify logic.

14. Vectorize Emission Probabilities (HMMEngine)
HMMEngine._log_emission_probs loops through HMM states sequentially using for k in range(K): to compute multivariate_normal.logpdf. This could be vectorized across the state dimension, which would accelerate inference during heavy walk-forward backtests.

15. Cleaner Divide-by-Zero Handling
LESSONS_LEARNED.md highlights using np.errstate(invalid="ignore") to suppress divide-by-zero warnings. Simplification: A mathematically safer and cleaner approach without context managers is totals / np.where(counts > 0, counts, 1).

16. Simplify Logging Handlers
The 4-file JSON logging pattern uses overly complex subclassed logging.Filter injections. Simplification: Switching to structlog or loguru would eliminate the boilerplate required for JSON routing and context injection.

17. Eliminate Global State Warnings
_WARNED_GROUPS is a module-level global set in main.py used to prevent duplicate console warnings. Simplification: Encapsulate this into a stateful logger class or adapter.

18. Centralize Metric String Formatting
LESSONS_LEARNED.md details a bug where +100.63% was formatted as +10063.00%. To prevent this from regressing elsewhere, all numeric/percentage formatting should be moved to a single utils.formatters module rather than scattered inline f-strings.

19. Optimise Forward Algorithm Inference
HMMEngine.predict_regime_filtered appends object states iteratively into a Python list. Pre-allocating the results array or caching properties would reduce memory reallocation overhead when parsing 100,000+ historical bars.

20. Synchronizing Current Weights Safely
In SignalGenerator.update_current_weights, it blindly updates its internal dictionary. If a new symbol is traded out-of-band or an API call fails, the weights might permanently drift from the broker's reality. Improvement: Add a reconciliation loop that queries Alpaca's actual position endpoint every N minutes to true-up the internal state.

4:16 PM
