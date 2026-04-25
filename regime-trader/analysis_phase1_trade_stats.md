# Phase 1: Trade Statistics & Slippage Analysis

**Baseline:** hmm_regime strategy, 2020-2026, walk-forward backtest

---

## Slippage & Spread Impact

| Metric | Value |
|--------|-------|
| **Total slippage cost** | $4,987.52 |
| **Total position changes** | 1,421 |
| **Avg slippage per change** | $3.51 |
| **Gross profit** | $137,098.50 |
| **Slippage as % of profit** | **3.638%** |
| **Assessment** | **MODERATE** (borderline; <5% is acceptable) |

### Slippage Sensitivity

- **If spreads double (10bp):** Return degradation = **3.64%** → Final equity ~$127,123
- **If spreads halve (2.5bp):** Return improvement = **1.82%** → Final equity ~$134,605
- **Implication:** Strategy is moderately sensitive to execution costs

---

## Trade Frequency by Regime

| Regime | Trades | Frequency |
|--------|--------|-----------|
| NEUTRAL | 341 | 24.0% |
| BEAR | 349 | 24.6% |
| BULL | 255 | 17.9% |
| CRASH | 253 | 17.8% |
| EUPHORIA | 223 | 15.7% |

**Observation:** NEUTRAL regime (lowest conviction) generates the MOST trades — potential overfitting?

---

## Regime Breakdown: Profitability

| Regime | Days | Cumulative Return | Daily Avg | Win Rate |
|--------|------|-------------------|-----------|----------|
| **BEAR** | 249 | **+53.27%** | +0.214% | 51.0% |
| **EUPHORIA** | 162 | **+32.99%** | +0.204% | 56.2% |
| **BULL** | 229 | +17.02% | +0.074% | 51.1% |
| **CRASH** | 196 | +7.94% | +0.041% | 51.0% |
| **NEUTRAL** | 235 | **-11.85%** | **-0.050%** | 51.5% |

### Key Finding
🚨 **NEUTRAL regime is NEGATIVE** (−11.85% cumulative return)
- This explains why total return is only +137% instead of +200%+
- Strategy should either:
  1. **Skip** NEUTRAL regime (go to cash), OR
  2. **Tighten** entry/exit criteria to reduce false signals

---

## Asset Group Suitability

### Recommended (Tight Spreads)
- **US Stocks (SPY, QQQ, AAPL, MSFT, etc.)** — Spreads <1bp → [**OPTIMAL**]
  - Example: SPY spread = 0.5bp, QQQ spread = 0.7bp
  - Impact: Negligible

- **US Indices (DIA, IWM, IJH)** — Spreads 1-2bp → [**GOOD**]
  - Example: DIA spread = 1bp
  - Impact: +0.3-0.5% cost

- **Large-cap bonds (TLT, IEF)** — Spreads 2-3bp → [**ACCEPTABLE**]
  - Example: TLT spread = 2bp
  - Impact: 1-1.5% cost

### Risky (Wide Spreads)
- **Crypto (BTC/USD, ETH/USD)** — Spreads 5-10bp → [**RISKY**]
  - Would double slippage cost to ~7%
  - Expected return degradation: **6-12%**
  - Strategy would collapse to +100-120% (unacceptable)

- **Emerging Markets (EEM, VWO)** — Spreads 4-5bp → [**RISKY**]
  - Impact: Similar to crypto, ~5-7% degradation

- **Small-cap (IWM, SQ)** — Spreads 2-5bp → [**MIXED**]
  - IWM: 2bp (acceptable)
  - SQ (Squarespace): 3-5bp (borderline)

### Not Recommended
- **Illiquid assets** — Spreads >10bp → [**BREAKS STRATEGY**]
  - Examples: penny stocks, micro-cap, emerging micro-caps
  - Impact: Returns go negative

---

## Conclusion

**Spreads are MODERATELY critical to strategy viability.**

Optimal asset groups = **US equities + liquid indices** (current portfolio works well).

Testing recommendation:
- ✓ Test: `indices` group (SPY, QQQ, DIA, IWM, GLD, EFA)
- ✓ Test: `defensive` group (TLT, IEF, bonds) — acceptable spread cost
- ✗ Skip: `crypto` group — spreads too wide, returns would collapse
- ~ Consider: Mixed (stocks + bonds) for lower correlation
