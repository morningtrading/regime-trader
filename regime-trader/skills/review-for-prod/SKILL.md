---
name: review-for-prod
description: Perform a pre-production review of the trading bot before switching from paper to live trading with real money. Use this skill whenever the user mentions "go live," "switch to live," "real money," "launch," "production," "deploy," "ready to trade," or asks "is this ready?" Also use before any major release, after adding new strategies, or after significant changes to risk management. This review is non-negotiable — a bot going live unreviewed is a bot that will lose money to preventable bugs. Be thorough. Surface every risk.
---

# Pre-Production Review

Switching a trading bot from paper to live is a one-way door. When it's paper, a bug is educational. When it's live, a bug costs money — and sometimes catastrophic money if it keeps running. This skill is a structured review that surfaces the issues to fix before flipping the switch.

**The review does not approve anything.** It produces a list of findings. The human decides what to fix and whether to proceed.

## Inputs

Ask the user:

1. **Which strategy(ies)** are they planning to live-trade?
2. **What's the intended capital allocation?** (This changes the severity of bugs.)
3. **How long have they been paper trading?** (If < 30 days, immediately flag — not enough.)
4. **What's their max acceptable loss in the first month?** (Anchors the risk review.)

## Workflow

Go through each of these sections in order. For each finding, tag severity:

- 🔴 **Critical** — must fix before going live. System will lose money or lose control.
- 🟡 **Warning** — should fix. Bug that will probably not trigger immediately but will eventually.
- 🟢 **Nice-to-have** — cleanup. Not blocking.

Present findings in a structured report at the end.

---

### Section 1 — Look-ahead bias

Run the look-ahead bias test yourself. Handle the environment first — install pytest if missing, adapt paths if the project uses a different test layout, install any missing dependencies the test needs.

Command to try first:

```bash
pytest tests/test_look_ahead.py -v
```

🔴 If this doesn't pass, stop the review. Nothing else matters. The backtest is a lie.

🔴 If `tests/test_look_ahead.py` doesn't exist, create it using the `write-lookahead-test` skill before proceeding.

Then scan the codebase yourself for patterns that indicate look-ahead bias. Report any matches:

- `.predict(` in `core/` or `data/` outside of the approved `predict_regime_filtered` method
- `.shift(-` (negative shifts of any size) in `core/` or `data/`
- `center=True` inside rolling windows in `core/` or `data/`
- `.bfill(` in `core/` or `data/`
- `fit_transform(` in `core/` or `data/` when applied to the full dataset (rolling fit_transform in a loop is fine)

Use whatever scanning approach works — `grep`, `ripgrep`, reading the files directly, whatever's available. If a tool isn't installed, either install it or use an alternative. Don't block on tooling.

### Section 2 — Paper trading track record

The user must have paper-traded for at least 30 days with the exact same code they're about to deploy live.

🔴 If paper trading < 30 days: blocks live trading. No exceptions.
🔴 If code has changed since paper trading started: paper record is meaningless. Paper-trade the current code for 30 days.
🟡 If paper trading period didn't include a drawdown > 5%: the strategy hasn't been tested against real adversity.
🟡 If paper trading period didn't include a regime change: the HMM hasn't been exercised.

Compare paper trading performance to backtest performance on the same period. Flag if they differ by more than 30%:

- If paper beats backtest: data issue — paper has different fills than backtest assumes.
- If backtest beats paper: slippage is worse than modeled, or there's a look-ahead bias somewhere.

### Section 3 — Risk manager integrity

Read `core/risk_manager.py` and check:

🔴 Every signal path is forced through `validate_signal()` — no bypass.
🔴 Stop losses are required on every order (grep for `stop_loss` being None or missing).
🔴 Circuit breakers exist for: daily DD, weekly DD, peak DD.
🔴 Peak DD breaker writes a lock file requiring manual deletion.
🔴 Max risk per trade is enforced (default 1%).
🔴 Max exposure is enforced (default 80%).
🔴 Max leverage is enforced (default 1.25x).

Run the risk tests yourself:

```bash
pytest tests/test_risk.py -v
```

Install pytest if missing, adapt paths as needed. Report which tests pass and fail.

🔴 Any failure blocks live trading.

### Section 4 — Credentials and security

Scan the codebase yourself for leaked credentials. Use whatever tool is available (grep, ripgrep, or just reading key files). Do all of these checks:

1. Scan all `.py` files for hardcoded API keys. Patterns to flag:
   - Any line matching `ALPACA_API_KEY =` (or equivalent for other brokers) with a literal string on the right
   - Any line with `api_key =` or `secret =` followed by a quoted string that isn't empty
   - Any key-looking token starting with `pk_`, `sk_`, `AKIA`, or similar

2. Confirm `.env` is in `.gitignore`. Read `.gitignore` directly.

3. Scan git history for previously-committed credentials. First check if the project is a git repo at all (`.git/` directory exists). If not, skip this check and note it — fresh local projects have nothing to scan. If yes, check for commits (`git rev-list --count HEAD` — 0 means no commits yet). If there are commits, use `git log -p` to search for api key / secret patterns across history. Report any matches.

Report results in plain English — flag every suspicious match you find, don't just report counts.

🔴 Any hardcoded API key = blocks live trading. Tell the user to rotate the key immediately, it's compromised.
🔴 `.env` not in `.gitignore` = blocks live trading.
🔴 Any leaked key in git history = blocks live trading, regardless of whether it's still valid. Rotate and tell the user to consider the git history compromised.
🟡 Ask the user: does the broker API key have "trade" permission but NOT "withdraw" permission? (If they can't answer, have them check broker account settings before going live.)
🟢 Recommend storing keys in a secrets manager in production (not just `.env`).

### Section 5 — The live trading switch

Open `broker/alpaca_client.py` (or equivalent for the broker in use) and confirm:

🔴 Default is paper trading (`paper_trading: true`).
🔴 Switching to live requires an explicit config change AND a runtime confirmation prompt.
🔴 The confirmation prompt is not bypassable with a flag.

Specifically:

```python
if not paper_trading:
    confirmation = input("⚠️  LIVE TRADING MODE. Type 'YES I UNDERSTAND THE RISKS' to confirm: ")
    if confirmation != "YES I UNDERSTAND THE RISKS":
        raise SystemExit("Live trading not confirmed.")
```

🔴 If this prompt is commented out, skipped, or bypassable — blocks live trading.

### Section 6 — Logging and monitoring

Open `monitoring/logger.py` and verify:

🟡 Structured JSON logging (not just free-text).
🟡 Separate log streams for trades, errors, regime changes, risk decisions.
🟡 Log rotation configured (10MB, 30 days is a reasonable default).
🟡 Every log entry includes: `timestamp`, `regime`, `equity`, `daily_pnl`.

Open `monitoring/alerts.py` and verify:

🟡 Alerts configured for: circuit breaker fires, regime changes, data feed down, API errors.
🟡 Alerts are rate-limited (default: 1 per event type per 15 minutes).
🟡 At least one alert channel is configured and tested (email, webhook, SMS).

🔴 If the user can't prove the alert system works (they haven't tested a real alert end-to-end), blocks live trading. The one alert that matters is the one that fires at 3am during a flash crash.

### Section 7 — Recovery

🔴 `state_snapshot.json` is saved on SIGINT/SIGTERM.
🔴 On startup, the bot reads `state_snapshot.json` and reconciles against actual broker positions.
🔴 If reconciliation fails (broker has positions the bot doesn't know about), the bot halts and alerts.

Simulate: kill the process with `SIGKILL`, restart, and verify it correctly picks up state. If it double-enters a position or ignores an open position, critical bug.

### Section 8 — Error handling

Read through `main.py` and the live trading loop. Verify:

🟡 API errors retry with exponential backoff (3 retries default).
🟡 WebSocket disconnects reconnect automatically.
🟡 Unhandled exceptions log the full traceback AND save state AND send an alert.
🟡 The bot does NOT continue trading after an unhandled exception in a critical path.

🔴 A "catch-all exception that logs and continues" is a red flag. Trading should halt, not retry blind.

### Section 9 — Position size sanity

Compute, for the intended initial capital:

- Max position size in dollars (should be 15% of portfolio)
- Max daily loss before circuit breaker (should be 3% of portfolio)
- Max drawdown before emergency halt (should be 10% of portfolio)

Tell the user these numbers explicitly:

> At $100,000 starting capital:
> - Max single position: $15,000
> - Circuit breaker halt at daily loss: $3,000
> - Emergency halt at drawdown: $10,000
>
> Are you comfortable losing $10,000 in the worst case? If no, reduce starting capital.

🔴 If the worst-case loss exceeds what the user said they can stomach (from Step 1 inputs), blocks live trading.

### Section 10 — Dry run the live config

Run the dry-run mode against what would be the live configuration. Handle dependencies — install anything missing, adapt to the project's actual CLI.

Command to try first:

```bash
python main.py --dry-run --config config/settings-live.yaml
```

Watch the output for at least several minutes (longer if the user is willing). Confirm:

- Data feeds stay connected
- Signals generate
- Risk manager behaves correctly (approves, rejects, modifies as expected)
- Orders are formatted correctly (inspected in logs, not sent)
- Dashboard updates
- No unhandled errors or warnings

🔴 Any error during dry-run blocks live trading until fixed. Report specific errors and likely causes.

---

## Output the report

Produce a report in this format:

```markdown
# Pre-Production Review — {date}

**Strategy:** {name}
**Intended capital:** ${amount}
**Paper trading days:** {N}
**Reviewer:** Claude

## Summary
{N} critical findings | {N} warnings | {N} nice-to-haves

## 🔴 Critical (blocks live trading)

1. {finding}
   **File/location:** {where}
   **Fix:** {what to do}

2. ...

## 🟡 Warnings (should fix)

1. ...

## 🟢 Nice-to-have (cleanup)

1. ...

## Recommendation

{Based on findings:}
- If ANY critical findings: "Do NOT go live. Fix critical findings and re-run this review."
- If only warnings: "You can go live, but you're accepting known risks. Here's what they are: ..."
- If clean: "The review did not find blocking issues. However, this is not a guarantee of profitability or safety. Start small."
```

## What this review cannot catch

Be explicit with the user about the limits of this review:

- This review does not evaluate whether the strategy is profitable. A bug-free bot losing money is still losing money.
- This review does not catch bugs in logic that tests don't cover. Write more tests.
- This review does not predict market regime changes that break the strategy's assumptions.
- This review does not replace competent human judgment. The human is responsible for the decision to go live.

## Do not

- Do not approve live trading. The review produces findings; the human decides.
- Do not minimize findings. If something is critical, call it critical.
- Do not accept "it's fine" for critical findings. Insist on the fix.
- Do not review and approve the same session as flipping to live. Sleep on it, re-review with fresh eyes.
