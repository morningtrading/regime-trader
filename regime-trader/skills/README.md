# AI Pathways — Trading Bot Skill Pack

Six production-grade Claude Code skills built specifically for algorithmic trading projects. They teach Claude how to build, test, and safely deploy trading bots — the right way, with look-ahead bias checks, walk-forward backtesting, adapter-pattern broker integrations, and a pre-production review before any real money hits the market.

## What's included

| Skill | What it does |
|---|---|
| `backtest-strategy` | Runs walk-forward backtest with proper metrics, benchmarks, and honest assessment |
| `write-lookahead-test` | Generates tests that detect future data leakage (the #1 silent bug in retail backtests) |
| `add-risk-check` | Adds new risk validations to the risk manager with tests and config |
| `add-broker-adapter` | Scaffolds new broker integrations (Hyperliquid, MT5, IB, ccxt, etc.) without breaking existing code |
| `generate-walkforward-report` | Turns backtest output into a publishable report with honest framing |
| `review-for-prod` | Pre-production review surfacing every blocker before you go live with real money |

All six follow the Agent Skills open standard — they work in Claude Code, Codex CLI, Cursor, Gemini CLI, and other compatible agents without modification.

---

## Prerequisites

These skills are built for the HMM regime-based trading bot architecture (the one from the main prompt). They assume your project has roughly this structure:

```
your-bot/
├── config/settings.yaml
├── core/
│   ├── hmm_engine.py           # with predict_regime_filtered
│   ├── regime_strategies.py
│   ├── risk_manager.py
│   └── signal_generator.py
├── broker/
│   ├── base.py
│   └── {broker}_client.py
├── data/
│   ├── market_data.py
│   └── feature_engineering.py
├── backtest/
│   └── backtester.py
├── monitoring/
│   ├── logger.py
│   └── alerts.py
└── tests/
    ├── test_look_ahead.py
    ├── test_risk.py
    └── test_orders.py
```

If your layout is slightly different (e.g., `features.py` instead of `feature_engineering.py`), the skills adapt — Claude Code reads your actual file structure and adjusts. But if you're missing entire modules (no HMM, no risk manager, no walk-forward backtester), build those first using the main prompt, then install the skill pack.

**Python 3.10+** is required. The skills generate code with modern type hints (`list[int]`, `X | None`). If you're on 3.9 or older, either upgrade or ask Claude Code to rewrite generated code with `typing` module imports (`List`, `Dict`, `Optional`).

---

## Install (easiest — ask Claude Code to do it)

If you have this folder on your machine, open Claude Code in your trading bot project and say:

> "Install the AI Pathways skill pack. The files are at [path to this folder]. Copy the `.claude/skills/` directory into this project, copy the `CLAUDE.md` file to the root (or merge it with the existing one if there is one). Then confirm the skills are discoverable by listing them."

Claude Code will handle the copy, adapt to your OS, and verify the skills load. This works on Windows, macOS, and Linux equally — no need to remember `cp -r` syntax.

## Install (manual — project-scoped, recommended for teams)

This is the recommended approach if you want the skills inside your project repo so they travel with the code.

### macOS / Linux

```bash
# From the root of your trading bot project:
mkdir -p .claude/skills
cp -r /path/to/this/pack/.claude/skills/* .claude/skills/
cp /path/to/this/pack/CLAUDE.md ./CLAUDE.md
# Then edit CLAUDE.md to reflect your project specifics
```

### Windows (PowerShell)

```powershell
# From the root of your trading bot project:
New-Item -ItemType Directory -Force -Path .claude\skills
Copy-Item -Recurse -Path "C:\path\to\this\pack\.claude\skills\*" -Destination .claude\skills\
Copy-Item -Path "C:\path\to\this\pack\CLAUDE.md" -Destination CLAUDE.md
```

### Verify installation (any OS)

Either check manually:

```
.claude/skills/
  ├── add-broker-adapter/
  ├── add-risk-check/
  ├── backtest-strategy/
  ├── generate-walkforward-report/
  ├── review-for-prod/
  └── write-lookahead-test/
```

Or ask Claude Code: "list the skills in this project." It'll report back what's installed.

## Install (personal — available in every project)

If you want these skills available in every Claude Code session regardless of which project you're in, install them to the personal skills directory:

- **macOS / Linux:** `~/.claude/skills/`
- **Windows:** `%USERPROFILE%\.claude\skills\`

Easiest: ask Claude Code: "Install these skills to my personal skills directory so they're available in every project."

---

## How to use

Skills can be triggered two ways:

**1. Automatically** — Claude loads them when the description matches your request. Examples that will trigger each skill:

```
"Run a backtest on the LowVolBullStrategy over the last 5 years"
   → triggers backtest-strategy

"I just added a new feature, make sure it doesn't peek at the future"
   → triggers write-lookahead-test

"Add a rule that blocks trades during FOMC announcements"
   → triggers add-risk-check

"Can you hook this up to Hyperliquid?"
   → triggers add-broker-adapter

"Turn these backtest results into a report I can share with the community"
   → triggers generate-walkforward-report

"I think I'm ready to go live, can you review the bot?"
   → triggers review-for-prod
```

**2. Explicitly** — Call the skill directly with a slash command:

```
/backtest-strategy
/write-lookahead-test
/add-risk-check
/add-broker-adapter
/generate-walkforward-report
/review-for-prod
```

---

## Why these skills exist

Each skill bakes in a non-obvious lesson learned the hard way:

- **backtest-strategy** — Most backtests lie. This one insists on benchmarks, `--compare` flag, and honest framing. A strategy without benchmarks isn't a result, it's a story.
- **write-lookahead-test** — Every retail trading bot has look-ahead bias until proven otherwise. The test catches `model.predict()` (Viterbi) usage, `.shift(-N)`, centered rolling windows, and global `fit_transform`.
- **add-risk-check** — New rules should never bypass existing ones. This skill enforces the pattern: config-driven threshold, tested with both happy and failure paths, wired into `validate_signal()` in the right order.
- **add-broker-adapter** — The adapter pattern keeps broker-specific logic out of the core. You can swap Alpaca for Hyperliquid without touching strategy code — but only if you resist the temptation to import broker SDKs in the wrong places.
- **generate-walkforward-report** — A good report lists weaknesses. A bad one sells the strategy. This skill produces the first kind.
- **review-for-prod** — Going live is a one-way door. This review is designed to find the specific bugs that only bite when real money is at stake.

---

## Pairing with CLAUDE.md

The included `CLAUDE.md` template is the project's persistent context — it's loaded at the start of every Claude Code session and tells Claude the absolute rules (never use `model.predict()`, never submit without a stop, never default to live trading, etc.).

`CLAUDE.md` is for always-true facts. Skills are for procedures that only apply sometimes. Together they give Claude a complete understanding of your project.

Read `CLAUDE.md` in this repo for the template — edit it to match your specific codebase before committing.

---

## Troubleshooting

**Skill isn't triggering automatically.** The description field in each `SKILL.md` frontmatter controls when Claude loads the skill. If it's not triggering, either (a) the phrasing of your request doesn't match what the description expects, or (b) you can always invoke it manually with `/skill-name`.

**Skill folder structure is wrong.** Each skill must be at `.claude/skills/skill-name/SKILL.md` — exactly that structure. Common mistake: extra nesting like `.claude/skills/downloaded-pack/skill-name/SKILL.md` won't work.

**Claude ignores CLAUDE.md rules.** If the file is under 200 lines and rules are still ignored, there's usually one specific rule that isn't phrased strongly enough. Add `IMPORTANT:` or `NEVER` to the rule. The current CLAUDE.md uses this pattern already.

**Updating a skill.** Edit the SKILL.md file in place. Claude Code hot-reloads skill changes within the current session — no restart needed.

---

## Philosophy

The whole point of this pack: **risk management > signal generation, and process > prompts.**

Most people building trading bots with AI stop at "write me a trading strategy." The strategy works in backtest, fails in paper, loses money live, and they don't understand why. These skills embed the process that catches bugs BEFORE they cost money. Teaching Claude the process once means every future strategy benefits.

If Claude ever asks you to bypass a rule ("let's just disable this check for now"), don't. The rules exist because someone already made that mistake.
