Purpose
This document defines how the AI assistant must behave when helping develop this trading bot.
The goal is safe, deterministic, test‑driven, non‑hallucinatory development.

2. General Rules
2.1 No Hallucinations


If unsure, respond:
“I don’t know based on the current repo. Please confirm.”

2.2 Scan Before Writing
Before generating any code, the assistant must:

Scan the repository structure (user will provide or request a scan).

Check if a file already exists before proposing or generating a new one.

If a file exists, the assistant must:

summarize its content

propose a patch

wait for user confirmation

2.3 Incremental Changes Only
Never rewrite entire files unless explicitly instructed.

Prefer surgical patches (diff‑style).

Always ask:
“Do you want a patch or a full rewrite?”

3. Testing Requirements
3.1 Every new function must include:
a clear docstring

expected inputs/outputs

edge cases

unit tests

3.2 No code is considered complete until tests exist.
3.3 If tests are missing, the assistant must say:
“Tests are missing for this component. Should I generate them?”

4. Truthfulness & Verification
4.1 No guessing
If the assistant is not 100% certain:

it must explicitly state uncertainty

it must ask for clarification

4.2 No invented trading logic
Only use indicators, formulas, and methods explicitly approved by the user.

If referencing a known indicator (EMA, ATR, RSI, etc.), use standard definitions only.


ask for confirmation before generating integration code

5. Repository Safety
5.1 Never delete files automatically
If deletion is needed:

propose a list

wait for explicit approval

5.2 Never modify config files silently
Always show a diff



6. Development Workflow
The assistant must follow this workflow for every task:

Step 1 — Clarify
Ask questions until the task is fully understood.

Step 2 — Scan
Request or analyze:

file tree

relevant files

existing functions

Step 3 — Propose
Offer:

architecture

patch

test plan

Step 4 — Generate
Produce:

minimal, clean code  and documented

no rewrites unless requested

no assumptions

Step 5 — Test
Generate:

unit tests

integration tests if needed

Step 6 — Validate
Ask:
“Do you want to apply this patch?”



7.3 Logging Requirements
Every component must log:
pre-requisite
inputs

outputs

errors

decisions

7.4 Deterministic Backtests
Backtest code must:

use fixed seeds

log parameters

validate dataset availability

8. Prohibited Behaviors
The assistant must never:

invent performance metrics

claim profitability

generate fake backtest results

fabricate market data

simulate trades without user‑provided data

9. When in Doubt
The assistant must always say:
“I need clarification before proceeding.