# Obsolete code

Files in this folder are **not imported or referenced** by any production code
path. They are kept here (rather than deleted) as historical references in case
any logic needs to be re-exhumed.

| File | Origin | Why obsolete |
|------|--------|--------------|
| `correlation_analysis.py` | `tools/` | Standalone diagnostic script, no import anywhere in the codebase. |
| `test_portfolios.yaml` | `config/` | Four curated stock universes, no loader references this file. |

If anything in here becomes useful again, move it back and wire it in.
