# Quarantine — review by 2026-05-02

Files moved here on 2026-04-25 during dead-code cleanup. If nothing here has been needed by 2026-05-02, the entire directory is safe to delete.

| File | Why quarantined |
|------|-----------------|
| `BREAKPOINT_2026-04-23.md` | Closed investigation breakpoint; kept briefly for historical reference. |
| `correlation_analysis.py` | Already flagged in `obsolete_code/`; no imports found. |
| `test_portfolios.yaml` | No loader code; flagged in `obsolete_code/`. |
| `settings-snapshot.yaml` | Old config snapshot; runtime uses `config/settings.yaml`. |

To restore something: `git mv _quarantine_2026-05-02/<file> <original-path>/`.
