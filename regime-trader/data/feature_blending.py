"""
data/feature_blending.py — Cross-symbol feature blending utility.

Historically the same "average log_ret_1 / realized_vol_20 across equity-like
symbols" block was copy-pasted into 5 places (3 in main.py, 2 in
backtest/backtester.py). This module centralises it so there is exactly one
source of truth.

The intent: when the HMM sees features from a *basket* of equities, the
return / vol signals should be a cross-sectional mean rather than a single
proxy symbol. Non-equity assets (e.g. GLD, TLT, USO) are excluded because
they respond to different macro drivers and add noise.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# The columns we blend by default. Both are per-bar, symbol-specific features
# whose cross-sectional mean is a meaningful basket signal.
DEFAULT_BLEND_COLS: Sequence[str] = ("log_ret_1", "realized_vol_20")


def blend_cross_symbol_features(
    base_features: pd.DataFrame,
    per_symbol_bars: Dict[str, pd.DataFrame],
    feature_engineer,
    blend_exclude: Optional[Iterable[str]] = None,
    blend_cols: Sequence[str] = DEFAULT_BLEND_COLS,
    min_bars: int = 10,
) -> pd.DataFrame:
    """Overwrite selected columns in ``base_features`` with the cross-symbol mean.

    Parameters
    ----------
    base_features
        DataFrame already containing single-symbol features. Columns in
        ``blend_cols`` that are present here will be overwritten in place
        with the basket mean.
    per_symbol_bars
        Mapping ``symbol -> OHLCV DataFrame``. The function will recompute
        the blend columns for each symbol via
        ``feature_engineer.build_feature_matrix`` and average them.
    feature_engineer
        An object exposing ``build_feature_matrix(bars, feature_names, dropna)``
        (i.e. a ``FeatureEngineer`` instance).
    blend_exclude
        Symbols to skip (e.g. non-equity ETFs like GLD/TLT/USO). Typically
        comes from ``hmm_cfg.get("blend_exclude", [])``.
    blend_cols
        Columns to blend. Any column not present in ``base_features`` is
        silently skipped.
    min_bars
        Minimum length for a symbol's bars to be included in the average.

    Returns
    -------
    The same ``base_features`` DataFrame, with blend columns rewritten if
    at least two symbols contributed. If fewer than two symbols are usable,
    ``base_features`` is returned untouched.
    """
    blend_cols_present = [c for c in blend_cols if c in base_features.columns]
    if not blend_cols_present:
        return base_features

    exclude = set(blend_exclude or ())
    blend_syms = [s for s in per_symbol_bars if s not in exclude]
    if len(blend_syms) < 2:
        return base_features

    per_sym_dfs   = []
    per_sym_keys  = []
    for sym in blend_syms:
        bars = per_symbol_bars.get(sym)
        if bars is None or len(bars) < min_bars:
            continue
        try:
            per_sym_dfs.append(
                feature_engineer.build_feature_matrix(
                    bars, feature_names=blend_cols_present, dropna=False,
                )
            )
            per_sym_keys.append(sym)
        except Exception as exc:
            logger.debug("blend: skipping %s (%s)", sym, exc)
            continue

    if len(per_sym_keys) < 2:
        return base_features

    per_sym = pd.concat(per_sym_dfs, axis=1, keys=per_sym_keys)
    for col in blend_cols_present:
        base_features[col] = per_sym.xs(col, level=1, axis=1).mean(axis=1)

    return base_features
