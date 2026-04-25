"""
hmm_engine.py — Gaussian HMM volatility regime detection engine.

DESIGN
------
The HMM is a *volatility classifier*, not a return predictor.  It learns
latent market regimes from price-derived features and labels them by their
mean return (ascending) so the strategy layer can map them to portfolio
allocations.

KEY INVARIANT — NO LOOK-AHEAD
------------------------------
Regime labels at bar t must depend *only* on data[0:t].
DO NOT use hmmlearn's ``model.predict()`` or ``model.predict_proba()``
for production/backtest inference.  Both run the Viterbi or
forward-backward algorithm over the **full sequence**, leaking future
information back to earlier bars.

Instead, :meth:`predict_regime_filtered` implements the **forward algorithm
only**, computing the filtered posterior P(s_t | x_{1:t}) one step at a time.

LABEL MAPPING (sorted by mean return, ascending)
-------------------------------------------------
  3 states : BEAR, NEUTRAL, BULL
  4 states : CRASH, BEAR, BULL, EUPHORIA
  5 states : CRASH, BEAR, NEUTRAL, BULL, EUPHORIA
  6 states : CRASH, STRONG_BEAR, WEAK_BEAR, WEAK_BULL, STRONG_BULL, EUPHORIA
  7 states : CRASH, STRONG_BEAR, WEAK_BEAR, NEUTRAL, WEAK_BULL, STRONG_BULL, EUPHORIA
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)
logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)

# ── Constants ──────────────────────────────────────────────────────────────────

REGIME_LABELS: Dict[int, List[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: [
        "CRASH",
        "STRONG_BEAR",
        "WEAK_BEAR",
        "NEUTRAL",
        "WEAK_BULL",
        "STRONG_BULL",
        "EUPHORIA",
    ],
}

# Strategy guidance per label (used to populate RegimeInfo)
_LABEL_META: Dict[str, Dict] = {
    "CRASH":       dict(strategy="defensive", max_lev=0.0,  max_pos=0.05, min_conf=0.70),
    "BEAR":        dict(strategy="reduce",    max_lev=0.0,  max_pos=0.08, min_conf=0.65),
    "STRONG_BEAR": dict(strategy="reduce",    max_lev=0.0,  max_pos=0.08, min_conf=0.65),
    "WEAK_BEAR":   dict(strategy="neutral",   max_lev=1.0,  max_pos=0.10, min_conf=0.60),
    "NEUTRAL":     dict(strategy="neutral",   max_lev=1.0,  max_pos=0.12, min_conf=0.55),
    "WEAK_BULL":   dict(strategy="growth",    max_lev=1.0,  max_pos=0.15, min_conf=0.55),
    "BULL":        dict(strategy="growth",    max_lev=1.0,  max_pos=0.15, min_conf=0.55),
    "STRONG_BULL": dict(strategy="aggressive",max_lev=1.25, max_pos=0.15, min_conf=0.55),
    "EUPHORIA":    dict(strategy="aggressive",max_lev=1.25, max_pos=0.15, min_conf=0.55),
}


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    """Static metadata about a regime state (computed once at training time)."""
    regime_id: int
    regime_name: str
    expected_return: float          # mean of log_ret_1 feature (z-scored)
    expected_volatility: float      # mean of realized_vol_20 feature (z-scored)
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Dynamic state produced by :meth:`HMMEngine.predict_regime_filtered`."""
    label: str                      # e.g. "BULL"
    state_id: int                   # raw HMM state index
    probability: float              # P(state_id | observations so far)
    state_probabilities: np.ndarray # full posterior over all states
    timestamp: Optional[pd.Timestamp]
    is_confirmed: bool              # False during N-bar confirmation window
    consecutive_bars: int           # bars the output regime has been stable


# ── Main class ─────────────────────────────────────────────────────────────────

class HMMEngine:
    """
    Gaussian HMM volatility classifier with automatic model selection.

    Workflow
    --------
    1. ``fit(features)`` — train over ``n_candidates`` state counts, pick
       lowest BIC, label states by mean log-return.
    2. ``predict_regime_filtered(features)`` — forward algorithm only;
       returns :class:`RegimeState` per bar.  Safe for backtests.
    3. ``update(row, timestamp)`` — single-step incremental update for live
       trading.  Maintains stability-filter state internally.

    Parameters
    ----------
    n_candidates :
        State counts evaluated during BIC model selection.
    n_init :
        Random restarts per candidate (best LL across restarts is kept).
    covariance_type :
        HMM emission covariance structure (default: ``"full"``).
    min_train_bars :
        Minimum rows required before ``fit`` is accepted.
    stability_bars :
        Bars a candidate state must persist before being confirmed.
    flicker_window :
        Rolling window (bars) for flicker-rate calculation.
    flicker_threshold :
        Max confirmed regime changes per ``flicker_window`` before
        ``is_flickering()`` returns True.
    min_confidence :
        Minimum posterior probability for a regime to be "actionable".
    min_covar :
        Covariance regularisation added to each diagonal element.
    """

    def __init__(
        self,
        n_candidates: Optional[List[int]] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        min_train_bars: int = 504,
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
        min_covar: float = 1e-3,
    ) -> None:
        self.n_candidates: List[int] = n_candidates or [3, 4, 5, 6, 7]
        self.n_init: int = n_init
        self.covariance_type: str = covariance_type
        self.min_train_bars: int = min_train_bars
        self.stability_bars: int = stability_bars
        self.flicker_window: int = flicker_window
        self.flicker_threshold: int = flicker_threshold
        self.min_confidence: float = min_confidence
        self.min_covar: float = min_covar

        # Set after fit()
        self._model: Optional[hmm.GaussianHMM] = None
        self._n_states: int = 0
        self._state_to_label: Dict[int, str] = {}
        self._regime_info: Dict[str, RegimeInfo] = {}
        self._training_bic: float = np.inf
        self._training_date: Optional[pd.Timestamp] = None
        self._n_features: int = 0
        self._is_fitted: bool = False

        # Stability-filter state (updated by update(), reset by fit())
        self._confirmed_state: int = -1
        self._candidate_state: int = -1
        self._candidate_bars: int = 0        # bars candidate has persisted
        self._confirmed_bars: int = 0        # bars confirmed regime has persisted
        self._confirmed_change_bars: List[int] = []  # bar indices of confirmed changes
        self._bar_counter: int = 0

        # Incremental forward-pass cache (updated by update())
        self._last_log_alpha: Optional[np.ndarray] = None
        self._log_transmat: Optional[np.ndarray] = None  # cached log(transmat)

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,
        lengths: Optional[List[int]] = None,
    ) -> "HMMEngine":
        """
        Select and fit the best Gaussian HMM via BIC.

        Trains separate models for each candidate state count (``n_candidates``),
        each with ``n_init`` random restarts.  Picks the model with the lowest
        BIC.  After fitting, sorts states by mean return and assigns labels.

        Parameters
        ----------
        features :
            2-D array of shape ``(n_bars, n_features)``.  Should already be
            z-score standardised via :class:`~data.feature_engineering.FeatureEngineer`.
        lengths :
            Optional sequence lengths for multi-sequence training.  Defaults
            to ``[n_bars]`` (single sequence).

        Returns
        -------
        self  (for method chaining)

        Raises
        ------
        ValueError  if ``n_bars < min_train_bars``.
        RuntimeError  if every candidate model fails to converge.
        """
        n_samples, n_features = features.shape
        if n_samples < self.min_train_bars:
            raise ValueError(
                f"Need >= {self.min_train_bars} bars, got {n_samples}. "
                "Add more training data."
            )

        if lengths is None:
            lengths = [n_samples]

        best_model: Optional[hmm.GaussianHMM] = None
        best_bic: float = np.inf
        bic_scores: Dict[int, float] = {}

        for n_states in self.n_candidates:
            if n_states not in REGIME_LABELS:
                logger.warning(
                    "n_states=%d has no label mapping in REGIME_LABELS — skipping.",
                    n_states,
                )
                continue

            candidate_model, candidate_ll = self._train_candidate(
                features, lengths, n_states
            )
            if candidate_model is None:
                logger.warning(
                    "All %d initializations failed for n_states=%d.",
                    self.n_init,
                    n_states,
                )
                continue

            bic = self._bic(candidate_model, features, lengths)
            bic_scores[n_states] = round(bic, 2)

            converged = candidate_model.monitor_.converged
            n_iter = candidate_model.monitor_.iter
            logger.info(
                "n_states=%d  BIC=%.2f  LL=%.4f  converged=%s  iter=%d",
                n_states, bic, candidate_ll, converged, n_iter,
            )

            if bic < best_bic:
                best_bic = bic
                best_model = candidate_model

        if best_model is None:
            raise RuntimeError(
                "Failed to fit any HMM model — all candidates diverged. "
                "Try increasing min_covar or providing more data."
            )

        self._model = best_model
        self._n_states = best_model.n_components
        self._n_features = n_features
        self._training_bic = best_bic
        self._training_date = pd.Timestamp.now()
        self._is_fitted = True
        self._log_transmat = np.log(self._model.transmat_ + 1e-300)

        logger.info(
            "BIC summary: %s  →  selected n_states=%d  BIC=%.2f",
            bic_scores, self._n_states, best_bic,
        )

        self._label_states_by_return()
        self._build_regime_info()
        self._reset_stability_state()

        return self

    # ── Forward-algorithm inference (NO LOOK-AHEAD) ────────────────────────────

    def predict_regime_filtered(
        self,
        features: np.ndarray,
        timestamps: Optional[List[pd.Timestamp]] = None,
    ) -> List[RegimeState]:
        """
        Compute P(s_t | x_{1:t}) at every bar using the **forward algorithm only**.

        This is the ONLY safe method for backtesting.  It processes the
        sequence left-to-right; state at bar t never looks at bars t+1 … T.

        The stability filter is applied locally (does not mutate ``self``
        state), so results are deterministic regardless of call order.

        Parameters
        ----------
        features :
            2-D array of shape ``(T, n_features)``.
        timestamps :
            Optional list of length T for :attr:`RegimeState.timestamp`.

        Returns
        -------
        List of :class:`RegimeState`, one per bar.
        """
        self._require_fitted()

        log_emission = self._log_emission_probs(features)   # (T, K)
        log_alpha = self._forward_pass(log_emission)         # (T, K)
        probs_all = np.exp(log_alpha)                        # (T, K) posteriors

        results: List[RegimeState] = []
        confirmed_state = -1
        candidate_state = -1
        candidate_bars = 0
        confirmed_bars = 0
        change_bar_indices: List[int] = []

        for t in range(len(features)):
            probs = probs_all[t]
            raw_state = int(np.argmax(probs))

            # ── Stability filter ──────────────────────────────────────────────
            if raw_state == candidate_state:
                candidate_bars += 1
            else:
                candidate_state = raw_state
                candidate_bars = 1

            # Promote candidate to confirmed when threshold met
            if (
                candidate_bars >= self.stability_bars
                and candidate_state != confirmed_state
            ):
                prev_label = self._state_to_label.get(confirmed_state, "NONE")
                new_label = self._state_to_label.get(candidate_state, f"S{candidate_state}")
                if confirmed_state < 0:
                    logger.info(
                        "Initial regime confirmed: %s at bar %d (p=%.3f)",
                        new_label, t, float(probs[candidate_state]),
                    )
                else:
                    logger.warning(
                        "Regime change: %s → %s at bar %d (p=%.3f)",
                        prev_label, new_label, t, float(probs[candidate_state]),
                    )
                    change_bar_indices.append(t)
                confirmed_state = candidate_state
                confirmed_bars = 1
            elif candidate_state == confirmed_state:
                confirmed_bars += 1

            # ── Output state (use confirmed if available, else candidate) ─────
            output_state = confirmed_state if confirmed_state >= 0 else candidate_state
            in_transition = (
                confirmed_state >= 0
                and candidate_state != confirmed_state
            )
            recent_changes = sum(
                1 for b in change_bar_indices if b >= t - self.flicker_window
            )
            is_confirmed = (
                confirmed_state >= 0
                and not in_transition
                and recent_changes <= self.flicker_threshold
            )

            label = self._state_to_label.get(output_state, f"STATE_{output_state}")
            ts = timestamps[t] if timestamps is not None else None

            results.append(
                RegimeState(
                    label=label,
                    state_id=output_state,
                    probability=float(probs[output_state]),
                    state_probabilities=probs.copy(),
                    timestamp=ts,
                    is_confirmed=is_confirmed,
                    consecutive_bars=confirmed_bars if is_confirmed else candidate_bars,
                )
            )

        return results

    def update(
        self,
        new_feature_row: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> RegimeState:
        """
        Incrementally update the filtered posterior with one new bar.

        Caches the previous ``log_alpha`` vector so the full history does not
        need to be reprocessed.  Mutates stability-filter state.

        Parameters
        ----------
        new_feature_row :
            1-D array of shape ``(n_features,)``.
        timestamp :
            Bar timestamp for the returned :class:`RegimeState`.

        Returns
        -------
        :class:`RegimeState` for the current bar.
        """
        self._require_fitted()

        row = new_feature_row.reshape(1, -1)
        log_emission = self._log_emission_probs(row)[0]   # (K,)

        if self._last_log_alpha is None:
            log_alpha = np.log(self._model.startprob_ + 1e-300) + log_emission
        else:
            log_predicted = logsumexp(
                self._last_log_alpha[:, np.newaxis] + self._log_transmat, axis=0
            )
            log_alpha = log_predicted + log_emission

        lse = logsumexp(log_alpha)
        log_alpha -= lse if np.isfinite(lse) else 0.0
        self._last_log_alpha = log_alpha

        probs = np.exp(log_alpha)
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        raw_state = int(np.argmax(probs))
        self._bar_counter += 1

        # ── Stability filter ──────────────────────────────────────────────────
        if raw_state == self._candidate_state:
            self._candidate_bars += 1
        else:
            self._candidate_state = raw_state
            self._candidate_bars = 1

        if (
            self._candidate_bars >= self.stability_bars
            and self._candidate_state != self._confirmed_state
        ):
            prev = self._state_to_label.get(self._confirmed_state, "NONE")
            nxt = self._state_to_label.get(self._candidate_state, f"S{self._candidate_state}")
            if self._confirmed_state < 0:
                logger.info("Initial regime confirmed: %s (p=%.3f)", nxt, float(probs[raw_state]))
            else:
                logger.info(
                    "Regime change confirmed: %s → %s  p=%.3f  t=%s",
                    prev, nxt, float(probs[raw_state]), timestamp,
                )
                self._confirmed_change_bars.append(self._bar_counter)
            self._confirmed_state = self._candidate_state
            self._confirmed_bars = 1
        elif self._candidate_state == self._confirmed_state:
            self._confirmed_bars += 1

        output_state = (
            self._confirmed_state if self._confirmed_state >= 0
            else self._candidate_state
        )
        in_transition = (
            self._confirmed_state >= 0
            and self._candidate_state != self._confirmed_state
        )
        is_confirmed = (
            self._confirmed_state >= 0
            and not in_transition
            and not self.is_flickering()
        )

        label = self._state_to_label.get(output_state, f"STATE_{output_state}")
        return RegimeState(
            label=label,
            state_id=output_state,
            probability=float(probs[output_state]),
            state_probabilities=probs.copy(),
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=self._confirmed_bars if is_confirmed else self._candidate_bars,
        )

    def predict_regime_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Raw filtered posterior probabilities — no stability filter applied.

        Parameters
        ----------
        features :
            2-D array of shape ``(T, n_features)``.

        Returns
        -------
        Array of shape ``(T, n_states)`` where each row sums to 1.
        """
        self._require_fitted()
        log_emission = self._log_emission_probs(features)
        log_alpha = self._forward_pass(log_emission)
        probs = np.exp(log_alpha)
        # Ensure rows sum to 1 (numerical noise guard)
        row_sums = probs.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, probs / row_sums, 1.0 / self._n_states)

    # ── Stability / flicker queries ────────────────────────────────────────────

    def get_regime_stability(self) -> int:
        """
        Consecutive bars the current *confirmed* regime has persisted.

        Returns 0 if no regime has been confirmed yet.
        """
        return max(self._confirmed_bars, 0)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Learned transition probability matrix.

        Returns
        -------
        Array of shape ``(n_states, n_states)`` where element ``[i, j]`` is
        P(state_j | state_i).
        """
        self._require_fitted()
        return self._model.transmat_.copy()

    def detect_regime_change(self) -> bool:
        """
        True only if a regime change was confirmed on the most recent
        :meth:`update` call.
        """
        return (
            len(self._confirmed_change_bars) > 0
            and self._confirmed_change_bars[-1] == self._bar_counter
        )

    def get_regime_flicker_rate(self) -> float:
        """
        Number of confirmed regime changes in the last ``flicker_window`` bars.
        """
        cutoff = self._bar_counter - self.flicker_window
        recent = sum(1 for b in self._confirmed_change_bars if b >= cutoff)
        return float(recent)

    def is_flickering(self) -> bool:
        """True if the regime-change rate exceeds ``flicker_threshold``."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    # ── Metadata accessors ─────────────────────────────────────────────────────

    def get_regime_info(self, label: str) -> Optional[RegimeInfo]:
        """Return :class:`RegimeInfo` for a given label, or None."""
        return self._regime_info.get(label)

    def get_all_regime_info(self) -> Dict[str, RegimeInfo]:
        """Return the full label → :class:`RegimeInfo` mapping."""
        return dict(self._regime_info)

    def get_state_label(self, state_id: int) -> str:
        """Map a raw HMM state index to its human-readable label."""
        self._require_fitted()
        return self._state_to_label.get(state_id, f"STATE_{state_id}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the fitted engine (model + metadata) to ``path`` with pickle.

        Parameters
        ----------
        path : str  destination file path (e.g. ``"models/hmm_2024.pkl"``).
        """
        self._require_fitted()
        payload = {
            "model": self._model,
            "n_states": self._n_states,
            "n_features": self._n_features,
            "state_to_label": self._state_to_label,
            "regime_info": self._regime_info,
            "training_bic": self._training_bic,
            "training_date": self._training_date,
            "config": {
                "n_candidates": self.n_candidates,
                "n_init": self.n_init,
                "covariance_type": self.covariance_type,
                "min_train_bars": self.min_train_bars,
                "stability_bars": self.stability_bars,
                "flicker_window": self.flicker_window,
                "flicker_threshold": self.flicker_threshold,
                "min_confidence": self.min_confidence,
                "min_covar": self.min_covar,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s (n_states=%d, BIC=%.2f)", path, self._n_states, self._training_bic)

    @classmethod
    def load(cls, path: str) -> "HMMEngine":
        """
        Restore a previously saved engine from ``path``.

        Parameters
        ----------
        path : str  source file path.

        Returns
        -------
        Fitted :class:`HMMEngine` instance.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cfg = payload["config"]
        engine = cls(**cfg)
        engine._model = payload["model"]
        engine._n_states = payload["n_states"]
        engine._n_features = payload["n_features"]
        engine._state_to_label = payload["state_to_label"]
        engine._regime_info = payload["regime_info"]
        engine._training_bic = payload["training_bic"]
        engine._training_date = payload["training_date"]
        engine._is_fitted = True
        engine._log_transmat = np.log(engine._model.transmat_ + 1e-300)
        logger.info(
            "Model loaded from %s (n_states=%d, trained=%s)",
            path, engine._n_states, engine._training_date,
        )
        return engine

    def needs_retraining(
        self, current_date: pd.Timestamp, retrain_interval_days: int = 63
    ) -> bool:
        """
        True if the model is older than ``retrain_interval_days`` calendar days.

        Parameters
        ----------
        current_date :
            Today's date.
        retrain_interval_days :
            Retraining cadence (default 63 ≈ 1 quarter).
        """
        if not self._is_fitted or self._training_date is None:
            return True
        age = (current_date - self._training_date).days
        return age >= retrain_interval_days

    # ── Private — forward algorithm ────────────────────────────────────────────

    def _log_emission_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Log-likelihood of each observation under each state's Gaussian emission.

        Parameters
        ----------
        obs :
            2-D array of shape ``(T, D)``.

        Returns
        -------
        Array of shape ``(T, K)`` — log P(obs_t | state_k).
        """
        T, _ = obs.shape
        K = self._n_states
        log_probs = np.full((T, K), -1e30)

        for k in range(K):
            try:
                cov = self._model.covars_[k].copy()
                # Regularise diagonal to avoid singularity
                cov += np.eye(cov.shape[0]) * self.min_covar
                log_probs[:, k] = multivariate_normal.logpdf(
                    obs,
                    mean=self._model.means_[k],
                    cov=cov,
                    allow_singular=False,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                logger.debug("Emission prob failed for state %d: %s", k, exc)

        return log_probs

    def _forward_pass(self, log_emission: np.ndarray) -> np.ndarray:
        """
        Forward algorithm — computes log P(s_t | x_{1:t}) at each bar.

        This is the **filtered** posterior.  It uses ONLY observations up to
        and including bar t.  No future data is ever accessed.

        Parameters
        ----------
        log_emission :
            Array of shape ``(T, K)`` — log P(x_t | state_k).

        Returns
        -------
        ``log_alpha`` : Array of shape ``(T, K)`` — log normalised filtered
        posteriors.  ``exp(log_alpha[t])`` sums to 1.
        """
        T, K = log_emission.shape
        log_alpha = np.empty((T, K), dtype=np.float64)

        # ── Initialise: P(s_0 | x_0) ─────────────────────────────────────────
        log_alpha[0] = np.log(self._model.startprob_ + 1e-300) + log_emission[0]
        lse = logsumexp(log_alpha[0])
        log_alpha[0] -= lse if np.isfinite(lse) else 0.0

        # ── Forward recursion ─────────────────────────────────────────────────
        # log_transmat[i, j] = log P(s_t = j | s_{t-1} = i)
        for t in range(1, T):
            # Predict: log P(s_t | x_{1:t-1}) via marginalization over s_{t-1}
            #   = logsumexp_i[ log alpha_{t-1}(i) + log A(i→j) ]  for each j
            log_predicted = logsumexp(
                log_alpha[t - 1, :, np.newaxis] + self._log_transmat,
                axis=0,
            )  # shape (K,)

            # Update: multiply by emission, normalise
            log_alpha[t] = log_predicted + log_emission[t]
            lse = logsumexp(log_alpha[t])
            if np.isfinite(lse):
                log_alpha[t] -= lse
            else:
                # All emissions are near-zero — use uniform fallback
                log_alpha[t] = np.full(K, -np.log(K))

        return log_alpha

    # ── Private — BIC and model selection ─────────────────────────────────────

    def _bic(
        self,
        model: hmm.GaussianHMM,
        features: np.ndarray,
        lengths: List[int],
    ) -> float:
        """
        Bayesian Information Criterion: −2·ℓ + p·ln(N).

        Free parameters for Gaussian HMM with full covariance:
          - Transition matrix: K·(K−1)
          - Initial distribution: K−1
          - Means: K·D
          - Full covariance matrices: K·D·(D+1)/2
        """
        K = model.n_components
        N, D = features.shape

        n_trans = K * (K - 1)
        n_start = K - 1
        n_means = K * D
        n_cov = K * D * (D + 1) // 2
        n_params = n_trans + n_start + n_means + n_cov

        try:
            log_lik = model.score(features, lengths)
        except Exception as exc:
            logger.debug("score() failed for n_states=%d: %s", K, exc)
            return np.inf

        return -2.0 * log_lik + n_params * np.log(N)

    def _train_candidate(
        self,
        features: np.ndarray,
        lengths: List[int],
        n_states: int,
    ) -> Tuple[Optional[hmm.GaussianHMM], float]:
        """
        Train an HMM with ``n_states`` using ``n_init`` random restarts.
        Returns the best model and its log-likelihood.
        """
        best_model: Optional[hmm.GaussianHMM] = None
        best_ll: float = -np.inf

        for seed in range(self.n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=300,
                    tol=1e-5,
                    min_covar=self.min_covar,
                    random_state=seed * 13 + n_states,
                    verbose=False,
                )
                model.fit(features, lengths)
                ll = model.score(features, lengths)
                logger.debug(
                    "  n_states=%d seed=%d  LL=%.4f  converged=%s  iter=%d",
                    n_states, seed, ll,
                    model.monitor_.converged, model.monitor_.iter,
                )
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception as exc:
                logger.debug("n_states=%d seed=%d failed: %s", n_states, seed, exc)

        return best_model, best_ll

    # ── Private — labelling and metadata ──────────────────────────────────────

    def _label_states_by_return(self) -> None:
        """
        Assign labels to HMM states by sorting their emission means on the
        first feature column (log_ret_1, z-scored) in ascending order.

        Lowest mean return → index 0 in REGIME_LABELS[n_states] (BEAR/CRASH).
        Highest mean return → last index (BULL/EUPHORIA).

        NOTE: labels are for human readability only.  The strategy layer
        independently identifies low/high-vol regimes by examining volatility
        feature means — it does NOT rely on these labels.
        """
        K = self._n_states
        mean_returns = self._model.means_[:, 0]          # log_ret_1 column
        sorted_states = np.argsort(mean_returns)          # ascending
        labels = REGIME_LABELS[K]

        self._state_to_label = {
            int(state_idx): labels[rank]
            for rank, state_idx in enumerate(sorted_states)
        }

        log_lines = {
            labels[rank]: f"μ_ret={mean_returns[state_idx]:.4f}"
            for rank, state_idx in enumerate(sorted_states)
        }
        logger.info("State labels (by ascending mean return): %s", log_lines)

    def _build_regime_info(self) -> None:
        """Populate :attr:`_regime_info` from emission means and _LABEL_META."""
        self._regime_info = {}
        for state_id, label in self._state_to_label.items():
            meta = _LABEL_META.get(label, _LABEL_META["NEUTRAL"])
            means = self._model.means_[state_id]

            # log_ret_1 at index 0; realized_vol_20 at index 3 (if available)
            exp_ret = float(means[0])
            exp_vol = float(means[3]) if len(means) > 3 else 0.0

            self._regime_info[label] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                recommended_strategy_type=meta["strategy"],
                max_leverage_allowed=meta["max_lev"],
                max_position_size_pct=meta["max_pos"],
                min_confidence_to_act=meta["min_conf"],
            )

    # ── Private — helpers ──────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HMMEngine is not fitted. Call fit() before predicting."
            )

    def _reset_stability_state(self) -> None:
        """Reset all stability-filter state variables (called after fit)."""
        self._confirmed_state = -1
        self._candidate_state = -1
        self._candidate_bars = 0
        self._confirmed_bars = 0
        self._confirmed_change_bars = []
        self._bar_counter = 0
        self._last_log_alpha = None

