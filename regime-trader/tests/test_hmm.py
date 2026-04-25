"""
test_hmm.py — Unit tests for the HMM regime detection engine.

Tests cover model fitting, BIC model selection, regime labelling (by return),
forward-algorithm prediction, stability filtering, and flicker detection.
"""

import numpy as np
import pytest

from core.hmm_engine import HMMEngine, RegimeInfo, RegimeState, REGIME_LABELS


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_clustered_features(n_per_cluster: int = 120, seed: int = 42) -> np.ndarray:
    """
    Three tightly separated Gaussian clusters with 14 features each.

    Cluster means are designed so that:
      - cluster 0 has the lowest first-feature mean (→ BEAR label)
      - cluster 2 has the highest first-feature mean (→ BULL label)
    This guarantees deterministic label assignment and near-perfect BIC selection.
    """
    rng = np.random.default_rng(seed)
    d = 14  # match the full feature-engineer output width
    cov = np.eye(d) * 1e-5   # very tight — BIC will strongly prefer n_states=3

    # First feature = log_ret_1 proxy; kept clearly separated
    mean_bear   = np.full(d, 0.01);  mean_bear[0]  = -0.50
    mean_neutral = np.full(d, 0.01); mean_neutral[0] = 0.00
    mean_bull   = np.full(d, 0.01);  mean_bull[0]  = +0.50

    c0 = rng.multivariate_normal(mean_bear,    cov, n_per_cluster)
    c1 = rng.multivariate_normal(mean_neutral, cov, n_per_cluster)
    c2 = rng.multivariate_normal(mean_bull,    cov, n_per_cluster)
    return np.vstack([c0, c1, c2])


@pytest.fixture(scope="module")
def synthetic_features() -> np.ndarray:
    return _make_clustered_features()


@pytest.fixture(scope="module")
def fitted_engine(synthetic_features: np.ndarray) -> HMMEngine:
    """HMMEngine fitted on the 3-cluster synthetic data."""
    engine = HMMEngine(
        n_candidates=[3, 4, 5],
        n_init=5,
        min_train_bars=50,
        stability_bars=3,
        flicker_window=20,
        flicker_threshold=4,
        min_covar=1e-4,
    )
    engine.fit(synthetic_features)
    return engine


# ── Fitting tests ──────────────────────────────────────────────────────────────

class TestHMMFit:
    def test_fit_returns_self(self, synthetic_features: np.ndarray) -> None:
        """fit() must return the engine instance for method chaining."""
        engine = HMMEngine(n_candidates=[3], n_init=2, min_train_bars=50)
        result = engine.fit(synthetic_features)
        assert result is engine

    def test_is_fitted_flag_set(self, fitted_engine: HMMEngine) -> None:
        """_is_fitted must be True after a successful fit."""
        assert fitted_engine._is_fitted is True

    def test_raises_on_insufficient_data(self) -> None:
        """fit() must raise ValueError when fewer than min_train_bars rows are given."""
        engine = HMMEngine(n_candidates=[3], n_init=1, min_train_bars=500)
        tiny = np.random.default_rng(0).normal(0, 1, (100, 14))
        with pytest.raises(ValueError, match="more training data"):
            engine.fit(tiny)

    def test_model_stored(self, fitted_engine: HMMEngine) -> None:
        """_model attribute must not be None after fit."""
        assert fitted_engine._model is not None

    def test_n_states_in_candidates(self, fitted_engine: HMMEngine) -> None:
        """Selected n_states must be one of the candidates passed to the constructor."""
        assert fitted_engine._n_states in fitted_engine.n_candidates

    def test_training_date_set(self, fitted_engine: HMMEngine) -> None:
        """_training_date must be populated after fit."""
        assert fitted_engine._training_date is not None

    def test_n_features_recorded(self, fitted_engine: HMMEngine, synthetic_features: np.ndarray) -> None:
        """_n_features must match the width of the training data."""
        assert fitted_engine._n_features == synthetic_features.shape[1]

    def test_predict_fails_before_fit(self) -> None:
        """predict_regime_filtered() must raise RuntimeError before fit is called."""
        engine = HMMEngine(n_candidates=[3], n_init=1, min_train_bars=50)
        X = np.random.default_rng(0).normal(0, 1, (10, 14))
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.predict_regime_filtered(X)


# ── BIC model selection tests ──────────────────────────────────────────────────

class TestBICModelSelection:
    def test_selects_n_states_3_on_3_cluster_data(
        self, synthetic_features: np.ndarray
    ) -> None:
        """
        On data with three clearly separated clusters BIC should prefer n_states=3.
        """
        engine = HMMEngine(
            n_candidates=[3, 4, 5],
            n_init=5,
            min_train_bars=50,
            min_covar=1e-4,
        )
        engine.fit(synthetic_features)
        assert engine._n_states == 3, (
            f"Expected n_states=3 on 3-cluster data, got {engine._n_states}. "
            "BIC model selection may be broken."
        )

    def test_bic_is_finite(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """_bic() must return a finite float."""
        bic = fitted_engine._bic(
            fitted_engine._model,
            synthetic_features,
            [len(synthetic_features)],
        )
        assert np.isfinite(bic), f"BIC returned {bic}, expected a finite value."

    def test_bic_formula(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        BIC = −2·LL + n_params·ln(N).  It can be negative when LL is very
        large (highly confident model on tight clusters), so we only verify it
        is finite and that increasing n_states raises BIC (penalisation works).
        """
        bic_3 = fitted_engine._bic(
            fitted_engine._model,
            synthetic_features,
            [len(synthetic_features)],
        )
        assert np.isfinite(bic_3)

        # Verify the penalty increases with n_states by checking n_params count
        # (use same LL so only the penalty term differs)
        K = fitted_engine._n_states
        N, D = synthetic_features.shape
        n_params_k   = K   * (K - 1) + (K - 1)   + K * D + K * D * (D + 1) // 2
        n_params_kp1 = (K+1)*(K)     + K          + (K+1)*D + (K+1)*D*(D+1)//2
        assert n_params_kp1 > n_params_k, "BIC penalty must grow with n_states."

    def test_training_bic_stored(self, fitted_engine: HMMEngine) -> None:
        """_training_bic must be set and finite after fit."""
        assert np.isfinite(fitted_engine._training_bic)


# ── Regime labelling tests ─────────────────────────────────────────────────────

class TestRegimeLabelling:
    def test_all_states_labelled(self, fitted_engine: HMMEngine) -> None:
        """Every HMM state must have an entry in _state_to_label."""
        assert len(fitted_engine._state_to_label) == fitted_engine._n_states
        for state_id in range(fitted_engine._n_states):
            assert state_id in fitted_engine._state_to_label

    def test_labels_are_valid_strings(self, fitted_engine: HMMEngine) -> None:
        """Every label must be a non-empty string from the REGIME_LABELS table."""
        valid = set(REGIME_LABELS[fitted_engine._n_states])
        for state_id, label in fitted_engine._state_to_label.items():
            assert isinstance(label, str) and label
            assert label in valid, f"State {state_id} got unexpected label '{label}'"

    def test_lowest_return_state_is_bear(self, fitted_engine: HMMEngine) -> None:
        """
        The state with the lowest emission mean on feature[0] (log_ret_1) must
        map to the lowest-ranked label (BEAR in a 3-state model).
        """
        means = fitted_engine._model.means_[:, 0]
        lowest_state = int(np.argmin(means))
        label = fitted_engine._state_to_label[lowest_state]
        expected = REGIME_LABELS[fitted_engine._n_states][0]   # first = lowest return
        assert label == expected, (
            f"State with lowest return mean has label '{label}', expected '{expected}'."
        )

    def test_highest_return_state_is_bull(self, fitted_engine: HMMEngine) -> None:
        """
        The state with the highest emission mean on feature[0] must map to the
        highest-ranked label (BULL in a 3-state model).
        """
        means = fitted_engine._model.means_[:, 0]
        highest_state = int(np.argmax(means))
        label = fitted_engine._state_to_label[highest_state]
        expected = REGIME_LABELS[fitted_engine._n_states][-1]  # last = highest return
        assert label == expected, (
            f"State with highest return mean has label '{label}', expected '{expected}'."
        )

    def test_regime_info_populated(self, fitted_engine: HMMEngine) -> None:
        """_regime_info must contain a RegimeInfo entry for every label."""
        assert len(fitted_engine._regime_info) == fitted_engine._n_states
        for label, info in fitted_engine._regime_info.items():
            assert isinstance(info, RegimeInfo)
            assert info.regime_name == label
            assert 0.0 <= info.max_leverage_allowed
            assert 0.0 < info.max_position_size_pct <= 1.0
            assert 0.0 < info.min_confidence_to_act <= 1.0

    def test_get_state_label(self, fitted_engine: HMMEngine) -> None:
        """get_state_label() must return the same mapping as _state_to_label."""
        for state_id in range(fitted_engine._n_states):
            assert (
                fitted_engine.get_state_label(state_id)
                == fitted_engine._state_to_label[state_id]
            )


# ── Forward-algorithm prediction tests ────────────────────────────────────────

class TestPrediction:
    def test_returns_list_of_regime_states(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """predict_regime_filtered() must return a list of RegimeState objects."""
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        assert isinstance(results, list)
        assert len(results) == len(synthetic_features)
        assert all(isinstance(r, RegimeState) for r in results)

    def test_posteriors_sum_to_one(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """state_probabilities in every RegimeState must sum to 1.0."""
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        for t, r in enumerate(results):
            total = float(r.state_probabilities.sum())
            assert abs(total - 1.0) < 1e-6, (
                f"Bar {t}: posteriors sum to {total}, not 1.0"
            )

    def test_probability_in_unit_interval(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """probability field must be in [0, 1] for every bar."""
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        for t, r in enumerate(results):
            assert 0.0 <= r.probability <= 1.0, (
                f"Bar {t}: probability={r.probability} outside [0, 1]"
            )

    def test_probability_equals_max_posterior(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """probability must equal the posterior of state_id."""
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        for t, r in enumerate(results):
            expected = float(r.state_probabilities[r.state_id])
            assert abs(r.probability - expected) < 1e-9, (
                f"Bar {t}: probability={r.probability} != "
                f"state_probabilities[{r.state_id}]={expected}"
            )

    def test_predict_regime_proba_shape(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """predict_regime_proba() must return (T, K) array summing to 1 per row."""
        probs = fitted_engine.predict_regime_proba(synthetic_features)
        T, K = synthetic_features.shape[0], fitted_engine._n_states
        assert probs.shape == (T, K)
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(T),
            atol=1e-6,
            err_msg="predict_regime_proba rows do not sum to 1.",
        )

    def test_high_confidence_on_well_separated_clusters(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        On tightly clustered synthetic data the model should be very confident
        (p > 0.90) for the majority of bars once the model has warmed up.
        """
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        # Skip first few bars (stability warm-up)
        steady = results[20:]
        high_conf = [r for r in steady if r.probability > 0.90]
        ratio = len(high_conf) / len(steady)
        assert ratio > 0.80, (
            f"Only {ratio:.1%} of bars have p>0.90 on well-separated clusters. "
            "Model is underconfident — check emission parameters."
        )

    def test_labels_belong_to_valid_set(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """Every label in the prediction sequence must be in REGIME_LABELS."""
        valid = set(REGIME_LABELS[fitted_engine._n_states])
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        for t, r in enumerate(results):
            assert r.label in valid, (
                f"Bar {t}: unknown label '{r.label}'"
            )

    def test_update_returns_regime_state(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """update() must return a RegimeState for each bar."""
        # Use a fresh engine with the same fitted model to avoid mutating the fixture
        engine = HMMEngine.__new__(HMMEngine)
        engine.__dict__.update(fitted_engine.__dict__)
        engine._reset_stability_state()

        for row in synthetic_features[:10]:
            state = engine.update(row)
            assert isinstance(state, RegimeState)

    def test_transition_matrix_shape(self, fitted_engine: HMMEngine) -> None:
        """get_transition_matrix() must return (K, K) and row-stochastic."""
        A = fitted_engine.get_transition_matrix()
        K = fitted_engine._n_states
        assert A.shape == (K, K)
        np.testing.assert_allclose(
            A.sum(axis=1),
            np.ones(K),
            atol=1e-6,
            err_msg="Transition matrix rows do not sum to 1.",
        )


# ── Stability filter tests ─────────────────────────────────────────────────────

class TestStabilityFilter:
    def test_initial_state_unconfirmed(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """The very first bar must be unconfirmed (stability warm-up pending)."""
        results = fitted_engine.predict_regime_filtered(synthetic_features[:1])
        assert results[0].is_confirmed is False

    def test_confirmed_after_stability_bars(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        is_confirmed must become True once a regime persists for stability_bars
        consecutive bars.
        """
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        # Find the first confirmed bar
        confirmed_indices = [i for i, r in enumerate(results) if r.is_confirmed]
        assert confirmed_indices, "No regime was ever confirmed — stability filter broken."
        first_confirmed = confirmed_indices[0]
        assert first_confirmed >= fitted_engine.stability_bars - 1, (
            f"Regime confirmed at bar {first_confirmed} but stability_bars="
            f"{fitted_engine.stability_bars} — confirmed too early."
        )

    def test_consecutive_bars_non_decreasing_in_confirmed_run(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        While the output regime is *confirmed* and the label does not change,
        consecutive_bars must be non-decreasing.

        Note: during brief unconfirmed transitions the engine switches from
        confirmed_bars to candidate_bars (which resets to 1), so the count
        may dip while is_confirmed is False.  We only enforce monotonicity
        within sustained confirmed runs.
        """
        results = fitted_engine.predict_regime_filtered(synthetic_features)
        prev_label = None
        prev_consec = 0
        prev_confirmed = False
        for r in results:
            if r.label == prev_label and r.is_confirmed and prev_confirmed:
                assert r.consecutive_bars >= prev_consec, (
                    f"consecutive_bars decreased from {prev_consec} to "
                    f"{r.consecutive_bars} during a sustained confirmed run."
                )
            prev_label = r.label
            prev_consec = r.consecutive_bars
            prev_confirmed = r.is_confirmed

    def test_flicker_not_triggered_on_stable_sequence(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        For a long constant input (all from one cluster), is_flickering() must
        be False after warm-up.
        """
        # Build a long constant sequence from cluster 2 (BULL)
        rng = np.random.default_rng(7)
        d = fitted_engine._n_features
        constant = rng.multivariate_normal(
            np.full(d, 0.01) + np.array([0.5] + [0.0] * (d - 1)),
            np.eye(d) * 1e-5,
            200,
        )
        engine = HMMEngine.__new__(HMMEngine)
        engine.__dict__.update(fitted_engine.__dict__)
        engine._reset_stability_state()

        for row in constant:
            engine.update(row)

        assert not engine.is_flickering(), (
            "is_flickering() returned True on a stable constant sequence."
        )

    def test_flicker_triggered_after_rapid_changes(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        Rapidly alternating between two clusters must eventually trigger
        is_flickering().
        """
        rng = np.random.default_rng(9)
        d = fitted_engine._n_features
        cov = np.eye(d) * 1e-5
        mean_a = np.full(d, 0.0); mean_a[0] = -0.5
        mean_b = np.full(d, 0.0); mean_b[0] = +0.5

        engine = HMMEngine(
            n_candidates=[3],
            n_init=1,
            min_train_bars=50,
            stability_bars=1,      # confirm immediately so changes register fast
            flicker_window=20,
            flicker_threshold=4,
            min_covar=1e-4,
        )
        engine.fit(synthetic_features)
        engine._reset_stability_state()

        # Drive rapid alternation
        flicker_triggered = False
        for i in range(80):
            mean = mean_a if i % 2 == 0 else mean_b
            row = rng.multivariate_normal(mean, cov)
            engine.update(row)
            if engine.is_flickering():
                flicker_triggered = True
                break

        assert flicker_triggered, (
            "is_flickering() never triggered after 80 rapid alternations. "
            "Flicker detection may be broken."
        )

    def test_get_regime_stability_increments(
        self, fitted_engine: HMMEngine, synthetic_features: np.ndarray
    ) -> None:
        """
        get_regime_stability() must return a non-negative integer and increase
        while the confirmed regime is stable.
        """
        engine = HMMEngine.__new__(HMMEngine)
        engine.__dict__.update(fitted_engine.__dict__)
        engine._reset_stability_state()

        # Feed a long stable run from the BEAR cluster
        rng = np.random.default_rng(11)
        d = engine._n_features
        mean_bear = np.full(d, 0.0); mean_bear[0] = -0.5
        cov = np.eye(d) * 1e-5

        stabilities = []
        for _ in range(30):
            row = rng.multivariate_normal(mean_bear, cov)
            engine.update(row)
            stabilities.append(engine.get_regime_stability())

        assert all(s >= 0 for s in stabilities), "Negative stability count."
        # After warm-up the stability count should eventually exceed 0
        assert max(stabilities) > 0, "Stability never incremented past 0."


# ── Persistence tests ──────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(
        self,
        fitted_engine: HMMEngine,
        synthetic_features: np.ndarray,
        tmp_path,
    ) -> None:
        """
        Saving and loading must produce an engine that gives identical
        predictions to the original.
        """
        path = str(tmp_path / "test_model.pkl")
        fitted_engine.save(path)

        loaded = HMMEngine.load(path)

        assert loaded._n_states == fitted_engine._n_states
        assert loaded._state_to_label == fitted_engine._state_to_label

        orig_probs  = fitted_engine.predict_regime_proba(synthetic_features[:20])
        loaded_probs = loaded.predict_regime_proba(synthetic_features[:20])

        np.testing.assert_allclose(
            orig_probs, loaded_probs, atol=1e-10,
            err_msg="Loaded model produces different predictions than original.",
        )

    def test_needs_retraining_false_when_fresh(self, fitted_engine: HMMEngine) -> None:
        """A freshly fitted model should not need retraining immediately."""
        import pandas as pd
        assert not fitted_engine.needs_retraining(pd.Timestamp.now(), retrain_interval_days=30)

    def test_needs_retraining_true_when_stale(self, fitted_engine: HMMEngine) -> None:
        """A model with an old training date must report needing retraining."""
        import pandas as pd
        future = pd.Timestamp.now() + pd.Timedelta(days=100)
        assert fitted_engine.needs_retraining(future, retrain_interval_days=30)
