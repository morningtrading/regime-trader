# Per-Group Parameters — Plan (non implémenté)

> Statut : **documentation uniquement**. Ne pas implémenter sans validation.
> Date : 2026-04-18

## 1. Pourquoi on en parle

Actuellement `config/settings.yaml` applique les mêmes paramètres HMM / strategy / risk à **tous** les groupes d'actifs (stocks, crypto, indices, midcap, forex_etf). Or :

- Les midcaps growth (AXON, CRDO, POWL…) ont une volatilité et un régime trendy très différents des large-cap liquides (SPY, QQQ).
- Le crypto tourne 24/7 avec des queues de distribution épaisses → `stability_bars=5` bars de 5min ≠ 5 bars crypto.
- Les indices ETF diversifiés ont des régimes lents → `min_confidence=0.62` peut être trop restrictif.
- Le sweep `cs-sweep` actuel a été validé **sur indices 2020-2026** et appliqué aveuglément partout.

Conséquence : on sous-exploite les groupes où un tuning dédié donnerait un edge.

## 2. Architecture proposée

Hiérarchie à **3 niveaux** de résolution des paramètres :

```
settings.yaml (base)
    ↓ overlay
config set actif (conservative | balanced | aggressive)
    ↓ overlay
asset_groups.yaml → group.overrides (NOUVEAU)
```

Exemple `config/asset_groups.yaml` :

```yaml
groups:
  midcap:
    class: equity
    tags: [midcap, growth, us]
    symbols: [AXON, CRDO, FIX, POWL, KTOS, CACI, AEIS, ONTO, FTAI, IBP]
    overrides:
      hmm:
        min_confidence: 0.55
        stability_bars: 3
      strategy:
        low_vol_leverage: 1.0       # pas de levier sur midcap
        trend_lookback: 30
      risk:
        max_single_position: 0.10   # concentration réduite
```

Resolution : `resolve_config(group_name) = deep_merge(base, config_set, group.overrides)`.

## 3. Paramètres à customiser par groupe

Classification ROI :

| Param | ROI | Raison |
|---|---|---|
| 🟢 `hmm.min_confidence` | Haut | Signal-to-noise très dépendant du groupe |
| 🟢 `hmm.stability_bars` | Haut | Vitesse de régime ≠ par classe |
| 🟢 `strategy.low_vol_leverage` | Haut | Midcap/crypto ≠ SPY en leverage safe |
| 🟢 `risk.max_single_position` | Haut | Concentration doit matcher la liquidité |
| 🟡 `strategy.trend_lookback` | Moyen | Optim par régime |
| 🟡 `hmm.flicker_threshold` | Moyen | Bruit ≠ par classe |
| 🟡 `risk.max_concurrent` | Moyen | Selon taille univers |
| 🔴 `hmm.n_candidates` | Bas | Grille large suffit |
| 🔴 `backtest.initial_capital` | Bas | Sans impact sur le signal |
| 🔴 `broker.timeframe` | Bas | Déjà géré ailleurs |

## 4. Conséquences positives

1. **Meilleure performance par groupe** — chaque classe optimisée sur sa propre distribution.
2. **Risque mieux calibré** — leverage/concentration adaptés à la liquidité réelle.
3. **Sweeps ciblés** — `cs-sweep --asset-group midcap` sans polluer les autres groupes.
4. **Expérimentation isolée** — tester un param agressif sur midcap sans casser stocks.
5. **Documentation naturelle** — chaque groupe porte sa logique dans le YAML.

## 5. Conséquences négatives / risques

| Risque | Mitigation |
|---|---|
| **Overfitting par groupe** (10 symboles, 5 params → facile de curve-fit) | Forward test obligatoire (hold-out 2024+) avant de figer |
| **Dette de maintenance** (5 groupes × N params à re-sweeper) | Cadence trimestrielle light, annuelle full ; scripts automatisés |
| **Complexité debug** (quel param a gagné ? base, set, ou override ?) | Logger la config résolue au démarrage de chaque run |
| **Incohérence entre groupes** (un groupe live avec params stale) | Timestamp `last_swept` dans le YAML + warning si > 6 mois |
| **Explosion de combinatoire** (5 groupes × 3 config sets × N overrides) | Limiter overrides aux 4 params 🟢 au départ |

## 6. Fréquence de re-sweep

**Déclencheurs** (par événement) :
- Changement de régime macro majeur (crise, hausse taux soudaine)
- Sharpe out-of-sample dégradé > 30% vs backtest
- Ajout/retrait de > 30% des symboles d'un groupe
- Changement de timeframe ou de feature set

**Cadence temporelle** :
- **Trimestriel** : re-sweep léger (conf × stab uniquement) sur données des 3 derniers mois
- **Annuel** : re-sweep complet (interval + cs + features) sur fenêtre roulante 3 ans
- **Ad-hoc** : après toute modification majeure du code HMM/strategy

## 7. Plan d'implémentation en phases (~8h)

**Phase 1 — Plumbing (~2h)**
- Ajouter `overrides:` optionnel dans `core/asset_groups.py`
- Fonction `resolve_config(base, group_name)` avec deep_merge
- Logger la config résolue au startup
- Tests unitaires du merge

**Phase 2 — Exposer dans les commandes (~1h)**
- `trade`, `backtest`, `sweep`, `cs-sweep` utilisent la config résolue
- `main.py groups show <name>` affiche les overrides effectifs

**Phase 3 — Sweeps par groupe (~3h)**
- Lancer `cs-sweep` pour chaque groupe individuellement
- Capturer les résultats dans `savedresults/sweeps/<group>_<date>/`
- Mettre à jour les `overrides:` avec les optima trouvés
- Forward-tester chaque groupe sur hold-out 2024+

**Phase 4 — Monitoring (~2h)**
- Ajouter `last_swept` timestamp par groupe
- Warning si > 6 mois depuis dernier sweep
- Dashboard comparatif avant/après overrides

## 8. Alternative — ne pas implémenter

**Pour** :
- Simplicité de maintenance (1 seul fichier de config)
- Moins de risque d'overfitting
- Cohérence entre groupes

**Contre** :
- On laisse de la performance sur la table (probablement 10-30% Sharpe)
- Impossible d'ajouter un groupe exotique (forex_etf, crypto small-cap) sans compromettre les autres

## 9. Recommandation

**Implémenter en phases avec discipline** :
1. Phase 1+2 d'abord (plumbing sans changer le comportement)
2. Un seul groupe piloté (midcap ou crypto) pour valider l'approche
3. Extension aux autres groupes seulement si gain OOS démontré
4. Ne jamais overrider les 🔴, commencer par les 4 🟢 seulement

**Pas avant** d'avoir :
- Un test de régression automatisé (Backtest Quick en CI)
- Forward tests 2024+ stables sur la config actuelle
- Un process de sweep reproductible
