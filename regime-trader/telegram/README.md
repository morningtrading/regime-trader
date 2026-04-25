# Telegram Notifications — Regime Trader

Module isolé. Ne touche à aucun autre fichier du projet.

---

## 1. Créer le bot (2 minutes)

1. Ouvrir Telegram → chercher **@BotFather**
2. Taper `/newbot`
3. Donner un nom (ex: `Regime Trader`) et un username (ex: `regime_trader_bot`)
4. BotFather envoie un **token** → le copier

---

## 2. Récupérer ton Chat ID

Envoyer n'importe quel message à ton bot, puis ouvrir dans un navigateur :

```
https://api.telegram.org/bot<TON_TOKEN>/getUpdates
```

Chercher `"chat":{"id":` dans la réponse → c'est ton Chat ID.

---

## 3. Configurer les credentials

**Option A — `.env`** (recommandé, déjà gitignored) :
```
TELEGRAM_TOKEN=1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=987654321
```

**Option B — `config/credentials.yaml`** :
```yaml
telegram:
  token: "1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  chat_id: 987654321
```

---

## 4. Tester

```bash
cd ~/regime-trader/darkvador/regime-trader
source .venv/bin/activate

python telegram/hooks.py test
```

Tu dois recevoir un message sur Telegram.

---

## 5. Commandes disponibles

```bash
python telegram/hooks.py test        # test de connexion
python telegram/hooks.py summary     # résumé du dernier backtest (4 lignes)
python telegram/hooks.py trades      # 5 derniers trades
python telegram/hooks.py stress      # résultats stress test
python telegram/hooks.py regime      # régime HMM actuel
python telegram/hooks.py all         # tout envoyer
```

---

## 6. Exemples de messages reçus

**Résumé backtest :**
```
📊 Backtest — stocks4
GLD, JNJ, LMT, NVDA, UNH, WMT
Période  : 2021-01-01 → 2026-04-24  (13 folds)
Retour   : +103.19%   Sharpe 1.98
MaxDD    : -5.60%   Calmar 4.39
24 Apr 2026  21:15 UTC
```

**Derniers trades :**
```
📈 Derniers trades — stocks4
  🟢 NVDA   +2.31%   2026-04-22
  🔴 JNJ    -0.84%   2026-04-21
  🟢 GLD    +1.12%   2026-04-20
  🟢 UNH    +0.93%   2026-04-18
  🟢 LMT    +1.45%   2026-04-17
24 Apr 2026  21:15 UTC
```

---

## Structure des fichiers

```
telegram/
├── __init__.py     — package marker
├── config.py       — lecture des credentials (.env / credentials.yaml)
├── bot.py          — send(text) → appel API Telegram
├── formatter.py    — formatage des messages depuis savedresults/
├── hooks.py        — CLI : python telegram/hooks.py <commande>
└── README.md       — ce fichier
```

**Zéro modification** aux autres fichiers du projet.
