#!/usr/bin/env bash
# verify_parity.sh — run on BOTH Windows (Git Bash) and Linux, diff the output.
# Usage:  bash tools/verify_parity.sh > parity_$(uname -s).txt
# Then:   diff parity_Linux.txt parity_MINGW64_NT-10.0.txt

set -u
cd "$(dirname "$0")/.."

echo "=== GIT STATE ==="
git rev-parse HEAD
git status --porcelain
echo

echo "=== ACTIVE SET ==="
cat config/active_set 2>/dev/null || echo "MISSING"
echo

echo "=== TRACKED FILE HASHES (code + config + model) ==="
# md5sum on Linux, certutil fallback not needed because Git Bash ships md5sum
for f in \
    main.py \
    backtest/backtester.py \
    core/hmm_engine.py \
    data/feature_engineering.py \
    data/vix_fetcher.py \
    config/settings.yaml \
    config/sets/balanced.yaml \
    config/sets/conservative.yaml \
    config/sets/aggressive.yaml \
    models/hmm.pkl \
    requirements.txt ; do
    if [ -f "$f" ]; then
        md5sum "$f"
    else
        echo "MISSING  $f"
    fi
done
echo

echo "=== GIT-IGNORED / UNTRACKED FILES IN REPO ==="
# Every file git ignores or doesn't track — these are the divergence suspects
git ls-files --others --exclude-standard
git ls-files --others --ignored --exclude-standard
echo

echo "=== LOCAL DATA / CACHE DIRS (size + file count) ==="
for d in data/cache data/ohlcv data/vix models logs .pytest_cache __pycache__ ; do
    if [ -d "$d" ]; then
        n=$(find "$d" -type f | wc -l)
        s=$(du -sb "$d" 2>/dev/null | awk '{print $1}')
        echo "$d  files=$n  bytes=$s"
    fi
done
echo

echo "=== ENV VARS THAT AFFECT NUMERICS ==="
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
         NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS PYTHONHASHSEED \
         PYTHONIOENCODING ; do
    echo "$v=${!v:-<unset>}"
done
echo

echo "=== PYTHON + LIBRARY VERSIONS ==="
py -3.12 -c "
import sys, numpy, scipy, pandas, sklearn, hmmlearn
print('python', sys.version.split()[0])
print('numpy',  numpy.__version__)
print('scipy',  scipy.__version__)
print('pandas', pandas.__version__)
print('sklearn', sklearn.__version__)
print('hmmlearn', hmmlearn.__version__)
numpy.show_config()
" 2>/dev/null || python3.12 -c "
import sys, numpy, scipy, pandas, sklearn, hmmlearn
print('python', sys.version.split()[0])
print('numpy',  numpy.__version__)
print('scipy',  scipy.__version__)
print('pandas', pandas.__version__)
print('sklearn', sklearn.__version__)
print('hmmlearn', hmmlearn.__version__)
numpy.show_config()
"
