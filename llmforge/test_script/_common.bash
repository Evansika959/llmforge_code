# shellcheck shell=bash
# Sourced by every test script. Sets up the env and exposes helpers.

set -e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs ckpts
TS="$(date +'%Y%m%d_%H%M%S')"

# Defaults shared across tests; individual scripts can override.
SURROGATE_CKPT=${SURROGATE_CKPT:-surrogate/ckpts/forgeformer.pt}
SEARCH_SPACE=${SEARCH_SPACE:-search_space_def/search_space_200M.yaml}
HOSTS_FILE=${HOSTS_FILE:-script/examples/hosts_example.yaml}

# Tiny scale for quick smoke
POP=${POP:-3}
OFFS=${OFFS:-2}
GENS=${GENS:-2}
LMAX=${LMAX:-12}
LMIN=${LMIN:-8}
MC=${MC:-3}

# HW measurement budget (small)
PFL=${PFL:-32}
DCL=${DCL:-8}
SQL=${SQL:-256}

# Real training budget — minimal so the cluster job lands fast
RT_ITERS=${RT_ITERS:-50}
RT_TIMEOUT=${RT_TIMEOUT:-1800}
# Poll cadence for remote-job completion — production runs leave the default
# (120s) but tests want ~5–10s so a finished training is detected quickly.
RT_POLL=${RT_POLL:-10}

# `t_start` lets each test print its own elapsed time at the end.
t_start() { _T0=$SECONDS; }
t_end()   { echo "[time] $((SECONDS - _T0))s"; }

# Run the unified driver, tee'ing into logs/${EXP}.log. Fails fast on error.
run_test() {
  local exp="$1"; shift
  local log="logs/${exp}.log"
  echo "[run] python -u run_cosearch.py --exp_name $exp $*"
  python -u run_cosearch.py --exp_name "$exp" "$@" 2>&1 | tee "$log"
  echo "[log] $log"
}

# Assert the per-gen JSON checkpoint was written for `gen`.
assert_ckpt() {
  local exp="$1"; local gen="$2"
  if ! ls "ckpts/${exp}/"*"_ckpt_gen${gen}.json" >/dev/null 2>&1; then
    echo "[FAIL] $exp: missing ckpts/${exp}/*_ckpt_gen${gen}.json"
    exit 1
  fi
}

pass() { echo "[PASS] $1 ($((SECONDS - _T0))s)"; }
