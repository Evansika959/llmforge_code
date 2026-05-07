#!/bin/bash
# Orchestrator: runs the full test suite and prints a PASS/FAIL summary.
#
# Tiers (default = fast only):
#   fast   01,02,08,09,10    surrogate-only, ~1-2 min total
#   slow   +03,04,05         Timeloop substrates (eyeriss/dxe/rdxe), 5-15 min
#   remote +06,07            real_train + finetune; cluster-bound, ~10-20 min
#
# Flags:
#   --slow      include Timeloop tests
#   --remote    include real-training tests (require --realtrain_hosts_file
#               default path or HOSTS_FILE override)
#   --all       slow + remote
#   --keep      do not delete ckpts/_test_* directories at exit

set -u
set -o pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."

INCLUDE_SLOW=0
INCLUDE_REMOTE=0
KEEP=0
for a in "$@"; do
  case "$a" in
    --slow)   INCLUDE_SLOW=1 ;;
    --remote) INCLUDE_REMOTE=1 ;;
    --all)    INCLUDE_SLOW=1; INCLUDE_REMOTE=1 ;;
    --keep)   KEEP=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown flag: $a"; exit 2 ;;
  esac
done

FAST=(01_surrogate_none 08_seed_arch 09_init_individuals 02_surrogate_zeus 11_kv_cache 10_resume_ckpt)
SLOW=(03_surrogate_timeloop_eyeriss 04_surrogate_timeloop_dxe 05_surrogate_timeloop_rdxe)
REMOTE=(06_real_train_none 07_finetune_none)

TESTS=("${FAST[@]}")
[[ $INCLUDE_SLOW   -eq 1 ]] && TESTS+=("${SLOW[@]}")
[[ $INCLUDE_REMOTE -eq 1 ]] && TESTS+=("${REMOTE[@]}")

declare -A RESULT
declare -A ELAPSED

for t in "${TESTS[@]}"; do
  echo
  echo "============================================================"
  echo "[suite] running $t"
  echo "============================================================"
  s=$SECONDS
  if bash "test_script/${t}.bash"; then
    RESULT[$t]="PASS"
  else
    RESULT[$t]="FAIL"
  fi
  ELAPSED[$t]=$((SECONDS - s))
done

echo
echo "============================================================"
echo "Suite summary"
echo "============================================================"
n_pass=0; n_fail=0
for t in "${TESTS[@]}"; do
  printf "  %-40s %4s  %ds\n" "$t" "${RESULT[$t]}" "${ELAPSED[$t]}"
  [[ "${RESULT[$t]}" == "PASS" ]] && n_pass=$((n_pass+1)) || n_fail=$((n_fail+1))
done
echo "------------------------------------------------------------"
echo "  ${n_pass} passed, ${n_fail} failed (of ${#TESTS[@]} run)"

if [[ $KEEP -eq 0 ]]; then
  echo "[cleanup] removing ckpts/_test_* (use --keep to preserve)"
  rm -rf ckpts/_test_*
fi

[[ $n_fail -eq 0 ]]
