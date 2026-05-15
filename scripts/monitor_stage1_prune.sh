#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-./log/tpcsd}"
RUN_NAME=""
LOG_FILE=""
MIN_EPOCH="${MIN_EPOCH:-25}"
PATIENCE="${PATIENCE:-12}"
MIN_DELTA="${MIN_DELTA:-0.002}"
PUNIF_RATIO_TH="${PUNIF_RATIO_TH:-0.35}"
PUNIF_STREAK_TH="${PUNIF_STREAK_TH:-3}"
POLL_SEC="${POLL_SEC:-20}"

usage() {
  echo "Usage: $0 --run-name <run_name> [--log-file <path>]"
  echo "Env overrides: MIN_EPOCH PATIENCE MIN_DELTA PUNIF_RATIO_TH PUNIF_STREAK_TH POLL_SEC LOG_ROOT"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"; shift 2 ;;
    --log-file)
      LOG_FILE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${LOG_FILE}" ]]; then
  [[ -n "${RUN_NAME}" ]] || { echo "--run-name is required when --log-file is not provided" >&2; exit 1; }
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
fi

echo "[MONITOR] file=${LOG_FILE}"
echo "[MONITOR] min_epoch=${MIN_EPOCH} patience=${PATIENCE} min_delta=${MIN_DELTA} punif_ratio_th=${PUNIF_RATIO_TH} punif_streak_th=${PUNIF_STREAK_TH}"

while true; do
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "[WAIT] log not found yet: ${LOG_FILE}"
    sleep "${POLL_SEC}"
    continue
  fi

  python - "$LOG_FILE" "$MIN_EPOCH" "$PATIENCE" "$MIN_DELTA" "$PUNIF_RATIO_TH" "$PUNIF_STREAK_TH" <<'PY'
import re
import sys

log_file = sys.argv[1]
min_epoch = int(sys.argv[2])
patience = int(sys.argv[3])
min_delta = float(sys.argv[4])
punif_ratio_th = float(sys.argv[5])
punif_streak_th = int(sys.argv[6])

pat = re.compile(r"Epoch\s+(\d+)/(\d+).+?loss=([0-9.eE+-]+).+?loss_punif=([0-9.eE+-]+).+?val_bacc=([0-9.eE+-]+)")
rows = []
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        loss = float(m.group(3))
        punif = float(m.group(4))
        vb = float(m.group(5))
        rows.append((ep, loss, punif, vb))

if not rows:
    print("[WAIT] no parsed epoch lines yet")
    sys.exit(0)

rows.sort(key=lambda x: x[0])
last_ep, last_loss, last_punif, last_vb = rows[-1]

best_v = -1.0
best_ep = -1
for ep, _, _, vb in rows:
    if vb > best_v:
        best_v = vb
        best_ep = ep

wait = last_ep - best_ep

# consecutive punif/loss spikes
streak = 0
for ep, loss, punif, _ in reversed(rows):
    ratio = punif / max(loss, 1e-8)
    if ratio > punif_ratio_th:
        streak += 1
    else:
        break

ratio_last = last_punif / max(last_loss, 1e-8)

status = "CONTINUE"
reasons = []
if last_ep >= min_epoch and wait >= patience:
    status = "PRUNE"
    reasons.append(f"val_bacc plateau: best={best_v:.4f}@ep{best_ep}, wait={wait}")
if streak >= punif_streak_th:
    status = "PRUNE"
    reasons.append(f"punif/loss spike streak={streak}, last_ratio={ratio_last:.3f}")

print(
    f"[CHECK] ep={last_ep} val_bacc={last_vb:.4f} best={best_v:.4f}@ep{best_ep} "
    f"wait={wait} punif_ratio_last={ratio_last:.3f} streak={streak} => {status}"
)
if reasons:
    print("[WHY] " + " | ".join(reasons))
PY

  sleep "${POLL_SEC}"
done
