#!/usr/bin/env bash
# =============================================================================
# run_queue.sh - run model runner notebooks one after another (e.g. overnight)
# =============================================================================
#
# WHAT IT DOES
#   Runs runners/<version>/run_<model>_<version>.ipynb via papermill, one at a
#   time. Each runner executes the full pipeline (stages from its config.yaml).
#   A failed model is logged and the queue continues with the next model.
#
# ENVIRONMENT
#   The conda env is loaded automatically:
#     1. machine id from current.pc (e.g. 2 -> PC2)
#     2. conda_env for that PC from config/car_coll/<version>/config.yaml
#        (same name the runners use as Jupyter kernel, e.g. carfax26v1 on PC2)
#     3. conda.sh is sourced and the env is activated
#   You do NOT need to activate anything before running this script.
#
# USAGE (from anywhere; paths are resolved from the project root)
#   bash runners/run_runners/run_queue.sh <version> [options] [model ...]
#
# ARGUMENTS
#   <version>        Required. Runner folder / config version, e.g. v3
#                    -> runs runners/v3/run_<model>_v3.ipynb
#   [model ...]      Optional. Models to run, in this order, e.g.
#                    car_coll suv_pip van_liab
#                    Default: every run_*_<version>.ipynb in runners/<version>/
#                    (alphabetical: car -> suv -> truck -> van,
#                    each coll, comp, liab, pip)
#
# OPTIONS
#   --dry-run        Show machine, env, python and the queue, then exit.
#                    Nothing is executed. Use this first on a new machine.
#   --stop-on-error  Stop the whole queue at the first failed model.
#                    Default: log the failure and continue with the next model.
#   --env <name>     Use this conda env instead of the one in config.yaml.
#                    Same as setting CONDA_ENV=<name> before the command.
#   -h, --help       Show this help.
#
# STATUS PER MODEL (in queue.log)
#   OK               Runner finished, all enabled stages succeeded.
#   STAGE FAILED     Runner finished but a stage failed
#                    ("PIPELINE COMPLETED WITH ERRORS" in the executed runner).
#                    Stage notebook with the error: <output_path>/notebooks/
#   RUNNER CRASHED   papermill itself failed (bad kernel, runner error, ...).
#   NOT FOUND        runners/<version>/run_<model>_<version>.ipynb missing.
#
# OUTPUT
#   logs/queue_<version>_<YYYYmmdd_HHMMSS>/
#     queue.log                      start/end/duration/status per model + summary
#     run_<model>_<version>.ipynb    executed runner notebook (full stage output)
#     run_<model>_<version>.log      papermill console output
#
# EXAMPLES
#   # check environment and queue (no execution)
#   bash runners/run_runners/run_queue.sh v3 --dry-run
#
#   # all 16 v3 models
#   bash runners/run_runners/run_queue.sh v3
#
#   # selected models, in this order
#   bash runners/run_runners/run_queue.sh v3 suv_coll suv_comp van_pip
#
#   # overnight in background - keeps running if the terminal is closed
#   nohup bash runners/run_runners/run_queue.sh v3 > /dev/null 2>&1 &
#   tail -f logs/queue_v3_*/queue.log      # Ctrl-C stops watching, not the job
#
#   # alternative: tmux (reattach later and watch live)
#   tmux new -s overnight                  # open a server-side terminal
#   bash runners/run_runners/run_queue.sh v3
#   #   Ctrl-b then d                      # detach, job keeps running
#   tmux attach -t overnight               # reattach later
#
#   # stop a running queue
#   pkill -f run_queue.sh; pkill -f papermill
# =============================================================================

set -u

# Print the header comment block above as help text
usage() { awk '/^# =====/ { n++; if (n == 3) exit; next } n >= 1 { sub(/^# ?/, ""); print }' "$0"; }

# ---------------------------------------------------------------- arguments --
VERSION=""
DRY_RUN=0
STOP_ON_ERROR=0
ENV_OVERRIDE="${CONDA_ENV:-}"
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1 ;;
        --stop-on-error) STOP_ON_ERROR=1 ;;
        --env)           shift; ENV_OVERRIDE="${1:-}" ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "Unknown option: $1 (see --help)"; exit 2 ;;
        *)               if [[ -z "$VERSION" ]]; then VERSION="$1"; else MODELS+=("$1"); fi ;;
    esac
    shift
done

if [[ -z "$VERSION" ]]; then
    echo "ERROR: version required, e.g.: bash runners/run_runners/run_queue.sh v3"
    exit 2
fi

# ------------------------------------------------------------- project root --
# Search upward from this script for current.pc (same rule as the runners)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != "/" && ! -f "$ROOT/current.pc" ]]; do ROOT="$(dirname "$ROOT")"; done
if [[ ! -f "$ROOT/current.pc" ]]; then
    echo "ERROR: current.pc not found above $(dirname "${BASH_SOURCE[0]}")"
    exit 1
fi
cd "$ROOT" || exit 1

PC_NUM="$(tr -d '[:space:]' < current.pc)"
PC_ID="PC${PC_NUM}"
RUNNER_DIR="runners/${VERSION}"
if [[ ! -d "$RUNNER_DIR" ]]; then
    echo "ERROR: $RUNNER_DIR not found"
    exit 1
fi

# --------------------------------------------------------------- conda env --
# conda_env of this PC from a config of this version (all configs share it)
REF_CFG="config/car_coll/${VERSION}/config.yaml"
[[ -f "$REF_CFG" ]] || REF_CFG="$(ls config/*/"${VERSION}"/config.yaml 2> /dev/null | head -1)"

ENV_NAME="$ENV_OVERRIDE"
if [[ -z "$ENV_NAME" && -n "$REF_CFG" ]]; then
    ENV_NAME="$(awk -v pc="$PC_ID" '
        $0 ~ "^[[:space:]]+"pc":"             { inpc = 1; next }
        inpc && $0 ~ "^[[:space:]]+PC[0-9]+:" { inpc = 0 }
        inpc && $1 == "conda_env:"            { v = $2; gsub(/["\047]/, "", v); print v; exit }
    ' "$REF_CFG")"
fi
if [[ -z "$ENV_NAME" ]]; then
    echo "ERROR: no conda_env for $PC_ID in $REF_CFG (use --env <name>)"
    exit 1
fi

# Load conda into this non-interactive shell, then activate the env
CONDA_BASE=""
if command -v conda > /dev/null 2>&1; then CONDA_BASE="$(conda info --base 2> /dev/null)"; fi
for b in "$CONDA_BASE" "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda" "/opt/anaconda3"; do
    if [[ -n "$b" && -f "$b/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$b"; break; fi
done
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found (looked in conda info --base, ~/anaconda3, ~/miniconda3, /opt/conda)"
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda activate "$ENV_NAME"; then
    echo "ERROR: conda activate $ENV_NAME failed (list envs: conda env list)"
    exit 1
fi
# Use the env's python directly (pyenv shims on the Mac can shadow conda in PATH)
export PATH="$CONDA_PREFIX/bin:$PATH"
PY="$CONDA_PREFIX/bin/python"
if ! "$PY" -c "import papermill" 2> /dev/null; then
    echo "ERROR: papermill not installed in env $ENV_NAME ($PY)"
    exit 1
fi

# ------------------------------------------------------------------- queue --
# Default queue: every runner in runners/<version>/ (alphabetical)
if [[ ${#MODELS[@]} -eq 0 ]]; then
    for f in "$RUNNER_DIR"/run_*_"${VERSION}".ipynb; do
        [[ -f "$f" ]] || continue
        m="$(basename "$f" .ipynb)"; m="${m#run_}"; m="${m%_${VERSION}}"
        MODELS+=("$m")
    done
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no runners found in $RUNNER_DIR"
    exit 1
fi

echo "Project root : $ROOT"
echo "Machine      : $PC_ID"
echo "Conda env    : $ENV_NAME ($PY)"
echo "Runners      : $RUNNER_DIR"
echo "Queue (${#MODELS[@]})   : ${MODELS[*]}"
echo "On error     : $([[ $STOP_ON_ERROR -eq 1 ]] && echo stop || echo continue)"

if [[ $DRY_RUN -eq 1 ]]; then
    for m in "${MODELS[@]}"; do
        nb="$RUNNER_DIR/run_${m}_${VERSION}.ipynb"
        if [[ -f "$nb" ]]; then echo "  [ok]        $nb"; else echo "  [NOT FOUND] $nb"; fi
    done
    echo "Dry run - nothing executed."
    exit 0
fi

# --------------------------------------------------------------------- run --
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/queue_${VERSION}_${STAMP}"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }
fmt() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

log "QUEUE START  version=$VERSION machine=$PC_ID env=$ENV_NAME models=${#MODELS[@]}"
log "Models: ${MODELS[*]}"

SUMMARY=()
Q_START=$SECONDS
for m in "${MODELS[@]}"; do
    nb="$RUNNER_DIR/run_${m}_${VERSION}.ipynb"
    out="$LOG_DIR/run_${m}_${VERSION}.ipynb"
    if [[ ! -f "$nb" ]]; then
        log "$m: NOT FOUND ($nb)"
        SUMMARY+=("$(printf '%-12s %-15s %s' "$m" "NOT FOUND" "-")")
        [[ $STOP_ON_ERROR -eq 1 ]] && break
        continue
    fi

    log "$m: START"
    t0=$SECONDS
    # papermill exit code only covers the runner itself; stage failures are
    # caught by the runner and reported as text in the executed notebook
    if "$PY" -m papermill -k "$ENV_NAME" "$nb" "$out" > "$LOG_DIR/run_${m}_${VERSION}.log" 2>&1; then
        if grep -q "PIPELINE COMPLETED WITH ERRORS" "$out"; then status="STAGE FAILED"
        else status="OK"; fi
    else
        status="RUNNER CRASHED"
    fi
    dur="$(fmt $(( SECONDS - t0 )))"
    log "$m: $status ($dur)"
    SUMMARY+=("$(printf '%-12s %-15s %s' "$m" "$status" "$dur")")

    if [[ "$status" != "OK" && $STOP_ON_ERROR -eq 1 ]]; then
        log "Stopping queue (--stop-on-error)"
        break
    fi
done

log "QUEUE END  total $(fmt $(( SECONDS - Q_START )))"
{
    echo ""
    echo "SUMMARY"
    printf '%-12s %-15s %s\n' "model" "status" "duration"
    printf '%s\n' "${SUMMARY[@]}"
} | tee -a "$QLOG"
