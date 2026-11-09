#!/usr/bin/env bash
# Run every flexible-adsorbate system/adsorbate example with the Hydra entry point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-adsorption-run}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
START="${START:-}"
STOP="${STOP:-}"
N_JOBS="${N_JOBS:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$HERE/../../.venv/bin/python" ]]; then
        PYTHON="$HERE/../../.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
DEFAULT_NEQUIP_MODEL="$HOME/.local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth"
if [[ -z "${NEQUIP_MODEL:-}" && -f "$DEFAULT_NEQUIP_MODEL" ]]; then
    NEQUIP_MODEL="$DEFAULT_NEQUIP_MODEL"
fi
export NEQUIP_MODEL

TEMPLATE="${TEMPLATE:-}"
if [[ -z "$TEMPLATE" ]]; then
    if [[ -n "${NEQUIP_MODEL:-}" ]]; then
        TEMPLATE="run.nequip.template.yaml"
    else
        TEMPLATE="run.template.yaml"
    fi
fi
MANIFEST="${MANIFEST:-manifest.yaml}"
if [[ "$TEMPLATE" != /* && ! -f "$TEMPLATE" ]]; then
    TEMPLATE="$HERE/$TEMPLATE"
fi
if [[ "$MANIFEST" != /* && ! -f "$MANIFEST" ]]; then
    MANIFEST="$HERE/$MANIFEST"
fi
GENERATE_CASES="${GENERATE_CASES:-1}"
BUILD="$HERE/../build_cases.py"
OVERRIDES=""
if [[ -n "$START" ]]; then OVERRIDES="$OVERRIDES start=$START"; fi
if [[ -n "$STOP" ]]; then OVERRIDES="$OVERRIDES stop=$STOP"; fi
if [[ -n "$N_JOBS" ]]; then OVERRIDES="$OVERRIDES parallel.n_jobs=$N_JOBS"; fi

if [[ "$GENERATE_CASES" == "1" ]]; then
    "$PYTHON" "$BUILD" "$MANIFEST" "$TEMPLATE" --root "$HERE"
fi

for system_dir in "$HERE"/*/; do
    for ads_dir in "$system_dir"/*/; do
        [[ -f "$ads_dir/run.yaml" ]] || continue
        if [[ "$SKIP_EXISTING" == "1" && -f "$ads_dir/results/summary.json" ]]; then
            continue
        fi
        echo "=== $(basename "$system_dir") / $(basename "$ads_dir") ==="
        if ! (
            cd "$ads_dir"
            "$RUNNER" --config-path "$ads_dir" --config-name run $OVERRIDES
        ); then
            echo "FAILED: $(basename "$system_dir") / $(basename "$ads_dir")" >&2
            if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
                exit 1
            fi
        fi
    done
done
