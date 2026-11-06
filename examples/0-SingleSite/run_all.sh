#!/usr/bin/env bash
# Run every site/adsorbate example with the packaged Hydra entry point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-adsorption-run}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

for site_dir in "$HERE"/*/; do
    for ads_dir in "$site_dir"/*/; do
        [[ -f "$ads_dir/run.yaml" ]] || continue
        if [[ "$SKIP_EXISTING" == "1" && -f "$ads_dir/results/summary.json" ]]; then
            continue
        fi
        echo "=== $(basename "$site_dir") / $(basename "$ads_dir") ==="
        (
            cd "$ads_dir"
            "$RUNNER" --config-path "$ads_dir" --config-name run
        )
    done
done
