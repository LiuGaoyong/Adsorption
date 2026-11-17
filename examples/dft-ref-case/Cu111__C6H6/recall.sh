#!/bin/bash
# Basin recall of this case; run it after relax.sbatch has finished.
# Run names are arm/rung@seed, so the seeds of the random arm are pooled.
# Extra options are passed on, e.g. ./recall.sh --window 0.5
set -euo pipefail
cd "$(dirname "$0")"
args=()
for dir in runs/deterministic/*/relaxed; do
    [ -f "$dir/relaxed.xyz" ] || continue
    args+=(--run "deterministic/$(basename "$(dirname "$dir")")=$dir")
done
for dir in runs/random/s*/*/relaxed; do
    [ -f "$dir/relaxed.xyz" ] || continue
    rung="$(basename "$(dirname "$dir")")"
    seed="$(basename "$(dirname "$(dirname "$dir")")")"
    args+=(--run "random/$rung@${seed#s}=$dir")
done
if [ -f manual-relaxed/relaxed.xyz ]; then
    args+=(--run "manual/paper=manual-relaxed")
fi
adsorption-relax recall "${args[@]}" --references reference.yaml \
    --out analysis "$@"
