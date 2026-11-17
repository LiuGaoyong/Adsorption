#!/bin/bash
#SBATCH --job-name=relax-Pd111__C6H6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=relax-%j.out
# Relax the placements of this case with the potential of relax.yaml.
# Template only: adapt the partition, the environment and the CPU count.
# One `adsorption-relax run` per campaign (arm, seed, rung); every run
# resumes if the job is submitted again.
#
#   STAGE=manual sbatch relax.sbatch            # the gate of the potential
#   STAGE=sweep RUNGS="g00100 g00300" sbatch relax.sbatch
#   sbatch relax.sbatch                         # everything
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # submit from the case folder
ACTORS="${ACTORS:-${SLURM_CPUS_PER_TASK:-8}}"
STAGE="${STAGE:-all}"   # manual | sweep | ablation | all
RUNGS="${RUNGS:-}"      # rungs to relax, e.g. "g00100 g00300"; empty: all

relax() {  # relax <extxyz> <output directory> [extra options]
    local xyz="$1" out="$2"
    shift 2
    [ -s "$xyz" ] || return 0   # a campaign that accepted nothing
    adsorption-relax run "$xyz" --config relax.yaml --out "$out" \
        --engine ray --n-jobs "$ACTORS" --cpus-per-actor 1 "$@" </dev/null
}

campaigns() {  # accepted.xyz of the selected rungs, the smallest rung first
    local xyz rung
    find runs -name accepted.xyz | awk -F/ '{print $(NF-2) "\t" $0}' \
        | sort | cut -f2 | while read -r xyz; do
        rung="$(basename "$(dirname "$(dirname "$xyz")")")"
        if [ -z "$RUNGS" ] || [[ " $RUNGS " == *" $rung "* ]]; then
            echo "$xyz"
        fi
    done
}

# Hand-placed starts in the manner of the paper. They are also the gate of
# the potential: check them before the sweep below is trusted.
if [ "$STAGE" = manual ] || [ "$STAGE" = all ]; then
    relax manual.xyz manual-relaxed
fi

# The two arms: runs/deterministic/<rung> and runs/random/s<seed>/<rung>.
if [ "$STAGE" = sweep ] || [ "$STAGE" = all ]; then
    campaigns | while read -r xyz; do
        relax "$xyz" "$(dirname "$(dirname "$xyz")")/relaxed"
    done
fi
