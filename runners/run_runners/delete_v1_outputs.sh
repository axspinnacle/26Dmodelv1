#!/usr/bin/env bash
# Delete the v1 output folders on prod (PC2). Configs and runners are not touched.
# Dry run by default: lists folders and sizes. Add --yes to delete.
#   bash runners/run_runners/delete_v1_outputs.sh          # list only
#   bash runners/run_runners/delete_v1_outputs.sh --yes    # delete

set -euo pipefail

BASE="/home/carfax_shared/dmodel_carfax/exploration/Dmodel_output_data"
MODELS="car_coll car_comp car_liab car_pip suv_coll suv_comp suv_liab suv_pip
        truck_coll truck_comp truck_liab truck_pip van_coll van_comp van_liab van_pip"

DELETE=0
[[ "${1:-}" == "--yes" ]] && DELETE=1

[[ -d "$BASE" ]] || { echo "Not found: $BASE (run this on prod)"; exit 1; }

found=0
for m in $MODELS; do
    d="$BASE/$m/v1"
    [[ -d "$d" ]] || continue
    found=$((found + 1))
    echo "$(du -sh "$d" | cut -f1)  $d"
done

if [[ $found -eq 0 ]]; then
    echo "No v1 output folders left."
    exit 0
fi

echo "Total: $(for m in $MODELS; do [[ -d "$BASE/$m/v1" ]] && echo "$BASE/$m/v1"; done | xargs du -sch | tail -1 | cut -f1) in $found folders"

if [[ $DELETE -eq 0 ]]; then
    echo "Dry run. Re-run with --yes to delete."
    exit 0
fi

for m in $MODELS; do
    d="$BASE/$m/v1"
    [[ -d "$d" ]] && rm -rf -- "$d" && echo "Deleted $d"
done
echo "Done."
