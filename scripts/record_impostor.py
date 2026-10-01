#!/usr/bin/env python3
"""Record a global-minimum gate result as an impostor fixture.

An impostor (ADR-0007) is a point the objective rated at or below the
published structure while its lattice is more than 3% away. The global-
minimum gate writes its fitted vector to <results>/global_minimum_<name>.json;
this script copies it to tests/fixtures/impostors/<name>.json with its
provenance, for the impostor regression gate.

    python scripts/record_impostor.py gate-results/global_minimum_topaz.json \
        --found-under "pre uniqueness-plan phases A-G, 2026-10-01"
"""

import argparse
import json
from pathlib import Path

import yaml

from mizuho.diagnostics import lattice_names

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "impostors"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", help="global_minimum_<name>.json from the gate")
    ap.add_argument("--found-under", required=True,
                    help="which objective found it (code state, date)")
    ap.add_argument("--tag", default="",
                    help="suffix for the fixture file, e.g. 'round2' (ADR-0005 rounds)")
    args = ap.parse_args()
    src = Path(args.result)
    name = src.stem.removeprefix("global_minimum_")
    d = json.loads(src.read_text())
    pv, p = d["published_vector"], d["param_vector"]
    fixture = OUT.parent / f"{name}.yaml"
    sg = yaml.safe_load(fixture.read_text())["spacegroup"]
    names = lattice_names(sg)
    record = {
        "reference": name,
        "found_under": args.found_under,
        "error_found": d["error_found"],
        "error_published": d["error_published"],
        "param_vector": p,
        "published_vector": pv,
        "spacegroup": sg,
        "lattice_deviation_percent": {k: 100 * (p[i] - pv[i]) / pv[i]
                                      for i, k in enumerate(names) if k in ("a", "b", "c")},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"{name}_{args.tag}.json" if args.tag else f"{name}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"recorded {out}")


if __name__ == "__main__":
    main()
