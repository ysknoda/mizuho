#!/usr/bin/env python3
"""Measure constraint values from a reference CIF, per ADR-0004 and ADR-0007.

For each centre element with a cited coordination number, prints:

- the first shell (the cited-CN nearest neighbours of the opposite charge
  sign), the next neighbour after it, and the cutoff at the midpoint of
  that gap (ADR-0004);
- the bond band: the measured first-shell span widened by 2% of the bond
  length on each side, rounded outward to 0.01 A (ADR-0007, ADR-0004's
  amendment);
- the angle band: first-shell angles below the 150 deg trans threshold,
  rounded outward to 5 deg (ADR-0004).

Coordination numbers are inputs, never outputs: they are cited facts
(ADR-0004). The script only measures where the shell they name ends.

    python scripts/measure_reference.py tests/fixtures/reference/wurtzite.cif \
        --charges Zn=2,O=-2 --cn Zn=4,O=4
"""

import argparse
import math

import numpy as np

from mizuho.io.cif import cif_to_mizuho
from mizuho.lattice import all_image_vectors
from mizuho.structure import build_structure

TRANS_DEG = 150.0  # energy.py's default ignore_trans_above
BAND_FRACTION = 0.02  # ADR-0007


def _pairs(arg: str) -> dict:
    out = {}
    for item in arg.split(","):
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out


def measure(path: str, charges: dict, cns: dict) -> None:
    m = cif_to_mizuho(path)
    st = build_structure(m.param_vector, m.config())
    lat, frac = st["lattice_matrix"], st["frac_coords"]
    sp = np.array(st["species"])
    print(f"{path}: SG {m.spacegroup}, {m.label()}, idealised at {m.symprec} A")
    lp = st["lattice_params"]
    print("  lattice " + " ".join(f"{k}={float(v):.4f}" for k, v in lp.items()))

    for centre, cn in cns.items():
        cn = int(cn)
        opposite = [e for e in charges if charges[e] * charges[centre] < 0]
        ci = np.where(sp == centre)[0]
        ni = np.where(np.isin(sp, opposite))[0]
        vecs, norms = all_image_vectors(frac[ci], frac[ni], lat)
        first, nxt, angles = [], [], []
        for k in range(len(ci)):
            flat_n = norms[k].ravel()
            flat_v = vecs[k].reshape(-1, 3)
            order = np.argsort(flat_n)
            order = order[flat_n[order] > 1e-6]
            shell = order[:cn]
            first.extend(flat_n[shell])
            nxt.append(flat_n[order[cn]])
            v = flat_v[shell]
            for i in range(cn):
                for j in range(i + 1, cn):
                    c = v[i] @ v[j] / (np.linalg.norm(v[i]) * np.linalg.norm(v[j]))
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                    if ang <= TRANS_DEG:
                        angles.append(ang)
        d_lo, d_hi, d_next = min(first), max(first), min(nxt)
        cutoff = (d_hi + d_next) / 2
        band_lo = math.floor((d_lo * (1 - BAND_FRACTION)) * 100) / 100
        band_hi = math.ceil((d_hi * (1 + BAND_FRACTION)) * 100) / 100
        print(
            f"  {centre} (CN {cn}, neighbours {opposite}): first shell "
            f"{d_lo:.4f}-{d_hi:.4f} A, next {d_next:.4f} A, gap {d_next - d_hi:.3f} A"
        )
        print(
            f"      cutoff {cutoff:.3f}   band [{band_lo:.2f}, {band_hi:.2f}]", end=""
        )
        if angles:
            a_lo, a_hi = min(angles), max(angles)
            print(
                f"   angles {a_lo:.2f}-{a_hi:.2f} -> "
                f"[{5 * math.floor(a_lo / 5):.0f}, {5 * math.ceil(a_hi / 5):.0f}]"
            )
        else:
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cif")
    ap.add_argument("--charges", required=True, help="e.g. Zn=2,O=-2")
    ap.add_argument(
        "--cn", required=True, help="cited coordination numbers, e.g. Zn=4,O=4"
    )
    args = ap.parse_args()
    measure(args.cif, _pairs(args.charges), _pairs(args.cn))


if __name__ == "__main__":
    main()
