"""ADR-0026's ground-truth table row from an exhaustive run's record.

Reads the JSONL record that `spacegroup_search.record` writes (one line per
finished group) and ranks every recorded entry with `run`'s global ranking key
(`mizuho.cli._ranking_key`): entries at or below the threshold first, highest
point-group order first, then error. The ranking is over the raw recorded
entries, i.e. before supergroup promotion (ADR-0010) and topology-class
merging (ADR-0019), and an entry's group is the group it was optimised in.

    python scripts/exhaustive_table.py runs/exhaustive_adr0026/si.jsonl \
        --name Si --expected 227 194
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuho.cli import _ranking_key  # noqa: E402


def load_entries(path):
    """[(error, sg, entry)], one per recorded Phase 2 entry, and the total
    wall time in seconds summed over the recorded groups."""
    rows, seconds, groups = [], 0.0, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        groups += 1
        seconds += rec["seconds"]
        for e in rec["entries"]:
            err = 1e10 if e["error"] is None else e["error"]
            rows.append((err, int(rec["sg"]), e))
    return rows, seconds, groups


def label(sg, entry):
    return " ".join(f"{s['element']}@{s['letter']}" for s in entry["specs"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("record", help="exhaustive run record (JSONL)")
    p.add_argument("--name", default=None, help="compound name for the row")
    p.add_argument(
        "--expected",
        type=int,
        nargs="+",
        required=True,
        help="published space group(s); polytypes may be listed together",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="zero-error threshold of the ranking (default: 1e-3)",
    )
    p.add_argument("--top", type=int, default=20, help="ranked entries to list")
    args = p.parse_args(argv)

    rows, seconds, groups = load_entries(args.record)
    rows.sort(key=lambda r: _ranking_key(r[0], r[1], args.threshold))
    n_zero = sum(r[0] <= args.threshold for r in rows)

    print(
        f"record: {args.record}  groups recorded: {groups}  entries: {len(rows)}  "
        f"at or below {args.threshold:g}: {n_zero}  time: {seconds / 3600:.1f} h\n"
    )
    print(f"{'rank':>5} {'error':>10} {'sg':>4} {'det':>4}  assignment")
    for i, (err, sg, e) in enumerate(rows[: args.top], 1):
        print(f"{i:>5} {err:>10.3g} {sg:>4} {e['detected_sg']!s:>4}  {label(sg, e)}")

    print()
    cells = []
    for want in args.expected:
        by_run = next(((i, r) for i, r in enumerate(rows, 1) if r[1] == want), None)
        by_det = next(
            ((i, r) for i, r in enumerate(rows, 1) if r[2]["detected_sg"] == want), None
        )
        for how, hit in (("run in", by_run), ("detected as", by_det)):
            if hit is None:
                print(f"  {want}: no entry {how} it")
            else:
                i, (err, sg, e) = hit
                print(
                    f"  {want}: best entry {how} it ranks {i} "
                    f"(run in {sg}, detected {e['detected_sg']}, {label(sg, e)}, "
                    f"error {err:.3g})"
                )
        if by_run is not None:
            i, (err, sg, e) = by_run
            cells.append(f"{i} ({sg}, {label(sg, e)}, {err:.3g})")
        else:
            cells.append(f"- ({want} not reached)")

    err, sg, e = rows[0]
    print(
        "\n| compound | entries | at or below 1e-3 | rank 1 | published group's best rank | time |"
    )
    print(
        f"| {args.name or args.record} | {len(rows)} | {n_zero} | "
        f"{sg}, {label(sg, e)}, {err:.3g} | {'; '.join(cells)} | {seconds / 3600:.1f} h |"
    )


if __name__ == "__main__":
    main()
