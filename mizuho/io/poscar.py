"""Write a candidate structure to a VASP POSCAR file."""


def write_poscar(structure: dict, config: dict, filepath: str):
    lp  = structure["lattice_params"]
    lat = structure["lattice_matrix"]
    sp  = structure["species"]
    fc  = structure["frac_coords"]

    seen_order = []
    for s in sp:
        if s not in seen_order:
            seen_order.append(s)

    lines = [
        f"{config['composition']} | SG {config['spacegroup']} "
        f"{config.get('spacegroup_symbol','')} | "
        f"a={lp['a']:.4f} b={lp['b']:.4f} c={lp['c']:.4f} A | "
        f"alpha={lp.get('alpha',90):.2f} beta={lp.get('beta',90):.2f} "
        f"gamma={lp.get('gamma',90):.2f} deg",
        "1.0",
    ]
    for vec in lat:
        lines.append(f"  {vec[0]:15.9f}  {vec[1]:15.9f}  {vec[2]:15.9f}")
    lines.append("  " + "  ".join(seen_order))
    lines.append("  " + "  ".join(str(sp.count(e)) for e in seen_order))
    lines.append("Direct")
    for elem in seen_order:
        for i, s in enumerate(sp):
            if s == elem:
                f = fc[i]
                lines.append(
                    f"  {f[0]:15.9f}  {f[1]:15.9f}  {f[2]:15.9f}  {elem}")

    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  POSCAR written: {filepath}")
