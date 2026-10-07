"""Lowest-energy frame of a VASP relaxation, written as a POSCAR
(design review 2026-10-02, Q14(a)): a chunk restarts from its best ionic
step instead of its last, so a line search that goes wrong at the end of a
chunk is not carried into the next one."""

import re
from pathlib import Path

import numpy as np


def xdatcar_frames(path):
    """[(header_lines, coord_lines)] for every frame of a variable-cell
    XDATCAR (VASP 6 writes the cell before every configuration)."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        header = lines[i:i + 7]
        n = sum(int(x) for x in header[6].split())
        assert lines[i + 7].strip().startswith("Direct configuration"), lines[i + 7]
        frames.append((header, lines[i + 8:i + 8 + n]))
        i += 8 + n
    return frames


def step_energies(oszicar):
    return [float(re.search(r"E0=\s*([-\d.Ee+]+)", l).group(1))
            for l in Path(oszicar).read_text().splitlines() if "F=" in l]


def best_frame(directory):
    """(index, energy, poscar_text) of the lowest-E0 ionic step."""
    d = Path(directory)
    energies = step_energies(d / "OSZICAR")
    frames = xdatcar_frames(d / "XDATCAR")
    assert len(frames) >= len(energies), (len(frames), len(energies))
    k = int(np.argmin(energies))
    header, coords = frames[k]
    text = "\n".join(header + ["Direct"] + coords) + "\n"
    return k, energies[k], text
