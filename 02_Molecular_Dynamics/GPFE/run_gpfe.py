#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch calculation of full GPFE/GSFE curves for three Fe-Cr-Ni-Mn alloys.

Workflow
--------
1. Generate an exact-composition random FCC bulk model.
2. Relax the cubic bulk cell at 0 K and zero pressure to obtain the
   equilibrium lattice parameter for each composition/random seed.
3. Build an orthorhombic (111) slab:
       x || [11-2], y || [-110], z || [111]
4. Relax only the z coordinates of the slab while preserving in-plane
   coordinates.
5. Calculate the generalized planar fault energy along the twinning path:
       0 <= d/b_p <= 1  : shear the upper block across the first (111) plane
       1 <  d/b_p <= 2  : retain the first full partial shift and shear the
                           block above the adjacent (111) plane
6. Export Origin-ready CSV files and a summary of gamma_usf, gamma_isf,
   gamma_utf, and gamma_esf.

Potential element order
-----------------------
The supplied setfl file contains: Fe Ni Cr Mn.
LAMMPS atom-type mapping used here:
    type 1 = Fe
    type 2 = Ni
    type 3 = Cr
    type 4 = Mn

Requirements
------------
- Python 3.9+
- numpy
- LAMMPS executable supporting pair_style eam/alloy
- matplotlib is optional and used only for automatic plotting

Examples
--------
python run_gpfe.py --lmp lmp
python run_gpfe.py --lmp "C:/LAMMPS/bin/lmp.exe"
python run_gpfe.py --lmp lmp --seeds 13579,24680,97531
python run_gpfe.py --lmp lmp --points 41 --force

Scientific note
---------------
For chemically disordered alloys, use at least three independent random
seeds for final publication-quality mean curves and standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


EV_A2_TO_MJ_M2 = 16021.76634

ELEMENT_TO_TYPE = {"Fe": 1, "Ni": 2, "Cr": 3, "Mn": 4}
TYPE_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_TYPE.items()}
MASSES = {
    1: 55.845000,
    2: 58.693400,
    3: 51.996100,
    4: 54.938044,
}

# Compositions are in at.%.
ALLOYS = {
    "Optimized": {"Fe": 35.68, "Cr": 5.00, "Ni": 35.85, "Mn": 23.47},
    "Equiatomic": {"Fe": 25.00, "Cr": 25.00, "Ni": 25.00, "Mn": 25.00},
    "LowG": {"Fe": 6.61, "Cr": 32.90, "Ni": 35.85, "Mn": 24.64},
}


@dataclass(frozen=True)
class ModelSettings:
    initial_lattice_a: float = 3.60
    bulk_n: int = 10
    slab_nx: int = 12
    slab_ny: int = 20
    slab_nz: int = 8
    vacuum_a: float = 15.0
    frozen_bottom_layers: int = 2
    points: int = 41


def format_path_for_lammps(path: Path) -> str:
    """Return an absolute path using forward slashes and quote it."""
    return '"' + path.resolve().as_posix() + '"'


def check_potential_header(path: Path) -> None:
    """Verify that the setfl element order is Fe Ni Cr Mn."""
    if not path.exists():
        raise FileNotFoundError(f"Potential file not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [f.readline().strip() for _ in range(5)]
    if len(lines) < 4:
        raise ValueError("Potential file header is incomplete.")
    tokens = lines[3].split()
    if not tokens or not tokens[0].isdigit():
        raise ValueError(
            "Could not parse setfl element header. Expected a line like "
            "'4 Fe Ni Cr Mn'."
        )
    n = int(tokens[0])
    elements = tokens[1:1+n]
    expected = ["Fe", "Ni", "Cr", "Mn"]
    if elements != expected:
        raise ValueError(
            f"Potential element order is {elements}, but this script expects {expected}. "
            "Update ELEMENT_TO_TYPE and pair_coeff mapping before running."
        )


def exact_counts(total: int, composition: Dict[str, float]) -> Dict[str, int]:
    """Convert atomic percentages into integer counts using largest remainders."""
    s = sum(composition.values())
    if not math.isclose(s, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Composition does not sum to 100 at.%: {composition}, sum={s}")
    raw = {el: total * pct / 100.0 for el, pct in composition.items()}
    base = {el: int(math.floor(v)) for el, v in raw.items()}
    remain = total - sum(base.values())
    order = sorted(raw, key=lambda el: raw[el] - base[el], reverse=True)
    for el in order[:remain]:
        base[el] += 1
    if sum(base.values()) != total:
        raise RuntimeError("Internal error while assigning exact atom counts.")
    return base


def assign_types(
    total: int,
    composition: Dict[str, float],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return shuffled exact atom types and the corresponding element counts."""
    counts = exact_counts(total, composition)
    types: List[int] = []
    for el in ("Fe", "Ni", "Cr", "Mn"):
        types.extend([ELEMENT_TO_TYPE[el]] * counts[el])
    arr = np.asarray(types, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return arr, counts


def conventional_fcc_positions(n: int, a: float) -> np.ndarray:
    """Generate an n x n x n conventional FCC supercell."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    positions = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cell = np.array([i, j, k], dtype=float)
                positions.append((cell + basis) * a)
    return np.vstack(positions)


# Six-atom orthorhombic FCC basis for:
# A = a/2 [1, 1, -2], B = a/2 [-1, 1, 0], C = a [1, 1, 1].
# The basis entries are fractional coordinates in A, B, C.
ORTHO_BASIS = np.array(
    [
        [0.0,             0.0, 0.0],
        [0.5,             0.5, 0.0],
        [1.0/3.0,         0.0, 1.0/3.0],
        [5.0/6.0,         0.5, 1.0/3.0],
        [2.0/3.0,         0.0, 2.0/3.0],
        [1.0/6.0,         0.5, 2.0/3.0],
    ],
    dtype=float,
)


def oriented_slab_positions(
    nx: int,
    ny: int,
    nz: int,
    a: float,
    vacuum: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Dict[str, float]]:
    """
    Generate a rectangular FCC slab with:
        x || [11-2], y || [-110], z || [111].

    Returns:
        positions, layer_indices, box_lengths, geometry_parameters
    """
    ux = a * math.sqrt(3.0 / 2.0)
    uy = a / math.sqrt(2.0)
    uz = a * math.sqrt(3.0)
    d111 = a / math.sqrt(3.0)

    positions: List[List[float]] = []
    layers: List[int] = []

    # Keep deterministic geometric ordering; types are randomized separately.
    for k in range(nz):
        for layer_local in range(3):
            entries = ORTHO_BASIS[
                np.isclose(ORTHO_BASIS[:, 2], layer_local / 3.0)
            ]
            for j in range(ny):
                for i in range(nx):
                    for fx, fy, fz in entries:
                        positions.append(
                            [
                                (i + fx) * ux,
                                (j + fy) * uy,
                                vacuum + (k + fz) * uz,
                            ]
                        )
                        layers.append(3 * k + layer_local)

    pos = np.asarray(positions, dtype=float)
    layer_arr = np.asarray(layers, dtype=np.int32)

    lx = nx * ux
    ly = ny * uy
    lz = 2.0 * vacuum + nz * uz

    n_layers = 3 * nz
    lower_layer = n_layers // 2 - 1
    upper_layer = lower_layer + 1
    zcut1 = vacuum + (lower_layer + 0.5) * d111
    zcut2 = vacuum + (upper_layer + 0.5) * d111
    zbottom = vacuum + (1.5) * d111
    bp = a / math.sqrt(6.0)

    geom = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "area": lx * ly,
        "d111": d111,
        "bp": bp,
        "zcut1": zcut1,
        "zcut2": zcut2,
        "zbottom": zbottom,
        "n_layers": float(n_layers),
    }
    return pos, layer_arr, (lx, ly, lz), geom


def write_lammps_data(
    path: Path,
    positions: np.ndarray,
    atom_types: np.ndarray,
    box: Tuple[float, float, float],
    comment: str,
) -> None:
    """Write a LAMMPS atomic-style data file."""
    if len(positions) != len(atom_types):
        raise ValueError("positions and atom_types have different lengths")
    lx, ly, lz = box
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{comment}\n\n")
        f.write(f"{len(positions)} atoms\n")
        f.write("4 atom types\n\n")
        f.write(f"0.0 {lx:.12f} xlo xhi\n")
        f.write(f"0.0 {ly:.12f} ylo yhi\n")
        f.write(f"0.0 {lz:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for atom_type in range(1, 5):
            f.write(
                f"{atom_type} {MASSES[atom_type]:.8f} "
                f"# {TYPE_TO_ELEMENT[atom_type]}\n"
            )
        f.write("\nAtoms # atomic\n\n")
        for atom_id, (atom_type, xyz) in enumerate(
            zip(atom_types, positions), start=1
        ):
            f.write(
                f"{atom_id} {int(atom_type)} "
                f"{xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}\n"
            )


def run_lammps(
    lmp_executable: str,
    input_path: Path,
    cwd: Path,
    force: bool,
    expected_output: Path,
) -> None:
    """Run LAMMPS and preserve stdout/stderr for troubleshooting."""
    if expected_output.exists() and not force:
        return

    cmd = [lmp_executable, "-in", input_path.name]
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"LAMMPS executable not found: {lmp_executable}\n"
            "Pass the correct executable, for example:\n"
            '  python run_gpfe.py --lmp "C:/LAMMPS/bin/lmp.exe"'
        ) from exc

    (cwd / "lammps.stdout.txt").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (cwd / "lammps.stderr.txt").write_text(
        completed.stderr or "", encoding="utf-8"
    )

    if completed.returncode != 0 or not expected_output.exists():
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        err_tail = "\n".join((completed.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"LAMMPS failed in {cwd}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {completed.returncode}\n"
            f"Last stdout lines:\n{tail}\n"
            f"Last stderr lines:\n{err_tail}\n"
            f"See {cwd / 'lammps.stdout.txt'} and "
            f"{cwd / 'lammps.stderr.txt'}."
        )


def read_tagged_value(path: Path, tag: str) -> float:
    pattern = re.compile(
        rf"^\s*{re.escape(tag)}\s+([-+0-9.eE]+)\s*$"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            return float(m.group(1))
    raise ValueError(f"Could not find tag {tag!r} in {path}")


def write_bulk_relax_input(
    path: Path,
    potential: Path,
    n: int,
) -> None:
    text = f"""\
clear
units metal
dimension 3
boundary p p p
atom_style atomic

read_data bulk_initial.data

pair_style eam/alloy
pair_coeff * * {format_path_for_lammps(potential)} Fe Ni Cr Mn

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 100
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz
thermo_modify lost error flush yes

min_style cg
min_modify line quadratic
minimize 1.0e-12 1.0e-12 5000 50000

fix RELAX all box/relax iso 0.0 vmax 0.001
minimize 1.0e-12 1.0e-12 10000 100000
unfix RELAX

run 0
variable alat equal lx/{n}
print "A_LATTICE ${{alat}}" file lattice.txt screen yes
write_data bulk_relaxed.data
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def write_slab_relax_input(
    path: Path,
    potential: Path,
    zbottom: float,
) -> None:
    text = f"""\
clear
units metal
dimension 3
boundary p p f
atom_style atomic

read_data slab_initial.data

pair_style eam/alloy
pair_coeff * * {format_path_for_lammps(potential)} Fe Ni Cr Mn

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

region BOTTOM block INF INF INF INF INF {zbottom:.12f} units box
group bottom region BOTTOM
group mobile subtract all bottom

fix FREEZE bottom setforce 0.0 0.0 0.0
fix ZRELAX mobile setforce 0.0 0.0 NULL

thermo 100
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz
thermo_modify lost error flush yes

min_style cg
min_modify line quadratic
minimize 1.0e-12 1.0e-12 10000 100000
run 0

variable e0 equal pe
variable area equal lx*ly
print "E0 ${{e0}}" file baseline.txt screen yes
print "AREA ${{area}}" append baseline.txt screen yes
write_data slab_relaxed.data
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def write_fault_input(
    path: Path,
    potential: Path,
    zbottom: float,
    zcut1: float,
    zcut2: float,
    dx1: float,
    dx2: float,
) -> None:
    text = f"""\
clear
units metal
dimension 3
boundary p p f
atom_style atomic

read_data ../slab_relaxed.data

pair_style eam/alloy
pair_coeff * * {format_path_for_lammps(potential)} Fe Ni Cr Mn

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

region BOTTOM block INF INF INF INF INF {zbottom:.12f} units box
group bottom region BOTTOM
group mobile subtract all bottom

region UPPER1 block INF INF INF INF {zcut1:.12f} INF units box
region UPPER2 block INF INF INF INF {zcut2:.12f} INF units box
group upper1 region UPPER1
group upper2 region UPPER2

fix FREEZE bottom setforce 0.0 0.0 0.0
fix ZRELAX mobile setforce 0.0 0.0 NULL

displace_atoms upper1 move {dx1:.12f} 0.0 0.0 units box
displace_atoms upper2 move {dx2:.12f} 0.0 0.0 units box

thermo 100
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz
thermo_modify lost error flush yes

min_style cg
min_modify line quadratic
minimize 1.0e-12 1.0e-12 10000 100000
run 0

variable efault equal pe
print "EFAULT ${{efault}}" file energy.txt screen yes
write_data fault_relaxed.data
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def calculate_characteristic_energies(
    dnorm: np.ndarray,
    gamma: np.ndarray,
) -> Dict[str, float]:
    """Extract grid-based characteristic energies from one full curve."""
    first = dnorm <= 1.0 + 1e-12
    second = dnorm >= 1.0 - 1e-12
    idx_isf = int(np.argmin(np.abs(dnorm - 1.0)))
    idx_esf = int(np.argmin(np.abs(dnorm - 2.0)))
    return {
        "gamma_usf_mJ_m2": float(np.max(gamma[first])),
        "gamma_isf_mJ_m2": float(gamma[idx_isf]),
        "gamma_utf_mJ_m2": float(np.max(gamma[second])),
        "gamma_esf_mJ_m2": float(gamma[idx_esf]),
        "twinning_barrier_utf_minus_isf_mJ_m2": float(
            np.max(gamma[second]) - gamma[idx_isf]
        ),
    }


def write_curve_csv(
    path: Path,
    dnorm: Sequence[float],
    displacement_a: Sequence[float],
    energy_ev: Sequence[float],
    gamma_mj_m2: Sequence[float],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Normalized displacement d/bp",
                "Applied displacement (Angstrom)",
                "Relaxed potential energy (eV)",
                "GPFE (mJ/m^2)",
            ]
        )
        for row in zip(dnorm, displacement_a, energy_ev, gamma_mj_m2):
            writer.writerow(
                [
                    f"{row[0]:.8f}",
                    f"{row[1]:.10f}",
                    f"{row[2]:.12f}",
                    f"{row[3]:.8f}",
                ]
            )


def save_plot(records: List[Dict[str, object]], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV files were generated without plots.")
        return

    # Individual seed curves.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for rec in records:
        label = f"{rec['alloy']} (seed {rec['seed']})"
        ax.plot(rec["dnorm"], rec["gamma"], marker="o", markersize=2.5, label=label)
    ax.set_xlabel(r"Normalized displacement, $d/b_p$")
    ax.set_ylabel(r"GPFE (mJ/m$^2$)")
    ax.set_xlim(0.0, 2.0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "GPFE_all_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Mean curves across seeds.
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        grouped.setdefault(str(rec["alloy"]), []).append(rec)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for alloy, recs in grouped.items():
        d = np.asarray(recs[0]["dnorm"], dtype=float)
        matrix = np.vstack([np.asarray(r["gamma"], dtype=float) for r in recs])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if len(recs) > 1 else np.zeros_like(mean)
        ax.plot(d, mean, marker="o", markersize=2.5, label=alloy)
        if len(recs) > 1:
            ax.fill_between(d, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel(r"Normalized displacement, $d/b_p$")
    ax.set_ylabel(r"GPFE (mJ/m$^2$)")
    ax.set_xlim(0.0, 2.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "GPFE_mean_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def aggregate_results(
    records: List[Dict[str, object]],
    outdir: Path,
) -> None:
    # Long-format combined data.
    combined = outdir / "GPFE_all_data.csv"
    with combined.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Alloy",
                "Seed",
                "Equilibrium lattice parameter (Angstrom)",
                "Normalized displacement d/bp",
                "GPFE (mJ/m^2)",
            ]
        )
        for rec in records:
            for d, g in zip(rec["dnorm"], rec["gamma"]):
                writer.writerow(
                    [
                        rec["alloy"],
                        rec["seed"],
                        f"{float(rec['a']):.10f}",
                        f"{float(d):.8f}",
                        f"{float(g):.8f}",
                    ]
                )

    # Characteristic values for each random seed.
    summary_rows: List[Dict[str, object]] = []
    for rec in records:
        vals = calculate_characteristic_energies(
            np.asarray(rec["dnorm"], dtype=float),
            np.asarray(rec["gamma"], dtype=float),
        )
        vals.update(
            {
                "Alloy": rec["alloy"],
                "Seed": rec["seed"],
                "Equilibrium_lattice_parameter_A": rec["a"],
            }
        )
        summary_rows.append(vals)

    summary = outdir / "GPFE_characteristic_energies_by_seed.csv"
    fields = [
        "Alloy",
        "Seed",
        "Equilibrium_lattice_parameter_A",
        "gamma_usf_mJ_m2",
        "gamma_isf_mJ_m2",
        "gamma_utf_mJ_m2",
        "gamma_esf_mJ_m2",
        "twinning_barrier_utf_minus_isf_mJ_m2",
    ]
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Mean and standard deviation by alloy.
    mean_summary = outdir / "GPFE_characteristic_energies_mean_std.csv"
    numeric_fields = fields[2:]
    with mean_summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Alloy", "Number of seeds"]
            + [f"{field}_mean" for field in numeric_fields]
            + [f"{field}_std" for field in numeric_fields]
        )
        for alloy in ALLOYS:
            rows = [r for r in summary_rows if r["Alloy"] == alloy]
            if not rows:
                continue
            means = []
            stds = []
            for field in numeric_fields:
                arr = np.asarray([float(r[field]) for r in rows], dtype=float)
                means.append(float(arr.mean()))
                stds.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
            writer.writerow([alloy, len(rows)] + means + stds)

    # Mean curve table suitable for Origin.
    mean_curve = outdir / "GPFE_mean_curves_for_Origin.csv"
    alloys_with_data = [a for a in ALLOYS if any(r["alloy"] == a for r in records)]
    d_ref = np.asarray(records[0]["dnorm"], dtype=float)
    with mean_curve.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = ["Normalized displacement d/bp"]
        for alloy in alloys_with_data:
            header.extend([f"{alloy} mean (mJ/m^2)", f"{alloy} std (mJ/m^2)"])
        writer.writerow(header)
        grouped = {
            alloy: [r for r in records if r["alloy"] == alloy]
            for alloy in alloys_with_data
        }
        for idx, d in enumerate(d_ref):
            row: List[object] = [f"{d:.8f}"]
            for alloy in alloys_with_data:
                vals = np.asarray(
                    [float(r["gamma"][idx]) for r in grouped[alloy]],
                    dtype=float,
                )
                row.extend(
                    [
                        f"{vals.mean():.8f}",
                        f"{(vals.std(ddof=1) if len(vals) > 1 else 0.0):.8f}",
                    ]
                )
            writer.writerow(row)

    save_plot(records, outdir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate full GPFE curves for three Fe-Cr-Ni-Mn compositions."
    )
    parser.add_argument(
        "--lmp",
        default="lmp",
        help='LAMMPS executable, e.g. lmp or "C:/LAMMPS/bin/lmp.exe".',
    )
    parser.add_argument(
        "--potential",
        default="../potential/FeNiCrMn_EAM_Daramola2022.eam.alloy",
        help="Path to the Daramola EAM/alloy potential file.",
    )
    parser.add_argument(
        "--workdir",
        default="gpfe_runs",
        help="Directory for all generated inputs and results.",
    )
    parser.add_argument(
        "--seeds",
        default="13579",
        help="Comma-separated random seeds, e.g. 13579,24680,97531.",
    )
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--bulk-n", type=int, default=10)
    parser.add_argument("--slab-nx", type=int, default=12)
    parser.add_argument("--slab-ny", type=int, default=20)
    parser.add_argument("--slab-nz", type=int, default=8)
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--initial-a", type=float, default=3.60)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run LAMMPS even when output files already exist.",
    )
    args = parser.parse_args()

    if args.points < 5 or args.points % 2 == 0:
        raise ValueError(
            "--points must be an odd integer >= 5 so that d/bp = 1 is included."
        )
    if min(args.bulk_n, args.slab_nx, args.slab_ny, args.slab_nz) <= 0:
        raise ValueError("All model dimensions must be positive.")

    script_dir = Path(__file__).resolve().parent
    potential = Path(args.potential)
    if not potential.is_absolute():
        potential = script_dir / potential
    check_potential_header(potential)

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = script_dir / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("--seeds must contain positive integers.")

    settings = ModelSettings(
        initial_lattice_a=args.initial_a,
        bulk_n=args.bulk_n,
        slab_nx=args.slab_nx,
        slab_ny=args.slab_ny,
        slab_nz=args.slab_nz,
        vacuum_a=args.vacuum,
        points=args.points,
    )

    records: List[Dict[str, object]] = []
    dnorm_values = np.linspace(0.0, 2.0, settings.points)

    print("Potential header verified: Fe Ni Cr Mn")
    print(f"Results directory: {workdir}")
    print(f"Random seeds: {seeds}")

    for alloy_name, composition in ALLOYS.items():
        for seed in seeds:
            tag = f"{alloy_name}_seed{seed}"
            job = workdir / tag
            job.mkdir(parents=True, exist_ok=True)
            print(f"\n=== {tag} ===")

            # 1. Bulk relaxation.
            bulk_positions = conventional_fcc_positions(
                settings.bulk_n, settings.initial_lattice_a
            )
            bulk_types, bulk_counts = assign_types(
                len(bulk_positions), composition, seed
            )
            bulk_data = job / "bulk_initial.data"
            if args.force or not bulk_data.exists():
                write_lammps_data(
                    bulk_data,
                    bulk_positions,
                    bulk_types,
                    (
                        settings.bulk_n * settings.initial_lattice_a,
                        settings.bulk_n * settings.initial_lattice_a,
                        settings.bulk_n * settings.initial_lattice_a,
                    ),
                    f"{tag} bulk FCC model; exact counts {bulk_counts}",
                )

            bulk_input = job / "in.relax_bulk"
            write_bulk_relax_input(
                bulk_input,
                potential,
                settings.bulk_n,
            )
            run_lammps(
                args.lmp,
                bulk_input,
                job,
                args.force,
                job / "lattice.txt",
            )
            lattice_a = read_tagged_value(job / "lattice.txt", "A_LATTICE")
            print(f"Equilibrium lattice parameter: {lattice_a:.8f} Angstrom")

            # 2. Oriented (111) slab.
            slab_positions, slab_layers, slab_box, geom = oriented_slab_positions(
                settings.slab_nx,
                settings.slab_ny,
                settings.slab_nz,
                lattice_a,
                settings.vacuum_a,
            )
            slab_types, slab_counts = assign_types(
                len(slab_positions), composition, seed + 1000003
            )
            slab_data = job / "slab_initial.data"
            if args.force or not slab_data.exists():
                write_lammps_data(
                    slab_data,
                    slab_positions,
                    slab_types,
                    slab_box,
                    (
                        f"{tag} oriented FCC slab: x=[11-2], y=[-110], z=[111]; "
                        f"exact counts {slab_counts}"
                    ),
                )

            metadata = {
                "alloy": alloy_name,
                "composition_at_percent": composition,
                "seed": seed,
                "type_mapping": {
                    "1": "Fe",
                    "2": "Ni",
                    "3": "Cr",
                    "4": "Mn",
                },
                "equilibrium_lattice_parameter_A": lattice_a,
                "model": {
                    "bulk_n": settings.bulk_n,
                    "slab_nx": settings.slab_nx,
                    "slab_ny": settings.slab_ny,
                    "slab_nz": settings.slab_nz,
                    "number_of_slab_atoms": int(len(slab_positions)),
                    "vacuum_A_each_side": settings.vacuum_a,
                    "x_direction": "[11-2]",
                    "y_direction": "[-110]",
                    "z_direction": "[111]",
                    "fault_plane": "(111)",
                    "partial_direction": "[11-2]",
                    "bp_A": geom["bp"],
                    "fault_area_A2": geom["area"],
                    "zcut_first_fault_A": geom["zcut1"],
                    "zcut_adjacent_fault_A": geom["zcut2"],
                },
            }
            import json
            (job / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            slab_input = job / "in.relax_slab"
            write_slab_relax_input(
                slab_input,
                potential,
                geom["zbottom"],
            )
            run_lammps(
                args.lmp,
                slab_input,
                job,
                args.force,
                job / "baseline.txt",
            )
            e0 = read_tagged_value(job / "baseline.txt", "E0")
            area = read_tagged_value(job / "baseline.txt", "AREA")
            if not math.isclose(area, geom["area"], rel_tol=1e-7, abs_tol=1e-7):
                print(
                    "Warning: LAMMPS area differs slightly from generated area: "
                    f"{area} vs {geom['area']}"
                )

            energies: List[float] = []
            applied_displacements: List[float] = []

            # 3. Independent point calculations along full twinning path.
            for idx, dnorm in enumerate(dnorm_values):
                if dnorm <= 1.0 + 1e-12:
                    dx1 = float(dnorm * geom["bp"])
                    dx2 = 0.0
                else:
                    dx1 = float(geom["bp"])
                    dx2 = float((dnorm - 1.0) * geom["bp"])

                point_dir = job / f"point_{idx:03d}_d_{dnorm:.4f}"
                point_dir.mkdir(parents=True, exist_ok=True)
                fault_input = point_dir / "in.gpfe_point"
                write_fault_input(
                    fault_input,
                    potential,
                    geom["zbottom"],
                    geom["zcut1"],
                    geom["zcut2"],
                    dx1,
                    dx2,
                )
                run_lammps(
                    args.lmp,
                    fault_input,
                    point_dir,
                    args.force,
                    point_dir / "energy.txt",
                )
                energy = read_tagged_value(point_dir / "energy.txt", "EFAULT")
                energies.append(energy)
                applied_displacements.append(dx1 + dx2)
                print(
                    f"  point {idx+1:02d}/{settings.points}: "
                    f"d/bp={dnorm:.4f}, E={energy:.8f} eV"
                )

            energy_arr = np.asarray(energies, dtype=float)
            gamma = (energy_arr - e0) / area * EV_A2_TO_MJ_M2

            curve_csv = job / f"GPFE_{tag}.csv"
            write_curve_csv(
                curve_csv,
                dnorm_values,
                applied_displacements,
                energies,
                gamma,
            )

            characteristic = calculate_characteristic_energies(
                dnorm_values, gamma
            )
            with (job / f"GPFE_characteristic_{tag}.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["Quantity", "Value (mJ/m^2)"])
                for key, value in characteristic.items():
                    writer.writerow([key, f"{value:.8f}"])

            records.append(
                {
                    "alloy": alloy_name,
                    "seed": seed,
                    "a": lattice_a,
                    "dnorm": dnorm_values.copy(),
                    "gamma": gamma.copy(),
                }
            )

    aggregate_results(records, workdir)
    print("\nAll calculations completed.")
    print(f"Origin-ready file: {workdir / 'GPFE_mean_curves_for_Origin.csv'}")
    print(
        f"Characteristic energies: "
        f"{workdir / 'GPFE_characteristic_energies_mean_std.csv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
