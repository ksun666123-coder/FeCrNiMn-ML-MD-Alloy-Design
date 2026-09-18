#!/usr/bin/env python3
"""Batch calculation of elastic constants and VRH shear modulus with LAMMPS.

Input CSV must contain columns: Fe, Cr, Ni, Mn.
Compositions may be fractions; each row is normalized before use.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POTENTIAL = BASE_DIR.parent / "potential" / "FeNiCrMn_EAM_Daramola2022.eam.alloy"
DEFAULT_LAMMPS_IN = BASE_DIR / "calc_elastic.in.lmp"
RESULT_RE = re.compile(r"^RESULT,(.+)$", re.MULTILINE)


def find_executable(user_value: str | None, candidates: tuple[str, ...]) -> str:
    if user_value:
        p = Path(user_value).expanduser()
        if p.is_file():
            return str(p.resolve())
        found = shutil.which(user_value)
        if found:
            return found
        raise FileNotFoundError(f"Executable not found: {user_value}")
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(f"Could not find any of: {', '.join(candidates)}")


def normalize(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    total = sum(values)
    if not (math.isfinite(total) and total > 0):
        raise ValueError("Invalid composition sum")
    return tuple(v / total for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with Fe, Cr, Ni, Mn columns")
    parser.add_argument("--lmp", default=os.environ.get("LAMMPS_EXE"), help="LAMMPS executable")
    parser.add_argument("--potential", default=str(DEFAULT_POTENTIAL))
    parser.add_argument("--lammps-input", default=str(DEFAULT_LAMMPS_IN))
    parser.add_argument("--output", default=str(BASE_DIR / "results_G.csv"))
    parser.add_argument("--runs-dir", default=str(BASE_DIR / "runs"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC"))
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    potential = Path(args.potential).expanduser().resolve()
    lammps_in = Path(args.lammps_input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    for p, label in [(input_csv, "input CSV"), (potential, "potential"), (lammps_in, "LAMMPS input")]:
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    lmp = find_executable(args.lmp, ("lmp", "lmp.exe", "lmp_serial", "lmp_mpi"))
    mpiexec = find_executable(args.mpiexec, ("mpiexec", "mpirun")) if args.mpi else None

    df = pd.read_csv(input_csv)
    required = {"Fe", "Cr", "Ni", "Mn"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain {sorted(required)}")

    runs_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Fe","Cr","Ni","Mn","C11_GPa","C12_GPa","C44_GPa","G_GPa","status","run_dir"])

    for idx, row in df.iterrows():
        fe, cr, ni, mn = normalize(tuple(float(row[c]) for c in ("Fe","Cr","Ni","Mn")))
        seed = args.seed + idx * 97
        run_dir = runs_dir / f"{idx+1:04d}_Fe{fe:.4f}_Cr{cr:.4f}_Ni{ni:.4f}_Mn{mn:.4f}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = ([str(mpiexec), "-n", str(args.nproc), lmp] if args.mpi else [lmp]) + [
            "-in", str(lammps_in),
            "-var", "xFe", f"{fe:.12f}",
            "-var", "xCr", f"{cr:.12f}",
            "-var", "xNi", f"{ni:.12f}",
            "-var", "xMn", f"{mn:.12f}",
            "-var", "seed", str(seed),
            "-var", "potfile", str(potential),
        ]
        proc = subprocess.run(cmd, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", shell=False)
        stdout = proc.stdout or ""
        (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
        record = [fe, cr, ni, mn, "", "", "", "", "", run_dir.as_posix()]
        if proc.returncode != 0:
            record[8] = f"FAIL_RC{proc.returncode}"
        else:
            matches = RESULT_RE.findall(stdout)
            if not matches:
                record[8] = "FAIL_NO_RESULT_LINE"
            else:
                fields = matches[-1].strip().split(",")
                if len(fields) == 8:
                    record = fields + ["OK", run_dir.as_posix()]
                else:
                    record[8] = "FAIL_BAD_RESULT_FORMAT"
        with output.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(record)
        print(f"[{idx+1}/{len(df)}] {record[8]}")

    print(f"Results written to: {output}")


if __name__ == "__main__":
    main()
