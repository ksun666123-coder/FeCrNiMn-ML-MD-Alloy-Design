# -*- coding: utf-8 -*-
"""
extract_ptm_dxa_curves.py

功能：
1. 从 LAMMPS dump/lammpstrj 文件中读取每一帧；
2. 用 OVITO 的 PTM 统计 FCC/HCP/BCC/ICO/Other 原子百分比随应变变化；
3. 用 OVITO 的 DXA 统计不同类型位错密度随应变变化；
4. 输出 CSV 和 PNG 曲线图。

运行方式（在 OVITO 自带 Python 或 ovitos 中运行）：
    ovitos extract_ptm_dxa_curves.py dump_tensile_xxx.lammpstrj

如果帧数很多、计算较慢，可以每隔 2 或 5 帧取一帧：
    ovitos extract_ptm_dxa_curves.py dump_tensile_xxx.lammpstrj --step 2
    ovitos extract_ptm_dxa_curves.py dump_tensile_xxx.lammpstrj --step 5
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier, DislocationAnalysisModifier


def get_cell_vectors(data):
    """Return 3x3 cell-vector matrix robustly for OVITO SimulationCell."""
    cell = data.cell
    if hasattr(cell, "matrix"):
        mat = np.array(cell.matrix)
    else:
        mat = np.array(cell)
    # OVITO cell matrix is usually 3x4: three cell vectors + origin
    if mat.shape[0] == 3 and mat.shape[1] >= 3:
        return mat[:, :3]
    if mat.shape[0] >= 3 and mat.shape[1] == 3:
        return mat[:3, :]
    raise RuntimeError("Cannot parse simulation cell matrix shape: {}".format(mat.shape))


def get_volume_and_lx(data):
    vecs = get_cell_vectors(data)
    volume = abs(np.linalg.det(vecs))
    # For orthogonal tensile cell, x-length is usually the norm of the first cell vector.
    lx = np.linalg.norm(vecs[0])
    return volume, lx


def get_ptm_counts(data):
    """Get PTM structure counts. Return count dictionary."""
    total = data.particles.count
    counts = {"FCC": 0, "HCP": 0, "BCC": 0, "ICO": 0, "OTHER": 0}

    # Priority 1: use OVITO generated attributes if available
    attr_map = {
        "FCC": ["PolyhedralTemplateMatching.counts.FCC", "PolyhedralTemplateMatching.counts.fcc"],
        "HCP": ["PolyhedralTemplateMatching.counts.HCP", "PolyhedralTemplateMatching.counts.hcp"],
        "BCC": ["PolyhedralTemplateMatching.counts.BCC", "PolyhedralTemplateMatching.counts.bcc"],
        "ICO": ["PolyhedralTemplateMatching.counts.ICO", "PolyhedralTemplateMatching.counts.ico"],
        "OTHER": ["PolyhedralTemplateMatching.counts.OTHER", "PolyhedralTemplateMatching.counts.Other", "PolyhedralTemplateMatching.counts.other"],
    }
    got_attrs = False
    for name, keys in attr_map.items():
        for k in keys:
            if k in data.attributes:
                counts[name] = int(data.attributes[k])
                got_attrs = True
                break

    if got_attrs:
        # Some OVITO versions do not provide OTHER count as an attribute.
        known = counts["FCC"] + counts["HCP"] + counts["BCC"] + counts["ICO"]
        if counts["OTHER"] == 0 and known <= total:
            counts["OTHER"] = int(total - known)
        return counts

    # Priority 2: use particle property "Structure Type"
    st = np.asarray(data.particles["Structure Type"])
    # OVITO PTM type ids are usually:
    # 0 OTHER, 1 FCC, 2 HCP, 3 BCC, 4 ICO
    counts["OTHER"] = int(np.count_nonzero(st == 0))
    counts["FCC"] = int(np.count_nonzero(st == 1))
    counts["HCP"] = int(np.count_nonzero(st == 2))
    counts["BCC"] = int(np.count_nonzero(st == 3))
    counts["ICO"] = int(np.count_nonzero(st == 4))
    return counts


def classify_dxa_attr_key(key):
    """Classify dislocation length attribute key from OVITO DXA."""
    low = key.lower()
    if "total" in low:
        return None
    if "perfect" in low or "1/2<110>" in low or "1/2 <110>" in low:
        return "Perfect"
    if "shockley" in low or "1/6<112>" in low or "1/6 <112>" in low:
        return "Shockley_partial"
    if "stair" in low or "1/6<110>" in low or "1/6 <110>" in low:
        return "Stair_rod"
    if "hirth" in low or "1/3<100>" in low or "1/3 <100>" in low:
        return "Hirth"
    if "frank" in low or "1/3<111>" in low or "1/3 <111>" in low:
        return "Frank_partial"
    if "other" in low:
        return "Other"
    return "Other"


def get_dxa_lengths(data):
    """
    Get dislocation line lengths in Angstrom from DXA attributes.
    Returns a dictionary of line lengths.
    """
    lengths = {
        "Perfect": 0.0,
        "Shockley_partial": 0.0,
        "Stair_rod": 0.0,
        "Hirth": 0.0,
        "Frank_partial": 0.0,
        "Other": 0.0,
        "Total": 0.0,
    }

    # Total line length
    total_keys = [
        "DislocationAnalysis.total_line_length",
        "DislocationAnalysis.total_line_length.1",
    ]
    for k in total_keys:
        if k in data.attributes:
            lengths["Total"] = float(data.attributes[k])
            break

    # Per-type line length attributes
    found_per_type = False
    for k, v in data.attributes.items():
        if not k.startswith("DislocationAnalysis"):
            continue
        if "length" not in k.lower():
            continue
        if "total" in k.lower():
            continue
        dtype = classify_dxa_attr_key(k)
        if dtype is not None:
            try:
                lengths[dtype] += float(v)
                found_per_type = True
            except Exception:
                pass

    # If per-type lengths are not provided, try to sum segment lengths as Total.
    if lengths["Total"] == 0.0:
        try:
            total = 0.0
            for seg in data.dislocations.segments:
                total += float(seg.length)
            lengths["Total"] = total
        except Exception:
            pass

    # If per-type not available but total exists, put total into Other as a fallback.
    if not found_per_type and lengths["Total"] > 0:
        lengths["Other"] = lengths["Total"]

    # If total missing but per-type exists, sum them.
    if lengths["Total"] == 0.0:
        lengths["Total"] = sum(lengths[k] for k in ["Perfect", "Shockley_partial", "Stair_rod", "Hirth", "Frank_partial", "Other"])

    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_file", help="LAMMPS dump/lammpstrj file")
    parser.add_argument("--step", type=int, default=1, help="Analyze every Nth frame, default=1")
    parser.add_argument("--outdir", default="ptm_dxa_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pipeline = import_file(args.dump_file, multiple_frames=True)

    # PTM: structural classification
    ptm = PolyhedralTemplateMatchingModifier()
    # Keep all structures enabled by default
    pipeline.modifiers.append(ptm)

    # DXA: dislocation analysis
    dxa = DislocationAnalysisModifier()
    # For FCC alloy
    dxa.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
    pipeline.modifiers.append(dxa)

    nframes = pipeline.source.num_frames
    print("Total frames: {}; analyzing every {} frame(s).".format(nframes, args.step))

    ptm_rows = []
    dxa_rows = []
    lx0 = None

    frames = list(range(0, nframes, args.step))
    for count, frame in enumerate(frames, 1):
        # 修复了这里的 f-string
        print("Analyzing frame {}/{} ...".format(frame+1, nframes))
        data = pipeline.compute(frame)
        volume, lx = get_volume_and_lx(data)

        if lx0 is None:
            lx0 = lx
        strain = (lx - lx0) / lx0
        strain_percent = strain * 100.0

        # PTM fractions
        counts = get_ptm_counts(data)
        total_atoms = data.particles.count
        ptm_rows.append({
            "frame": frame,
            "strain": strain,
            "strain_percent": strain_percent,
            "total_atoms": total_atoms,
            "FCC_percent": counts["FCC"] / total_atoms * 100.0,
            "HCP_percent": counts["HCP"] / total_atoms * 100.0,
            "BCC_percent": counts["BCC"] / total_atoms * 100.0,
            "ICO_percent": counts["ICO"] / total_atoms * 100.0,
            "Other_percent": counts["OTHER"] / total_atoms * 100.0,
        })

        # DXA densities
        lengths = get_dxa_lengths(data)
        # If length is in Å and volume is in Å^3:
        # density (m^-2) = L/V * 1e20
        # density in 1e18 m^-2 = L/V * 100
        factor = 100.0 / volume if volume != 0 else np.nan
        dxa_rows.append({
            "frame": frame,
            "strain": strain,
            "strain_percent": strain_percent,
            "volume_A3": volume,
            "Perfect_density_1e18_m2": lengths["Perfect"] * factor,
            "Shockley_partial_density_1e18_m2": lengths["Shockley_partial"] * factor,
            "Stair_rod_density_1e18_m2": lengths["Stair_rod"] * factor,
            "Hirth_density_1e18_m2": lengths["Hirth"] * factor,
            "Frank_partial_density_1e18_m2": lengths["Frank_partial"] * factor,
            "Other_density_1e18_m2": lengths["Other"] * factor,
            "Total_density_1e18_m2": lengths["Total"] * factor,
            "Total_line_length_A": lengths["Total"],
        })

    ptm_df = pd.DataFrame(ptm_rows)
    dxa_df = pd.DataFrame(dxa_rows)

    ptm_csv = os.path.join(args.outdir, "ptm_structure_fraction_vs_strain.csv")
    dxa_csv = os.path.join(args.outdir, "dxa_dislocation_density_vs_strain.csv")
    ptm_df.to_csv(ptm_csv, index=False, encoding="utf-8-sig")
    dxa_df.to_csv(dxa_csv, index=False, encoding="utf-8-sig")

    # Plot PTM fractions
    plt.figure(figsize=(7, 5))
    for col, label in [
        ("BCC_percent", "BCC"),
        ("FCC_percent", "FCC"),
        ("HCP_percent", "HCP"),
        ("ICO_percent", "ICO"),
        ("Other_percent", "OTHER"),
    ]:
        plt.plot(ptm_df["strain_percent"], ptm_df[col], linewidth=1.8, label=label)
    plt.xlabel("Strain (%)")
    plt.ylabel("Fraction (%)")
    plt.legend(frameon=False)
    plt.tight_layout()
    ptm_png = os.path.join(args.outdir, "ptm_structure_fraction_vs_strain.png")
    plt.savefig(ptm_png, dpi=300)
    plt.close()

    # Plot total dislocation density
    plt.figure(figsize=(7, 5))
    plt.plot(dxa_df["strain_percent"], dxa_df["Total_density_1e18_m2"], linewidth=1.8, label="Total")
    plt.xlabel("Strain (%)")
    plt.ylabel(r"Total dislocation density ($\times 10^{18}$ m$^{-2}$)")
    plt.legend(frameon=False)
    plt.tight_layout()
    total_png = os.path.join(args.outdir, "dxa_total_dislocation_density_vs_strain.png")
    plt.savefig(total_png, dpi=300)
    plt.close()

    # Plot different dislocation types
    plt.figure(figsize=(7, 5))
    for col, label in [
        ("Perfect_density_1e18_m2", "Perfect"),
        ("Shockley_partial_density_1e18_m2", "Shockley partial"),
        ("Stair_rod_density_1e18_m2", "Stair-rod"),
        ("Hirth_density_1e18_m2", "Hirth"),
        ("Frank_partial_density_1e18_m2", "Frank partial"),
        ("Other_density_1e18_m2", "Other"),
    ]:
        if col in dxa_df.columns and dxa_df[col].max() > 0:
            plt.plot(dxa_df["strain_percent"], dxa_df[col], linewidth=1.8, label=label)
    plt.xlabel("Strain (%)")
    plt.ylabel(r"Dislocation density ($\times 10^{18}$ m$^{-2}$)")
    plt.legend(frameon=False)
    plt.tight_layout()
    dxa_png = os.path.join(args.outdir, "dxa_dislocation_types_vs_strain.png")
    plt.savefig(dxa_png, dpi=300)
    plt.close()

    print("Done.")
    print("Output files:")
    print(ptm_csv)
    print(dxa_csv)
    print(ptm_png)
    print(total_png)
    print(dxa_png)


if __name__ == "__main__":
    main()
