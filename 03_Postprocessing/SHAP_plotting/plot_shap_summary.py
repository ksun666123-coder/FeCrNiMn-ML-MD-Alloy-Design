#!/usr/bin/env python3
"""Plot a SHAP summary beeswarm figure from a long-format CSV.

Required CSV columns:
Feature, Feature_rank, Y_for_Origin, SHAP_value, Feature_value_normalized
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LABEL_MAP = {
    "delta": r"$\delta$",
    "Cr": "Cr",
    "Fe": "Fe",
    "Ni": "Ni",
    "Smix": r"$S_{\mathrm{mix}}$",
    "Hmix": r"$H_{\mathrm{mix}}$",
    "VEC": "VEC",
    "Mn": "Mn",
    "Omega": r"$\Omega$",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", help="Long-format SHAP summary CSV")
    parser.add_argument("--output", default="SHAP_summary_beeswarm.png")
    args = parser.parse_args()

    records = []
    with open(args.summary_csv, newline="", encoding="utf-8-sig") as f:
        records.extend(csv.DictReader(f))
    if not records:
        raise ValueError("No rows found in SHAP CSV")

    required = {"Feature", "Feature_rank", "Y_for_Origin", "SHAP_value", "Feature_value_normalized"}
    missing = required.difference(records[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Arial", "Times New Roman", "Liberation Serif", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False

    feature_rank = {r["Feature"]: int(float(r["Feature_rank"])) for r in records}
    ordered_features = [k for k, _ in sorted(feature_rank.items(), key=lambda item: item[1])]
    display_features = [LABEL_MAP.get(f, f) for f in ordered_features]

    x = np.array([float(r["SHAP_value"]) for r in records])
    y = np.array([float(r["Y_for_Origin"]) for r in records])
    c = np.array([float(r["Feature_value_normalized"]) for r in records])

    cmap = LinearSegmentedColormap.from_list("blue_purple_pink", ["#1395ff", "#7b3fcf", "#ff0d57"])
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=220)
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=24, edgecolors="none")
    ax.axvline(0, color="gray", linewidth=2.0, alpha=0.8, zorder=0)
    for i in range(1, len(ordered_features) + 1):
        ax.axhline(i, color="#d9d9d9", linestyle=(0, (1, 3)), linewidth=1, zorder=0)

    ranks = np.arange(1, len(ordered_features) + 1)
    ax.set_yticks(ranks)
    ax.set_yticklabels(display_features, fontsize=18)
    ax.set_ylim(len(ordered_features) + 0.7, 0.3)
    ax.tick_params(axis="x", labelsize=15, width=1.8, length=6)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=20)
    ax.set_title("SHAP Summary Plot for G", fontsize=20, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)

    cbar = fig.colorbar(sc, ax=ax, pad=0.04, fraction=0.045, aspect=60)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=16, length=0)
    cbar.set_label("Feature value", fontsize=18, rotation=90, labelpad=18)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
