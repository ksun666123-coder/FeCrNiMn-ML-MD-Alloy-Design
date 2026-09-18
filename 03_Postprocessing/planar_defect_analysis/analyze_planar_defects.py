#!/usr/bin/env python3
"""Planar-defect analysis using OVITO CNA and local-neighbor classification.

Usage:
    ovitos analyze_planar_defects.py dump_tensile_xxx.lammpstrj
    ovitos analyze_planar_defects.py dump_tensile_xxx.lammpstrj --step 2 --output defect_statistics.txt
"""
from __future__ import annotations

import argparse
import numpy as np
from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier
from ovito.data import NearestNeighborFinder


def defect_analysis_modifier(frame, data):
    if "Structure Type" not in data.particles:
        raise ValueError("'Structure Type' is missing; apply CNA before this modifier.")

    structure_types = data.particles["Structure Type"][...]
    num_particles = data.particles.count
    pda = np.zeros(num_particles, dtype=int)
    finder = NearestNeighborFinder(12, data)

    def esf_twin(k, neighbors):
        for index in neighbors:
            if structure_types[index] == 1:
                fcc = hcp = 0
                for neigh in finder.find(index):
                    if structure_types[neigh.index] == 1:
                        fcc += 1
                    elif structure_types[neigh.index] == 2:
                        hcp += 1
                if (5 <= fcc <= 6) and (5 <= hcp <= 6):
                    pda[k] = 4
                    break
        if pda[k] != 4:
            pda[k] = 5

    def classify_hcp(index):
        fcc = hcp = 0
        hcp_neighbors = []
        neighbors = []
        for neigh in finder.find(index):
            neighbors.append(neigh.index)
            if structure_types[neigh.index] == 1:
                fcc += 1
            elif structure_types[neigh.index] == 2:
                hcp += 1
                hcp_neighbors.append(neigh.index)

        if hcp >= 11:
            pda[index] = 3
            for i in hcp_neighbors:
                pda[i] = 3
        elif (2 <= fcc <= 3) and (8 <= hcp <= 9):
            if pda[index] != 3:
                pda[index] = 2
        elif (5 <= fcc <= 6) and (5 <= hcp <= 6):
            esf_twin(index, neighbors)

    for index in range(num_particles):
        st = structure_types[index]
        if st == 2:
            classify_hcp(index)
        elif st == 1:
            pda[index] = 1
        elif st == 3:
            pda[index] = 6
        else:
            pda[index] = 0

    data.particles_.create_property("planar defect", data=pda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_file", help="LAMMPS dump/lammpstrj file")
    parser.add_argument("--step", type=int, default=2, help="Analyze every Nth frame")
    parser.add_argument("--output", default="defect_statistics.txt")
    args = parser.parse_args()

    pipeline = import_file(args.dump_file, multiple_frames=True)
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())
    pipeline.modifiers.append(defect_analysis_modifier)

    total_frames = pipeline.source.num_frames
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("Frame\tOther(0)\tFCC(1)\tISF(2)\tHCP(3)\tESF(4)\tTB(5)\tBCC(6)\n")
        for frame in range(0, total_frames, args.step):
            print(f"Analyzing frame {frame}/{total_frames - 1}")
            data = pipeline.compute(frame)
            values = data.particles["planar defect"][...]
            counts = np.bincount(values, minlength=7)
            f.write(
                f"{frame}\t{counts[0]}\t{counts[1]}\t{counts[2]}\t{counts[3]}\t"
                f"{counts[4]}\t{counts[5]}\t{counts[6]}\n"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
