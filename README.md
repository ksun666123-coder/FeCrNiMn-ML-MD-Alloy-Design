# FeCrNiMn-ML-MD-Alloy-Design

Source code for machine-learning-guided inverse design and molecular-dynamics analysis of non-equiatomic Fe-Cr-Ni-Mn alloys.

## Scope of this repository

This repository contains the source code used for machine-learning-guided alloy design, molecular-dynamics calculations, and post-processing analyses. The Fe-Ni-Cr-Mn EAM/alloy potential used by the LAMMPS workflows is also included.

Training datasets, MD result files, stress-strain data, SHAP data, LAMMPS trajectories, figures, and other numerical outputs are not included.

## Repository structure

```text
01_Machine_Learning/
  BP_GA_inverse_design/
    BP_GA_inverse_design.m

02_Molecular_Dynamics/
  potential/
    FeNiCrMn_EAM_Daramola2022.eam.alloy
    README.md

  shear_modulus/
    calc_elastic.in.lmp
    run_shear_modulus_batch.py

  SFE/
    calc_sfe.in.lmp
    run_sfe_batch.py

  GPFE/
    run_gpfe.py

  tensile/
    in.tensile_Equiatomic.lmp
    in.tensile_LowG.lmp
    in.tensile_TopG.lmp

03_Postprocessing/
  planar_defect_analysis/
    analyze_planar_defects.py

  dislocation_analysis/
    extract_ptm_dxa_curves.py

  SHAP_plotting/
    plot_shap_summary.py

CODE_MANIFEST.md
requirements.txt
.gitignore
```

## 1. BP-GA inverse design

`01_Machine_Learning/BP_GA_inverse_design/BP_GA_inverse_design.m` provides the MATLAB workflow for:

- comparing different composition- and physics-based feature sets;
- training BP neural-network models for shear modulus `G`;
- evaluating model performance using R², RMSE, and MAE;
- constructing an ensemble BP model;
- performing constrained genetic-algorithm optimization;
- applying model-disagreement and applicability-domain penalties;
- exporting optimized candidate compositions and GA convergence information.

The script prompts the user to select the modeling dataset.

Required input columns are:

```text
Fe, Cr, Ni, Mn, G_GPa
```

`SFE_mJm2` may also be included as an optional column.

The numerical training dataset used in the manuscript is not distributed in this repository.

MATLAB functionality used by this workflow includes `fitnet`, `trainbr`, and `mapminmax`.

## 2. Molecular dynamics calculations

LAMMPS with `pair_style eam/alloy` support is required.

The interatomic potential is provided at:

```text
02_Molecular_Dynamics/potential/FeNiCrMn_EAM_Daramola2022.eam.alloy
```

### 2.1 Shear modulus

The shear-modulus workflow consists of:

```text
02_Molecular_Dynamics/shear_modulus/calc_elastic.in.lmp
02_Molecular_Dynamics/shear_modulus/run_shear_modulus_batch.py
```

Example:

```bash
python 02_Molecular_Dynamics/shear_modulus/run_shear_modulus_batch.py \
    --input YOUR_COMPOSITIONS.csv \
    --lmp lmp
```

The input CSV must contain:

```text
Fe, Cr, Ni, Mn
```

### 2.2 Stacking-fault energy

The SFE workflow consists of:

```text
02_Molecular_Dynamics/SFE/calc_sfe.in.lmp
02_Molecular_Dynamics/SFE/run_sfe_batch.py
```

Example:

```bash
python 02_Molecular_Dynamics/SFE/run_sfe_batch.py \
    --input YOUR_COMPOSITIONS.csv \
    --lmp lmp
```

### 2.3 GPFE/GSFE calculations

`02_Molecular_Dynamics/GPFE/run_gpfe.py` calculates generalized planar fault-energy curves for the three representative Fe-Cr-Ni-Mn alloys used in the comparative analysis.

The script includes:

- the optimized high-G alloy;
- the equiatomic alloy;
- the low-G reference alloy.

Example:

```bash
cd 02_Molecular_Dynamics/GPFE
python run_gpfe.py --lmp lmp --seeds 13579,24680,97531
```

The workflow can output the GPFE/GSFE curves and characteristic energies including `gamma_usf`, `gamma_isf`, `gamma_utf`, and `gamma_esf`.

### 2.4 Tensile simulations

Three LAMMPS tensile input files are provided:

```text
in.tensile_Equiatomic.lmp
in.tensile_LowG.lmp
in.tensile_TopG.lmp
```

They correspond to:

| Input file | Alloy |
|---|---|
| `in.tensile_Equiatomic.lmp` | Equiatomic Fe-Cr-Ni-Mn alloy |
| `in.tensile_LowG.lmp` | Low-G reference alloy |
| `in.tensile_TopG.lmp` | Optimized high-G alloy (Fe35.68Cr5.00Ni35.85Mn23.47, at.%) |

Example:

```bash
cd 02_Molecular_Dynamics/tensile

lmp -in in.tensile_Equiatomic.lmp
lmp -in in.tensile_LowG.lmp
lmp -in in.tensile_TopG.lmp
```

Stress-strain files and atomic trajectories generated during the simulations are not included in the repository.

## 3. OVITO post-processing

### 3.1 Planar-defect analysis

`03_Postprocessing/planar_defect_analysis/analyze_planar_defects.py` performs CNA-based planar-defect classification.

Example:

```bash
ovitos 03_Postprocessing/planar_defect_analysis/analyze_planar_defects.py \
    YOUR_DUMP.lammpstrj --step 2
```

The output categories are:

```text
Other, FCC, ISF, HCP, ESF, TB, BCC
```

### 3.2 PTM/DXA dislocation analysis

`03_Postprocessing/dislocation_analysis/extract_ptm_dxa_curves.py` extracts structural fractions and dislocation-density evolution during tensile deformation.

Example:

```bash
ovitos 03_Postprocessing/dislocation_analysis/extract_ptm_dxa_curves.py \
    YOUR_DUMP.lammpstrj --step 2
```

The DXA output includes:

```text
Perfect
Shockley partial
Stair-rod
Hirth
Frank partial
Other
Total
```

## 4. SHAP plotting

`03_Postprocessing/SHAP_plotting/plot_shap_summary.py` generates a SHAP summary beeswarm plot from a precomputed long-format SHAP CSV.

Required columns are:

```text
Feature
Feature_rank
Y_for_Origin
SHAP_value
Feature_value_normalized
```

Example:

```bash
python 03_Postprocessing/SHAP_plotting/plot_shap_summary.py \
    YOUR_SHAP_SUMMARY.csv
```

The SHAP input data used for the manuscript are not included in this repository.

## Python dependencies

Install the standard Python dependencies with:

```bash
pip install -r requirements.txt
```

OVITO-based scripts should be run using `ovitos` or another Python environment containing the OVITO Python module.

## Notes

- Numerical training datasets and simulation results are not included.
- The Fe-Ni-Cr-Mn EAM/alloy potential required by the LAMMPS workflows is included.
- Large LAMMPS output files, trajectories, logs, CSV/DAT results, and generated images are excluded by `.gitignore`.
- File and folder names are case-sensitive on some operating systems.
