# Tuner

**Tuner** is an evidence-aware visual analytics system for comparing dynamic protein-tunnel ensembles across molecular-dynamics datasets. It is designed to help researchers investigate how mutations, ligands, or conformational states reshape tunnel geometry and the surrounding residue environment.

Directly superimposing multiple tunnel ensembles in three dimensions often produces severe visual clutter, while independently assigned cluster identifiers make cross-dataset comparison ambiguous. Tuner addresses these problems through a coordinated, revisable workflow. It aligns protein structures, proposes geometrically plausible tunnel-cluster correspondences, maps local residue environments onto normalized physical arc length, and preserves the selected correspondence as an inspectable analytical state.

The interface integrates four linked components: **Dataset Compare** for cross-dataset tunnel correspondence, the **Full-path Residue Replacement Map** for localizing residue-environment changes, the **Residue Observer** for side-by-side structural inspection, and an **Evidence Workspace** for examining geometric and ensemble-level support. Together, these views help analysts distinguish visually salient differences from interpretations that remain stable across controls or independent trajectories.

Tuner operates on precomputed tunnel ensembles and aligned PDB snapshots. Example datasets and accompanying metadata are distributed separately through the project’s Zenodo records.

## Installation

Tuner currently targets Windows 10/11 and Python 3.10 or 3.11. Clone the repository using the internal package directory name, create an isolated environment, and install the dependencies:

```powershell
git clone https://github.com/TowlGol/Tuner.git TopoTunnel_UI
cd <directory-containing-TopoTunnel_UI>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r TopoTunnel_UI\requirements.txt
```

Launch the application from the directory containing `TopoTunnel_UI`:

```powershell
python -m TopoTunnel_UI
```

An isolated virtual environment is recommended because Qt, VTK, and system-wide Conda installations may contain conflicting binary libraries.

## Data setup and use

Each dataset folder must contain a precomputed tunnel file named `preprocessed_paths.pkl` or a supported `.db` file. A `representative_frame.pdb` file is recommended for structural display. To enable frame-level inspection in the Residue Observer, add an `MD_path.txt` file to the dataset folder; its first non-empty line should contain the absolute path to the directory of aligned PDB frames, for example:

```text
D:\data\CYP2J2\WT\aligned_frames
```

After starting Tuner:

1. Open **Datasets** and select **Add Dataset** to load one or more dataset folders.
2. In **Dataset Compare**, choose a source dataset, a target dataset, and the source tunnel cluster to inspect the proposed cross-dataset correspondence.
3. Use the **Full-path Residue Replacement Map** to localize changes along normalized tunnel arc length.
4. Select a highlighted region to compare its residue environment in the **Residue Observer**.
5. Switch target datasets or inspect the **Evidence Workspace** to assess whether an interpretation persists across controls or independent trajectories.

The repository contains the visual analytics application; molecular-dynamics trajectories and precomputed tunnel ensembles are distributed separately.

## License

Copyright (c) 2026 Yapeng Liu. All rights reserved. This project is provided for local, non-commercial research evaluation only. Republishing, reproducing, mirroring, redistributing, sublicensing, selling, or incorporating any part of the code or documentation into another distributed work is prohibited without prior written permission. See [LICENSE](https://github.com/TowlGol/Tuner/blob/main/LICENSE) for the complete terms.
