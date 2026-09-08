# Tuner

**Tuner** is an evidence-aware visual analytics system for comparing dynamic protein-tunnel ensembles across molecular-dynamics datasets. It is designed to help researchers investigate how mutations, ligands, or conformational states reshape tunnel geometry and the surrounding residue environment.

Directly superimposing multiple tunnel ensembles in three dimensions often produces severe visual clutter, while independently assigned cluster identifiers make cross-dataset comparison ambiguous. Tuner addresses these problems through a coordinated, revisable workflow. It aligns protein structures, proposes geometrically plausible tunnel-cluster correspondences, maps local residue environments onto normalized physical arc length, and preserves the selected correspondence as an inspectable analytical state.

The interface integrates four linked components: **Dataset Compare** for cross-dataset tunnel correspondence, the **Full-path Residue Replacement Map** for localizing residue-environment changes, the **Residue Observer** for side-by-side structural inspection, and an **Evidence Workspace** for examining geometric and ensemble-level support. Together, these views help analysts distinguish visually salient differences from interpretations that remain stable across controls or independent trajectories.

Tuner operates on precomputed tunnel ensembles and aligned PDB snapshots. The processed CYP2J2 case-study dataset and accompanying metadata are available from [Zenodo record 22655693](https://zenodo.org/records/22655693).

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

## Reproducing the CYP2J2 case study

Download the CYP2J2 archive from [Zenodo record 22655693](https://zenodo.org/records/22655693). The archive contains the processed data needed to reproduce the visual analysis reported in the paper. This procedure reproduces the Tuner workflow and its visual observations; it does not rerun the upstream MD simulations or tunnel extraction.

### 1. Prepare the dataset

Extract the archive without changing its directory structure. It should contain:

```text
CYP2J2/
|-- dataset/
|   |-- reference/WT/
|   `-- targets/
|       |-- WT_r2/
|       |-- WT_r3/
|       |-- WT_r4/
|       |-- R111A_r2/
|       |-- R111A_r3/
|       `-- R117A/
`-- MD/
    |-- WT_frames/
    |-- WT_r2_frames/
    |-- WT_r3_frames/
    |-- WT_r4_frames/
    |-- R111A_r2_frames/
    |-- R111A_r3_frames/
    `-- R117A_frames/
```

Edit the `MD_path.txt` file inside each dataset directory so that it contains the absolute path to the corresponding aligned-frame directory. For an archive extracted to `D:\data\CYP2J2`, `dataset/reference/WT/MD_path.txt` should contain:

```text
D:\data\CYP2J2\MD\WT_frames
```

Similarly, `dataset/targets/R111A_r2/MD_path.txt` should contain:

```text
D:\data\CYP2J2\MD\R111A_r2_frames
```

Apply the same pattern to the other target datasets. Each available frame directory should contain `aligned_1.pdb` through `aligned_500.pdb`. If a frame directory is unavailable, its processed tunnel ensemble can still be compared, but frame-specific Residue Observer inspection will not be available.

### 2. Load the seven datasets

Start Tuner and add `dataset/reference/WT` and all six directories under `dataset/targets/`. Use `WT` as the source dataset. Keep the supplied correspondence settings unchanged unless intentionally conducting a sensitivity analysis.

### 3. Reproduce the rejected R117A lead

1. In **Dataset Compare**, select source cluster `WT C1` and target dataset `R117A`.
2. Inspect the accepted `R117A C42 + C6` target family, then open its linked Tunnel Profile and Full-path Residue Replacement Map.
3. Keep `WT C1` fixed and switch the target successively to `WT_r2`, `WT_r3`, and `WT_r4`.
4. Compare the replacement bands and bottleneck context under the same visual encoding. Similar or stronger responses in the WT controls show that the initially salient R117A pattern is not mutation-specific.

This stage is intentionally a negative analytical result: it demonstrates how control switching prevents a visually plausible single-trajectory difference from being promoted into an unsupported allosteric interpretation.

### 4. Reproduce the retained R111A hypothesis

1. Return to **Dataset Compare** and select source cluster `WT C32`.
2. Select `R111A_r2`; inspect the one-to-many target family `C14 + C10`. The strongest distal response should occur approximately within 65.6--78.1% of normalized path length.
3. Switch the target to `R111A_r3`; inspect target cluster `C8`. Its strongest distal response should occur approximately within 75.0--87.5% of normalized path length.
4. Select each highlighted interval in the **Full-path Residue Replacement Map** and open the **Residue Observer**. Compare `WT`, `R111A_r2`, and `R111A_r3` with a shared camera and consistent residue colors.
5. Inspect the local direction of change: both R111A trajectories should show depletion of F148 from, and recruitment of N147 into, the tunnel environment. In the UI sequence numbering, these may appear as PHE S149 and ASN S148.
6. Keep the source cluster, selected distal region, and color assignment fixed while switching through `WT_r2`, `WT_r3`, and `WT_r4`. The WT responses are heterogeneous and do not reproduce the same paired direction consistently.

### 5. Interpret the result conservatively

The reproducible outcome is a localized, directionally repeated R111A-associated residue redistribution in the C32 route family. Because only two R111A trajectories and one R117A trajectory are available, Tuner presents this result as a testable remote-regulation hypothesis, not as proof of an allosteric transmission pathway or altered arachidonic-acid transport. Treat trajectories, rather than their individual frames or tunnel instances, as the independent observational units.

## License

Copyright (c) 2026 Yapeng Liu. All rights reserved. This project is provided for local, non-commercial research evaluation only. Republishing, reproducing, mirroring, redistributing, sublicensing, selling, or incorporating any part of the code or documentation into another distributed work is prohibited without prior written permission. See [LICENSE](https://github.com/TowlGol/Tuner/blob/main/LICENSE) for the complete terms.
