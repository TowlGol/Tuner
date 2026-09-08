# Tuner

**Tuner** is an evidence-aware visual analytics system for comparing dynamic protein-tunnel ensembles across molecular-dynamics datasets. It is designed to help researchers investigate how mutations, ligands, or conformational states reshape tunnel geometry and the surrounding residue environment.

Directly superimposing multiple tunnel ensembles in three dimensions often produces severe visual clutter, while independently assigned cluster identifiers make cross-dataset comparison ambiguous. Tuner addresses these problems through a coordinated, revisable workflow. It aligns protein structures, proposes geometrically plausible tunnel-cluster correspondences, maps local residue environments onto normalized physical arc length, and preserves the selected correspondence as an inspectable analytical state.

The interface integrates four linked components: **Dataset Compare** for cross-dataset tunnel correspondence, the **Full-path Residue Replacement Map** for localizing residue-environment changes, the **Residue Observer** for side-by-side structural inspection, and an **Evidence Workspace** for examining geometric and ensemble-level support. Together, these views help analysts distinguish visually salient differences from interpretations that remain stable across controls or independent trajectories.

Tuner operates on precomputed tunnel ensembles and aligned PDB snapshots. Example datasets and accompanying metadata are distributed separately through the project’s Zenodo records.
