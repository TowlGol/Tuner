"""Pure helpers for the linked Hex evidence workspace.

The functions in this module deliberately separate evidence calculation from
Qt rendering.  A selected cluster relation is an analysis hypothesis; the
returned summaries describe its robustness and do not turn it into ground
truth or causal evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.60)
# Current ablation: automatic ranking uses geometry and the optional population
# term only. Residue similarity remains visible and can still gate eligibility,
# but it contributes no weighted penalty to relation cost.
DEFAULT_RELATION_RESIDUE_WEIGHT = 0.0
DEFAULT_RELATION_POPULATION_WEIGHT = 0.0
DEFAULT_RMSD_FAMILY_MARGIN = 0.79
# Absolute identity gate applied before the relative family margin.  Without
# this gate, a geometrically unrelated target is still selected merely because
# it is the least-bad candidate in that dataset.
DEFAULT_RMSD_MAX_THRESHOLD = 3.0
DEFAULT_WEIGHT_SCENARIOS = (
    ("Geometry", 0.0, 0.0),
    ("Tuned", DEFAULT_RELATION_RESIDUE_WEIGHT, DEFAULT_RELATION_POPULATION_WEIGHT),
    ("Legacy", 0.35, 0.10),
    ("Residue", 0.70, 0.10),
    ("Population", 0.35, 0.30),
)


def _relation_cost(
    row: Mapping,
    scale: float,
    residue_weight: float,
    population_weight: float,
) -> float:
    cost = float(row.get("center_rmsd", float("inf")))
    residue_similarity = row.get("residue_similarity")
    if residue_similarity is not None:
        cost += scale * float(residue_weight) * (1.0 - float(residue_similarity))
    count_similarity = row.get("path_count_similarity")
    if count_similarity is not None:
        cost += scale * float(population_weight) * (1.0 - float(count_similarity))
    if bool(row.get("pipeline_match")):
        cost *= 0.85
    return float(cost)


def mapping_sensitivity(
    candidates: Iterable[Mapping],
    selected_cluster: int,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    scenarios: Sequence[tuple[str, float, float]] = DEFAULT_WEIGHT_SCENARIOS,
) -> dict:
    """Rank one selected target under threshold and weight perturbations."""
    rows = [dict(row) for row in candidates if row.get("center_rmsd") is not None]
    rmsd_values = [float(row["center_rmsd"]) for row in rows if np.isfinite(float(row["center_rmsd"]))]
    scale = max(float(np.median(rmsd_values)) if rmsd_values else 1.0, 1e-6)
    cells: list[dict] = []
    for threshold in thresholds:
        for scenario, residue_weight, population_weight in scenarios:
            ranked = []
            for row in rows:
                similarity = row.get("residue_similarity")
                if float(threshold) > 0.0 and (
                    similarity is None or float(similarity) < float(threshold)
                ):
                    continue
                ranked.append({
                    **row,
                    "scenario_cost": _relation_cost(
                        row, scale, residue_weight, population_weight,
                    ),
                })
            ranked.sort(key=lambda row: (
                float(row["scenario_cost"]),
                int(row.get("target_cluster_id", 0)),
            ))
            rank = next((
                index for index, row in enumerate(ranked, start=1)
                if int(row.get("target_cluster_id", 0)) == int(selected_cluster)
            ), None)
            automatic = int(ranked[0].get("target_cluster_id", 0)) if ranked else None
            cells.append({
                "threshold": float(threshold),
                "scenario": str(scenario),
                "eligible": rank is not None,
                "rank": rank,
                "automatic_cluster": automatic,
                "top": bool(rank == 1),
            })
    eligible = [row for row in cells if row["eligible"]]
    return {
        "thresholds": [float(value) for value in thresholds],
        "scenarios": [str(row[0]) for row in scenarios],
        "cells": cells,
        "selected_cluster": int(selected_cluster),
        "eligible_count": len(eligible),
        "top_count": sum(bool(row["top"]) for row in cells),
        "total_count": len(cells),
        "rank_min": min((int(row["rank"]) for row in eligible), default=None),
        "rank_max": max((int(row["rank"]) for row in eligible), default=None),
    }


def ensemble_consistency_points(mapping_rows: Iterable[Mapping], reference_cluster: int) -> list[dict]:
    """Return one selected relation per target dataset for an ensemble plot."""
    points = []
    for row in mapping_rows:
        payload = dict(row)
        if int(payload.get("reference_cluster_id", -1)) != int(reference_cluster):
            continue
        if payload.get("target_cluster_id") is None:
            continue
        residue_similarity = payload.get("residue_similarity")
        center_rmsd = payload.get("center_rmsd")
        if residue_similarity is None or center_rmsd is None:
            continue
        points.append({
            "dataset_key": str(payload.get("target_dataset") or ""),
            "dataset_label": str(payload.get("target_label") or payload.get("target_dataset") or "target"),
            "target_cluster_id": int(payload.get("target_cluster_id", 0)),
            "residue_similarity": float(residue_similarity),
            "center_rmsd": float(center_rmsd),
            "locked": bool(payload.get("locked")),
            "pipeline_match": bool(payload.get("pipeline_match")),
        })
    return points


def split_path_blocks_by_frame(profiles: Iterable[Mapping], block_count: int = 5) -> list[dict]:
    """Split paths into sequential blocks and retain their real frame ranges."""
    records = []
    for row in profiles:
        raw_path_id = row.get("localPathIndex", row.get("pathIndex"))
        if raw_path_id is None:
            continue
        path_id = int(raw_path_id)
        frame_id = int(row.get("frameId", 0) or 0)
        if path_id >= 0:
            records.append((frame_id, path_id))
    records.sort(key=lambda item: (item[0], item[1]))
    if not records:
        return []
    chunks = np.array_split(np.asarray(records, dtype=np.int64), max(1, int(block_count)))
    return [
        {
            "path_ids": [int(path_id) for _frame_id, path_id in chunk.tolist()],
            "frame_start": int(chunk[0][0]),
            "frame_end": int(chunk[-1][0]),
            "path_count": int(len(chunk)),
        }
        for chunk in chunks if len(chunk)
    ]


def split_path_ids_by_frame(profiles: Iterable[Mapping], block_count: int = 5) -> list[list[int]]:
    """Split paths into sequential frame blocks without treating blocks as replicas."""
    return [
        list(block["path_ids"])
        for block in split_path_blocks_by_frame(profiles, block_count)
    ]


def hotspot_residue_deltas(profile: Mapping, hotspot: Mapping) -> dict[int, float]:
    """Read target-minus-reference residue frequencies at one arc-length hotspot."""
    position_rows = [dict(row) for row in profile.get("position_rows", []) or []]
    if not position_rows:
        return {}
    start = float(hotspot.get("start_fraction", hotspot.get("peak_fraction", 0.5)))
    end = float(hotspot.get("end_fraction", hotspot.get("peak_fraction", 0.5)))
    selected = [
        row for row in position_rows
        if float(row.get("end_fraction", 0.0)) >= start
        and float(row.get("start_fraction", 1.0)) <= end
    ]
    if not selected:
        peak = float(hotspot.get("peak_fraction", 0.5))
        selected = [min(position_rows, key=lambda row: abs(
            0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0))) - peak
        ))]
    reference: dict[int, float] = {}
    target: dict[int, float] = {}
    for row in selected:
        for residue_id, value in dict(row.get("reference_frequency", {}) or {}).items():
            reference[int(residue_id)] = reference.get(int(residue_id), 0.0) + float(value) / len(selected)
        for residue_id, value in dict(row.get("target_frequency", {}) or {}).items():
            target[int(residue_id)] = target.get(int(residue_id), 0.0) + float(value) / len(selected)
    residue_ids = set(reference) | set(target)
    return {
        int(residue_id): float(target.get(residue_id, 0.0) - reference.get(residue_id, 0.0))
        for residue_id in residue_ids
    }


def summarize_ca_distance_frames(
    frame_paths: Iterable[str],
    anchor_sequence_id: int,
    residue_sequence_ids: Iterable[int],
    *,
    max_frames: int | None = None,
) -> dict:
    """Summarize framewise C-alpha distances from one anchor residue.

    The helper reads only requested ``CA`` records, so Evidence can audit a
    trajectory without constructing molecular render objects.  Rigid display
    alignment is intentionally not applied because Euclidean distances are
    invariant under the pipeline rotation/translation.
    """
    anchor_id = int(anchor_sequence_id)
    residue_ids = tuple(dict.fromkeys(
        int(value) for value in residue_sequence_ids
        if int(value) > 0 and int(value) != anchor_id
    ))
    paths = [str(Path(path)) for path in frame_paths if str(path or "").strip()]
    if max_frames is not None and int(max_frames) > 0 and len(paths) > int(max_frames):
        indices = np.linspace(0, len(paths) - 1, int(max_frames), dtype=np.int64)
        paths = [paths[int(index)] for index in indices]
    wanted = {anchor_id, *residue_ids}
    values: dict[int, list[float]] = {residue_id: [] for residue_id in residue_ids}
    readable_frames = 0
    for raw_path in paths:
        positions: dict[int, np.ndarray] = {}
        try:
            with open(raw_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line[:6].strip().upper() != "ATOM":
                        continue
                    if line[12:16].strip().upper() != "CA":
                        continue
                    try:
                        residue_id = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if residue_id not in wanted or residue_id in positions:
                        continue
                    try:
                        positions[residue_id] = np.asarray(
                            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            dtype=np.float64,
                        )
                    except ValueError:
                        continue
                    if len(positions) == len(wanted):
                        break
        except OSError:
            continue
        anchor = positions.get(anchor_id)
        if anchor is None:
            continue
        readable_frames += 1
        for residue_id in residue_ids:
            point = positions.get(residue_id)
            if point is not None:
                values[residue_id].append(float(np.linalg.norm(point - anchor)))

    rows = []
    for residue_id in residue_ids:
        array = np.asarray(values.get(residue_id, ()), dtype=np.float64)
        if not len(array):
            continue
        rows.append({
            "residue_id": int(residue_id),
            "frame_count": int(len(array)),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        })
    return {
        "anchor_sequence_id": anchor_id,
        "requested_frame_count": int(len(paths)),
        "readable_frame_count": int(readable_frames),
        "rows": rows,
        "statistic": "framewise C-alpha Euclidean distance; median and interquartile range",
    }
