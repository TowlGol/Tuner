"""Path-scope matching and arc-length residue comparison helpers.

These helpers deliberately have no Qt dependency so the scientific comparison
logic can be tested independently from the desktop interface.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


# CAVER seeds every tunnel at the user-defined starting point.  The seed sphere
# is commonly smaller than the first biologically meaningful channel section,
# so treating it as the bottleneck pins population summaries to 0% arc length.
# Keep the exclusion relative to path length so it remains stable across paths
# with different point densities.
_BOTTLENECK_START_EXCLUSION_FRACTION = 0.10


def _normalized_keep_ratio(value: float) -> float:
    ratio = max(0.0, min(1.0, float(value or 0.0)))
    return ratio if ratio > 0.0 else 1.0


def _trimmed_length_region(values: Sequence[float], keep_ratio: float) -> dict | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    ratio = _normalized_keep_ratio(keep_ratio)
    trim_each = int(math.floor(len(ordered) * (1.0 - ratio) / 2.0))
    start = min(trim_each, max(0, len(ordered) - 1))
    end = max(start + 1, len(ordered) - trim_each)
    kept = ordered[start:end]
    return {
        "min": float(min(kept)),
        "max": float(max(kept)),
        "count": len(kept),
        "original_count": len(ordered),
        "keep_ratio": ratio,
    }


def _trimmed_exit_region(points: Sequence[Sequence[float]], keep_ratio: float) -> dict | None:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 3:
        return None
    array = array[:, :3]
    ratio = _normalized_keep_ratio(keep_ratio)
    center = np.mean(array, axis=0)
    distances = np.linalg.norm(array - center, axis=1)
    keep_count = max(1, int(math.ceil(len(array) * ratio)))
    kept = array[np.argsort(distances)[:keep_count]]
    kept_center = np.mean(kept, axis=0)
    radius = float(np.max(np.linalg.norm(kept - kept_center, axis=1), initial=0.0))
    return {
        "center": tuple(float(value) for value in kept_center),
        "radius": radius,
        "count": len(kept),
        "original_count": len(array),
        "keep_ratio": ratio,
    }


def _rows_in_chunks(connection, sql_prefix: str, ids: Sequence[int], *, chunk_size: int = 800):
    ordered = sorted({int(value) for value in ids if int(value) > 0})
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start:start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        yield from connection.execute(sql_prefix.format(placeholders=placeholders), chunk)


def match_reference_cluster_paths(
    reference_connection,
    target_connection,
    reference_cluster_id: int,
    *,
    keep_ratio: float = 1.0,
) -> dict:
    """Match one source cluster to path-level candidates across a target dataset.

    The constraints match the established Pipeline/UI semantics: a trimmed
    source exit region, a trimmed source path-length interval, and a target
    residue set that must be a subset of the source cluster's residue universe.
    No target cluster restriction is applied; cluster membership is reported
    only after the path-level match.
    """
    source_rows = list(reference_connection.execute(
        "SELECT id, path_length, end_x, end_y, end_z FROM paths WHERE cluster_id = ? ORDER BY id",
        (int(reference_cluster_id),),
    ))
    valid_source_rows = [
        row for row in source_rows
        if row[1] is not None and all(value is not None for value in row[2:5])
    ]
    if not valid_source_rows:
        return {
            "reference_path_ids": [],
            "target_path_ids": [],
            "target_cluster_counts": {},
            "source_residue_ids": [],
            "prefiltered_count": 0,
            "keep_ratio": _normalized_keep_ratio(keep_ratio),
            "error": "The reference cluster has no valid path geometry.",
        }

    length_region = _trimmed_length_region([row[1] for row in valid_source_rows], keep_ratio)
    exit_region = _trimmed_exit_region([row[2:5] for row in valid_source_rows], keep_ratio)
    source_residue_ids = {
        int(row[0])
        for row in reference_connection.execute(
            "SELECT DISTINCT rp.residue_id FROM residue_paths rp "
            "JOIN paths p ON p.id = rp.path_id WHERE p.cluster_id = ? AND rp.residue_id > 0",
            (int(reference_cluster_id),),
        )
    }
    if not source_residue_ids or not length_region or not exit_region:
        return {
            "reference_path_ids": [int(row[0]) for row in valid_source_rows],
            "target_path_ids": [],
            "target_cluster_counts": {},
            "source_residue_ids": sorted(source_residue_ids),
            "prefiltered_count": 0,
            "keep_ratio": _normalized_keep_ratio(keep_ratio),
            "length_region": length_region,
            "exit_region": exit_region,
            "error": "The reference cluster has no usable residue/length/exit constraints.",
        }

    center = np.asarray(exit_region["center"], dtype=np.float64)
    radius = float(exit_region["radius"]) + 1e-6
    candidate_cluster_by_path: dict[int, int] = {}
    for path_id, cluster_id, path_length, end_x, end_y, end_z in target_connection.execute(
        "SELECT id, cluster_id, path_length, end_x, end_y, end_z FROM paths "
        "WHERE path_length >= ? AND path_length <= ? ORDER BY id",
        (float(length_region["min"]), float(length_region["max"])),
    ):
        if any(value is None for value in (end_x, end_y, end_z)):
            continue
        endpoint = np.asarray((end_x, end_y, end_z), dtype=np.float64)
        if float(np.linalg.norm(endpoint - center)) <= radius:
            candidate_cluster_by_path[int(path_id)] = int(cluster_id or 0)

    candidate_ids = sorted(candidate_cluster_by_path)
    seen: set[int] = set()
    invalid: set[int] = set()
    if candidate_ids:
        query = (
            "SELECT path_id, residue_id FROM residue_paths "
            "WHERE path_id IN ({placeholders}) ORDER BY path_id"
        )
        for path_id, residue_id in _rows_in_chunks(target_connection, query, candidate_ids):
            path_id = int(path_id)
            residue_id = int(residue_id or 0)
            seen.add(path_id)
            if residue_id <= 0 or residue_id not in source_residue_ids:
                invalid.add(path_id)
    matched_ids = [path_id for path_id in candidate_ids if path_id in seen and path_id not in invalid]
    cluster_counts: dict[int, int] = defaultdict(int)
    target_path_ids_by_cluster: dict[int, list[int]] = defaultdict(list)
    for path_id in matched_ids:
        cluster_id = int(candidate_cluster_by_path[path_id])
        cluster_counts[cluster_id] += 1
        target_path_ids_by_cluster[cluster_id].append(int(path_id))

    return {
        "reference_path_ids": [int(row[0]) for row in valid_source_rows],
        "target_path_ids": matched_ids,
        "target_cluster_counts": dict(sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))),
        "target_path_ids_by_cluster": {
            int(cluster_id): sorted(path_ids)
            for cluster_id, path_ids in sorted(
                target_path_ids_by_cluster.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        },
        "source_residue_ids": sorted(source_residue_ids),
        "prefiltered_count": len(candidate_ids),
        "keep_ratio": _normalized_keep_ratio(keep_ratio),
        "length_region": length_region,
        "exit_region": exit_region,
    }


def _scope_path_ids(connection, cluster_id: int | None, path_ids: Iterable[int] | None) -> list[int]:
    if path_ids is not None:
        return sorted({int(path_id) for path_id in path_ids if int(path_id) > 0})
    if cluster_id is None:
        return [int(row[0]) for row in connection.execute("SELECT id FROM paths ORDER BY id")]
    return [
        int(row[0])
        for row in connection.execute(
            "SELECT id FROM paths WHERE cluster_id = ? ORDER BY id",
            (int(cluster_id),),
        )
    ]


def _collect_arc_length_bins(
    connection,
    cluster_id: int | None,
    path_ids: Iterable[int] | None,
    bin_count: int,
) -> tuple[list[dict[int, int]], int, dict]:
    scoped_ids = _scope_path_ids(connection, cluster_id, path_ids)
    residue_counts = [defaultdict(int) for _ in range(bin_count)]
    if not scoped_ids:
        return residue_counts, 0, {}

    query = (
        "SELECT path_id, seq, x, y, z, radius, res_1, res_2, res_3, res_4 "
        "FROM path_points WHERE path_id IN ({placeholders}) ORDER BY path_id, seq"
    )

    bottleneck_fractions: list[float] = []
    bottleneck_radii: list[float] = []

    def consume_path(rows: list[tuple]) -> None:
        if not rows:
            return
        coords = np.asarray([row[2:5] for row in rows], dtype=np.float64)
        if len(coords) <= 1:
            fractions = np.zeros(len(coords), dtype=np.float64)
        else:
            steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(steps)))
            total = float(cumulative[-1])
            fractions = cumulative / total if total > 1e-12 else np.linspace(0.0, 1.0, len(coords))
        radii = np.asarray([
            float(row[5]) if row[5] is not None else np.nan
            for row in rows
        ], dtype=np.float64)
        valid_radius_indices = np.flatnonzero(
            np.isfinite(radii)
            & (radii > 0.0)
            & (fractions >= _BOTTLENECK_START_EXCLUSION_FRACTION)
        )
        # Very short or degenerate profiles may contain no sample outside the
        # seed zone.  Preserve a useful result for those paths as a fallback.
        if not len(valid_radius_indices):
            valid_radius_indices = np.flatnonzero(np.isfinite(radii) & (radii > 0.0))
        if len(valid_radius_indices):
            local_index = int(valid_radius_indices[np.argmin(radii[valid_radius_indices])])
            bottleneck_fractions.append(float(fractions[local_index]))
            bottleneck_radii.append(float(radii[local_index]))
        # Sample each path at common physical arc-length bin centers.  Picking
        # the nearest observed local environment keeps the comparison stable
        # when the same polyline is stored with different point densities;
        # assigning only observed points would leave artificial empty bins on
        # sparsely sampled paths.
        marks: set[tuple[int, int]] = set()
        bin_centers = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
        nearest_indices = np.abs(
            fractions.reshape(-1, 1) - bin_centers.reshape(1, -1)
        ).argmin(axis=0)
        for bin_index, row_index in enumerate(nearest_indices):
            row = rows[int(row_index)]
            for value in row[6:10]:
                residue_id = int(value or 0)
                if residue_id > 0:
                    marks.add((bin_index, residue_id))
        for bin_index, residue_id in marks:
            residue_counts[bin_index][residue_id] += 1

    current_path_id = None
    path_rows: list[tuple] = []
    for row in _rows_in_chunks(connection, query, scoped_ids):
        path_id = int(row[0])
        if current_path_id is not None and path_id != current_path_id:
            consume_path(path_rows)
            path_rows = []
        current_path_id = path_id
        path_rows.append(tuple(row))
    consume_path(path_rows)
    if bottleneck_fractions:
        fraction_array = np.asarray(bottleneck_fractions, dtype=np.float64)
        radius_array = np.asarray(bottleneck_radii, dtype=np.float64)
        bottleneck_summary = {
            "fraction": float(np.median(fraction_array)),
            "fraction_q25": float(np.percentile(fraction_array, 25.0)),
            "fraction_q75": float(np.percentile(fraction_array, 75.0)),
            "radius": float(np.median(radius_array)),
            "radius_q25": float(np.percentile(radius_array, 25.0)),
            "radius_q75": float(np.percentile(radius_array, 75.0)),
            "path_count": len(bottleneck_fractions),
            "definition": (
                "median of each path's minimum-radius point after excluding "
                f"the first {_BOTTLENECK_START_EXCLUSION_FRACTION:.0%} seed zone"
            ),
        }
    else:
        bottleneck_summary = {}
    return residue_counts, len(scoped_ids), bottleneck_summary


def _replacement_score(reference: dict[int, float], target: dict[int, float]) -> float:
    residue_ids = set(reference) | set(target)
    if not residue_ids:
        return 0.0
    shared = sum(min(reference.get(rid, 0.0), target.get(rid, 0.0)) for rid in residue_ids)
    union = sum(max(reference.get(rid, 0.0), target.get(rid, 0.0)) for rid in residue_ids)
    return float(1.0 - shared / union) if union > 1e-12 else 0.0


def detect_residue_substitutions(
    reference_names: Mapping[int, str] | None,
    target_names: Mapping[int, str] | None,
    *,
    sequence_offset: int = 1,
) -> list[dict]:
    """Return explicit amino-acid substitutions shared by two residue maps.

    Tunnel databases use the MD residue index, whereas the interface normally
    displays sequence numbering.  Both identifiers are retained so downstream
    views never have to infer the offset from a formatted label.
    """
    reference = {
        int(residue_id): str(name or "").strip().upper()
        for residue_id, name in dict(reference_names or {}).items()
    }
    target = {
        int(residue_id): str(name or "").strip().upper()
        for residue_id, name in dict(target_names or {}).items()
    }
    substitutions = []
    for residue_id in sorted(set(reference) & set(target)):
        reference_name = reference.get(residue_id, "")
        target_name = target.get(residue_id, "")
        if (
            not reference_name
            or not target_name
            or reference_name in {"UNK", "X", "?"}
            or target_name in {"UNK", "X", "?"}
            or reference_name == target_name
        ):
            continue
        sequence_id = int(residue_id) + int(sequence_offset)
        substitutions.append({
            "residue_id": int(residue_id),
            "sequence_id": int(sequence_id),
            "reference_name": reference_name,
            "target_name": target_name,
            "label": f"S{sequence_id} {reference_name}→{target_name}",
        })
    return substitutions


def _point_at_arc_fraction(points: Sequence[Sequence[float]] | np.ndarray | None, fraction: float):
    """Interpolate a polyline at a physical arc-length fraction."""
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0 or array.shape[1] < 3:
        return None
    array = array[:, :3]
    if len(array) == 1:
        return array[0].copy()
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return array[0].copy()
    target = max(0.0, min(1.0, float(fraction))) * total
    right = min(len(array) - 1, max(1, int(np.searchsorted(cumulative, target, side="right"))))
    left = right - 1
    span = float(cumulative[right] - cumulative[left])
    ratio = 0.0 if span <= 1e-12 else (target - float(cumulative[left])) / span
    return array[left] * (1.0 - ratio) + array[right] * ratio


def annotate_remote_hotspots(
    profile: Mapping | None,
    substitutions: Sequence[Mapping] | None,
    *,
    reference_center_points=None,
    target_center_points=None,
    reference_points_by_sequence: Mapping[int, Sequence[float]] | None = None,
    target_points_by_sequence: Mapping[int, Sequence[float]] | None = None,
    reference_active_site_point: Sequence[float] | None = None,
    target_active_site_point: Sequence[float] | None = None,
    remote_distance_threshold: float = 10.0,
) -> dict:
    """Attach a distal-mutation→path-response chain without claiming causality.

    The aligned tunnel seed is the pipeline's active-site origin.  A hotspot is
    a *remote association candidate* only when the mutation linked to that
    hotspot is outside its residue set and the mutation C-alpha is at least
    ``remote_distance_threshold`` from the active-site origin in every
    structure for which a distance is available.  Mutation-to-region distance
    is retained separately to localize the observed path response.
    """
    result = dict(profile or {})
    mutations = [dict(row) for row in (substitutions or ()) if isinstance(row, Mapping)]
    reference_points = {
        int(residue_id): np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        for residue_id, point in dict(reference_points_by_sequence or {}).items()
        if np.asarray(point).size >= 3
    }
    target_points = {
        int(residue_id): np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        for residue_id, point in dict(target_points_by_sequence or {}).items()
        if np.asarray(point).size >= 3
    }
    reference_active_site = (
        np.asarray(reference_active_site_point, dtype=np.float64).reshape(-1)[:3]
        if reference_active_site_point is not None
        else _point_at_arc_fraction(reference_center_points, 0.0)
    )
    target_active_site = (
        np.asarray(target_active_site_point, dtype=np.float64).reshape(-1)[:3]
        if target_active_site_point is not None
        else _point_at_arc_fraction(target_center_points, 0.0)
    )
    threshold = max(0.0, float(remote_distance_threshold))
    annotated = []
    evidence_rows = []
    for hotspot in result.get("hotspots", []) or []:
        row = dict(hotspot or {})
        peak_fraction = max(0.0, min(1.0, float(row.get("peak_fraction", 0.5))))
        reference_peak = _point_at_arc_fraction(reference_center_points, peak_fraction)
        target_peak = _point_at_arc_fraction(target_center_points, peak_fraction)
        hotspot_residue_ids = {
            int(residue_id)
            for key in ("reference_residue_ids", "target_residue_ids")
            for residue_id in (row.get(key, ()) or ())
            if int(residue_id) > 0
        }
        nearest = None
        for mutation in mutations:
            sequence_id = int(mutation.get("sequence_id", 0) or 0)
            residue_id = int(mutation.get("residue_id", sequence_id - 1) or 0)
            region_distances = {}
            if reference_peak is not None and sequence_id in reference_points:
                region_distances["reference"] = float(np.linalg.norm(reference_points[sequence_id] - reference_peak))
            if target_peak is not None and sequence_id in target_points:
                region_distances["target"] = float(np.linalg.norm(target_points[sequence_id] - target_peak))
            active_site_distances = {}
            if reference_active_site is not None and sequence_id in reference_points:
                active_site_distances["reference"] = float(
                    np.linalg.norm(reference_points[sequence_id] - reference_active_site)
                )
            if target_active_site is not None and sequence_id in target_points:
                active_site_distances["target"] = float(
                    np.linalg.norm(target_points[sequence_id] - target_active_site)
                )
            if not region_distances:
                continue
            region_distance = min(region_distances.values())
            active_site_distance = (
                min(active_site_distances.values()) if active_site_distances else None
            )
            candidate = {
                **mutation,
                # ``distance`` remains the region distance for backward
                # compatibility with saved evidence and scene labels.
                "distance": float(region_distance),
                "reference_distance": region_distances.get("reference"),
                "target_distance": region_distances.get("target"),
                "region_distance": float(region_distance),
                "region_distance_reference": region_distances.get("reference"),
                "region_distance_target": region_distances.get("target"),
                "active_site_distance": active_site_distance,
                "active_site_distance_reference": active_site_distances.get("reference"),
                "active_site_distance_target": active_site_distances.get("target"),
                "outside_hotspot": residue_id not in hotspot_residue_ids,
            }
            if nearest is None or float(candidate["region_distance"]) < float(nearest["region_distance"]):
                nearest = candidate
        if nearest is not None:
            active_site_distance = nearest.get("active_site_distance")
            remote_candidate = bool(
                nearest.get("outside_hotspot")
                and active_site_distance is not None
                and float(active_site_distance) >= threshold
            )
            row.update({
                "nearest_mutation": dict(nearest),
                "nearest_mutation_label": str(nearest.get("label") or "mutation"),
                "mutation_distance": float(nearest["distance"]),
                "mutation_distance_reference": nearest.get("reference_distance"),
                "mutation_distance_target": nearest.get("target_distance"),
                "mutation_region_distance": float(nearest["region_distance"]),
                "mutation_region_distance_reference": nearest.get("region_distance_reference"),
                "mutation_region_distance_target": nearest.get("region_distance_target"),
                "active_site_distance": active_site_distance,
                "active_site_distance_reference": nearest.get("active_site_distance_reference"),
                "active_site_distance_target": nearest.get("active_site_distance_target"),
                "remote_candidate": remote_candidate,
                "remote_distance_threshold": threshold,
                "evidence_status": (
                    "distal perturbation / path-response association candidate"
                    if remote_candidate
                    else "active-site distal criterion not met"
                ),
            })
            evidence_rows.append({
                "region_id": str(row.get("region_id") or row.get("hotspot_id") or "R"),
                "peak_fraction": peak_fraction,
                "replacement_score": float(row.get("max_score", 0.0) or 0.0),
                "mutation_label": str(nearest.get("label") or "mutation"),
                "distance": float(nearest["distance"]),
                "region_distance": float(nearest["region_distance"]),
                "region_distance_reference": nearest.get("region_distance_reference"),
                "region_distance_target": nearest.get("region_distance_target"),
                "active_site_distance": active_site_distance,
                "active_site_distance_reference": nearest.get("active_site_distance_reference"),
                "active_site_distance_target": nearest.get("active_site_distance_target"),
                "remote_candidate": remote_candidate,
            })
        else:
            row.update({
                "remote_candidate": False,
                "remote_distance_threshold": threshold,
                "evidence_status": "distance unavailable",
            })
        annotated.append(row)
    result["hotspots"] = annotated
    result["substitutions"] = mutations
    result["remote_distance_threshold"] = threshold
    result["remote_distance_basis"] = "mutation C-alpha to aligned tunnel seed / active-site origin"
    result["active_site_definition"] = "first point of the aligned center path (pipeline tunnel seed)"
    result["allosteric_evidence"] = evidence_rows
    result["remote_candidate_count"] = sum(bool(row.get("remote_candidate")) for row in evidence_rows)
    reference_bottleneck = dict(result.get("reference_bottleneck", {}) or {})
    target_bottleneck = dict(result.get("target_bottleneck", {}) or {})
    if reference_bottleneck.get("fraction") is not None and target_bottleneck.get("fraction") is not None:
        result["bottleneck_fraction_shift"] = (
            float(target_bottleneck["fraction"]) - float(reference_bottleneck["fraction"])
        )
    if reference_bottleneck.get("radius") is not None and target_bottleneck.get("radius") is not None:
        result["bottleneck_radius_shift"] = (
            float(target_bottleneck["radius"]) - float(reference_bottleneck["radius"])
        )
    return result


def build_residue_change_profile(
    reference_connection,
    target_connection,
    reference_cluster_id: int | None,
    target_cluster_id: int | None,
    *,
    reference_path_ids: Iterable[int] | None = None,
    target_path_ids: Iterable[int] | None = None,
    bin_count: int = 64,
    max_regions: int = 6,
) -> dict:
    """Compare residue environments at equal physical arc-length fractions."""
    bin_count = max(16, int(bin_count))
    reference_counts, reference_total, reference_bottleneck = _collect_arc_length_bins(
        reference_connection, reference_cluster_id, reference_path_ids, bin_count,
    )
    target_counts, target_total, target_bottleneck = _collect_arc_length_bins(
        target_connection, target_cluster_id, target_path_ids, bin_count,
    )
    if not reference_total and not target_total:
        return {"bins": [], "hotspots": [], "position_rows": [], "threshold": 1.0}

    rows: list[dict] = []
    raw_scores: list[float] = []
    for bin_index in range(bin_count):
        reference_frequency = {
            int(residue_id): float(count) / max(1, reference_total)
            for residue_id, count in reference_counts[bin_index].items()
        }
        target_frequency = {
            int(residue_id): float(count) / max(1, target_total)
            for residue_id, count in target_counts[bin_index].items()
        }
        score = _replacement_score(reference_frequency, target_frequency)
        raw_scores.append(score)
        rows.append({
            "bin_index": bin_index,
            "start_fraction": bin_index / bin_count,
            "end_fraction": (bin_index + 1) / bin_count,
            "score_raw": score,
            "reference_frequency": reference_frequency,
            "target_frequency": target_frequency,
        })

    values = np.asarray(raw_scores, dtype=np.float64)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")
    for row, score in zip(rows, smoothed):
        row["score"] = float(score)

    window_size = max(4, int(round(bin_count * 0.12)))
    stride = max(1, window_size // 3)
    candidates = []
    for start_index in range(0, max(1, bin_count - window_size + 1), stride):
        end_index = min(bin_count - 1, start_index + window_size - 1)
        window = smoothed[start_index:end_index + 1]
        candidates.append((float(np.mean(window)), float(np.max(window)), start_index, end_index))
    final_start = max(0, bin_count - window_size)
    if not any(item[2] == final_start for item in candidates):
        window = smoothed[final_start:bin_count]
        candidates.append((float(np.mean(window)), float(np.max(window)), final_start, bin_count - 1))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    regions: list[tuple[int, int]] = []
    selected_scores: list[float] = []
    for mean_score, _max_score, start_index, end_index in candidates:
        if mean_score < 0.20:
            continue
        if any(not (end_index < left or start_index > right) for left, right in regions):
            continue
        regions.append((start_index, end_index))
        selected_scores.append(mean_score)
        if len(regions) >= max(1, int(max_regions)):
            break
    if not regions and float(np.max(smoothed, initial=0.0)) >= 0.15:
        peak_index = int(np.argmax(smoothed))
        start_index = max(0, peak_index - window_size // 2)
        end_index = min(bin_count - 1, start_index + window_size - 1)
        regions.append((start_index, end_index))
        selected_scores.append(float(np.mean(smoothed[start_index:end_index + 1])))

    hotspots = []
    region_payloads = []
    for start_index, end_index in regions:
        region_rows = rows[start_index:end_index + 1]
        reference_average: dict[int, float] = defaultdict(float)
        target_average: dict[int, float] = defaultdict(float)
        for row in region_rows:
            for residue_id, frequency in row["reference_frequency"].items():
                reference_average[int(residue_id)] += float(frequency) / len(region_rows)
            for residue_id, frequency in row["target_frequency"].items():
                target_average[int(residue_id)] += float(frequency) / len(region_rows)
        residue_ids = set(reference_average) | set(target_average)
        deltas = {
            residue_id: target_average.get(residue_id, 0.0) - reference_average.get(residue_id, 0.0)
            for residue_id in residue_ids
        }
        lost_ranked = sorted(
            ((rid, delta) for rid, delta in deltas.items() if delta <= -0.02),
            key=lambda item: item[1],
        )
        gained_ranked = sorted(
            ((rid, delta) for rid, delta in deltas.items() if delta >= 0.02),
            key=lambda item: item[1], reverse=True,
        )
        lost_ids = tuple(int(rid) for rid, _delta in lost_ranked[:4])
        gained_ids = tuple(int(rid) for rid, _delta in gained_ranked[:4])
        reference_ranked = [rid for rid, _value in sorted(reference_average.items(), key=lambda item: item[1], reverse=True)]
        target_ranked = [rid for rid, _value in sorted(target_average.items(), key=lambda item: item[1], reverse=True)]
        reference_ids = tuple(dict.fromkeys((*lost_ids, *reference_ranked)))[:4]
        target_ids = tuple(dict.fromkeys((*gained_ids, *target_ranked)))[:4]
        region_scores = smoothed[start_index:end_index + 1]
        peak_index = start_index + int(np.argmax(region_scores))
        region_payloads.append({
            "start_fraction": start_index / bin_count,
            "end_fraction": (end_index + 1) / bin_count,
            "peak_fraction": (peak_index + 0.5) / bin_count,
            "mean_score": float(np.mean(region_scores)),
            "max_score": float(np.max(region_scores)),
            "reference_residue_ids": reference_ids,
            "target_residue_ids": target_ids,
            "lost_residue_ids": lost_ids,
            "gained_residue_ids": gained_ids,
            "reference_residue_deltas": {int(rid): float(delta) for rid, delta in lost_ranked[:8]},
            "target_residue_deltas": {int(rid): float(delta) for rid, delta in gained_ranked[:8]},
            "reference_motif": "residue environment",
            "target_motif": "residue environment",
            "transition_type": "path_residue_hotspot",
        })

    region_payloads.sort(key=lambda row: -float(row.get("max_score", 0.0)))
    for rank, region in enumerate(region_payloads[:max_regions], start=1):
        hotspot = dict(region)
        hotspot["hotspot_id"] = f"R{rank}"
        hotspot["region_id"] = f"R{rank}"
        hotspots.append(hotspot)

    return {
        "bins": [
            {
                "bin_index": int(row["bin_index"]),
                "start_fraction": float(row["start_fraction"]),
                "end_fraction": float(row["end_fraction"]),
                "score": float(row.get("score", 0.0)),
            }
            for row in rows
        ],
        "hotspots": hotspots,
        "position_rows": rows,
        "threshold": float(min(selected_scores) if selected_scores else 1.0),
        "reference_path_count": int(reference_total),
        "target_path_count": int(target_total),
        "reference_bottleneck": reference_bottleneck,
        "target_bottleneck": target_bottleneck,
        "position_basis": "physical_arc_length",
    }
