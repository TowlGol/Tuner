"""Residue-conditioned chemical and path-dynamic profile properties.

The values in this module are descriptive residue-level features.  They are
not interaction energies or free energies.  Properties are derived from the
residue identity attached to each path point and are therefore available for
both legacy SQLite and pickle datasets.
"""
from __future__ import annotations

import csv
import os
import zlib
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


# Kyte-Doolittle hydropathy values.  The existing path ``hydrophobicity``
# field is retained as the input qss value; this table provides an explicit
# residue-conditioned counterpart.
_HYDROPATHY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5,
    "CYS": 2.5, "GLN": -3.5, "GLU": -3.5, "GLY": -0.4,
    "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8,
    "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

# Zimmerman polarity values normalized to [0, 1].  Keeping the scale
# normalized makes mixtures of four neighboring residues easy to compare.
_POLARITY_RAW = {
    "ALA": 8.1, "ARG": 10.5, "ASN": 11.6, "ASP": 13.0,
    "CYS": 5.5, "GLN": 10.5, "GLU": 12.3, "GLY": 9.0,
    "HIS": 10.4, "ILE": 5.2, "LEU": 4.9, "LYS": 11.3,
    "MET": 5.7, "PHE": 5.2, "PRO": 8.0, "SER": 9.2,
    "THR": 8.6, "TRP": 5.4, "TYR": 6.2, "VAL": 5.9,
}
_POLARITY_MIN = min(_POLARITY_RAW.values())
_POLARITY_MAX = max(_POLARITY_RAW.values())
_POLARITY = {
    key: (value - _POLARITY_MIN) / (_POLARITY_MAX - _POLARITY_MIN)
    for key, value in _POLARITY_RAW.items()
}

# Approximate formal charge at near-neutral pH.  Histidine variants are kept
# neutral because their protonation state is not encoded in tunnel data.
_CHARGE = {name: 0.0 for name in _HYDROPATHY}
_CHARGE.update({"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0})

_HBD = {name: 0.0 for name in _HYDROPATHY}
_HBD.update({name: 1.0 for name in ("ARG", "ASN", "CYS", "GLN", "HIS", "LYS", "SER", "THR", "TRP", "TYR")})
_HBA = {name: 0.0 for name in _HYDROPATHY}
_HBA.update({name: 1.0 for name in ("ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "SER", "THR", "TYR")})

PROPERTY_KEYS = (
    "hydrophobicity",
    "polarity",
    "charge",
    "hbond_donor",
    "hbond_acceptor",
)

PROPERTY_LABELS = {
    "hydrophobicity": "Residue Hydrophobicity",
    "polarity": "Polarity",
    "charge": "Charge",
    "hbond_donor": "H-bond Donor",
    "hbond_acceptor": "H-bond Acceptor",
    "hbond": "H-bond Capacity",
    "throughput": "Throughput",
    "path_progress": "Path Progress",
    "bottleneck_score": "Bottleneck Score",
    "residue_turnover": "Residue Turnover",
}

MATERIALIZED_PROPERTY_VERSION = 1
MATERIALIZED_PROPERTY_KEYS = (
    "residue_hydrophobicity",
    "polarity",
    "charge",
    "hbond_donor",
    "hbond_acceptor",
    "hbond",
    "path_progress",
    "bottleneck_score",
    "residue_turnover",
)
MATERIALIZED_PROPERTY_SCHEMA = ",".join(MATERIALIZED_PROPERTY_KEYS)


def normalize_residue_name(value: object) -> str:
    """Normalize PDB/CSV residue names, including common histidine aliases."""
    name = str(value or "").strip().upper()
    if not name:
        return "UNK"
    aliases = {
        "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "ASH": "ASP", "GLH": "GLU", "CYM": "CYS", "CYX": "CYS",
    }
    return aliases.get(name, name)


def property_names() -> tuple[str, ...]:
    return PROPERTY_KEYS


def property_label(key: str) -> str:
    return PROPERTY_LABELS.get(str(key), str(key).replace("_", " ").title())


def residue_property(residue_name: object, key: str) -> float:
    """Return a descriptive numeric feature, with neutral fallback for UNK."""
    name = normalize_residue_name(residue_name)
    table = {
        "hydrophobicity": _HYDROPATHY,
        "polarity": _POLARITY,
        "charge": _CHARGE,
        "hbond_donor": _HBD,
        "hbond_acceptor": _HBA,
    }.get(str(key))
    if table is None:
        raise KeyError(f"Unknown residue property: {key}")
    return float(table.get(name, 0.0))


def load_residue_name_map(path: str | None) -> Dict[int, str]:
    """Load ``Residue_ID -> Residue_name`` from the optional statistics CSV."""
    if not path:
        return {}
    abs_path = os.path.abspath(str(path))
    if not os.path.isfile(abs_path):
        return {}
    if not abs_path.lower().endswith(".csv"):
        return load_residue_name_map_from_pdb(abs_path)
    result: Dict[int, str] = {}
    try:
        with open(abs_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {str(field).strip().lower(): field for field in (reader.fieldnames or [])}
            id_field = fields.get("residue_id")
            name_field = fields.get("residue_name")
            if not id_field or not name_field:
                return result
            for row in reader:
                try:
                    residue_id = int(float(str(row.get(id_field, "")).strip()))
                except (TypeError, ValueError):
                    continue
                if residue_id > 0:
                    result[residue_id] = normalize_residue_name(row.get(name_field))
    except (OSError, csv.Error):
        return {}
    return result


def load_residue_name_map_from_pdb(path: str | None) -> Dict[int, str]:
    """Best-effort fallback for representative PDB files."""
    if not path or not os.path.isfile(str(path)):
        return {}
    result: Dict[int, str] = {}
    try:
        with open(str(path), "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
                    continue
                try:
                    residue_id = int(line[22:26].strip())
                except ValueError:
                    continue
                if residue_id > 0:
                    result.setdefault(residue_id, normalize_residue_name(line[17:20]))
    except OSError:
        return {}
    return result


def _coerce_slot_arrays(residue_arrays: Sequence[Iterable[object]], length: int) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for values in list(residue_arrays)[:4]:
        arr = np.asarray(values if values is not None else [], dtype=np.int32).reshape(-1)
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)), constant_values=0)
        arrays.append(arr[:length])
    while len(arrays) < 4:
        arrays.append(np.zeros(length, dtype=np.int32))
    return arrays


def derive_residue_properties(
    residue_arrays: Sequence[Iterable[object]],
    residue_name_map: Mapping[int, str] | None = None,
    length: int | None = None,
) -> Dict[str, list[float]]:
    """Derive per-slot and local mean residue properties for a path."""
    raw_arrays = [np.asarray(values if values is not None else [], dtype=np.int32).reshape(-1) for values in residue_arrays]
    count = int(length if length is not None else max((len(arr) for arr in raw_arrays), default=0))
    arrays = _coerce_slot_arrays(raw_arrays, count)
    names = residue_name_map or {}
    result: Dict[str, list[float]] = {}
    valid = np.zeros(count, dtype=np.float32)
    for slot, residue_ids in enumerate(arrays, start=1):
        mask = residue_ids > 0
        valid += mask.astype(np.float32)
        for key in PROPERTY_KEYS:
            values = np.asarray(
                [residue_property(names.get(int(residue_id), "UNK"), key) if residue_id > 0 else 0.0 for residue_id in residue_ids],
                dtype=np.float32,
            )
            result[f"{key}_{slot}"] = values.tolist()
    denominator = np.where(valid > 0, valid, 1.0)
    for key in PROPERTY_KEYS:
        stacked = np.asarray([result[f"{key}_{slot}"] for slot in range(1, 5)], dtype=np.float32)
        result[f"residue_{key}"] = (stacked.sum(axis=0) / denominator).tolist()
        # Short aliases make the profile convenient for downstream analysis;
        # the explicit ``residue_`` names remain the chart-facing contract.
        if key != "hydrophobicity":
            result[key] = result[f"residue_{key}"]
    result["hbond"] = (
        np.asarray(result["residue_hbond_donor"], dtype=np.float32)
        + np.asarray(result["residue_hbond_acceptor"], dtype=np.float32)
    ).tolist()
    return result


def derive_path_dynamic_properties(
    radius: Iterable[object],
    path_length: Iterable[object],
    throughput: Iterable[object],
    residue_arrays: Sequence[Iterable[object]],
    frame_id: int = 0,
) -> Dict[str, list[float]]:
    """Derive stable, non-energy dynamic descriptors along a path."""
    radius_arr = np.asarray(radius if radius is not None else [], dtype=np.float32).reshape(-1)
    length_arr = np.asarray(path_length if path_length is not None else [], dtype=np.float32).reshape(-1)
    throughput_arr = np.asarray(throughput if throughput is not None else [], dtype=np.float32).reshape(-1)
    count = max(len(radius_arr), len(length_arr), len(throughput_arr), *(len(np.asarray(a).reshape(-1)) for a in residue_arrays))
    def fit(values: np.ndarray) -> np.ndarray:
        if len(values) >= count:
            return values[:count]
        return np.pad(values, (0, count - len(values)), constant_values=0.0)
    radius_arr, length_arr, throughput_arr = fit(radius_arr), fit(length_arr), fit(throughput_arr)
    progress = np.zeros(count, dtype=np.float32)
    if count:
        endpoint = float(length_arr[-1])
        if endpoint > 1e-8:
            progress = np.clip(length_arr / endpoint, 0.0, 1.0).astype(np.float32)
        elif count > 1:
            progress = np.linspace(0.0, 1.0, count, dtype=np.float32)
    score = np.zeros(count, dtype=np.float32)
    finite_radius = radius_arr[np.isfinite(radius_arr)]
    if count and len(finite_radius):
        lo, hi = float(np.min(finite_radius)), float(np.max(finite_radius))
        if hi - lo > 1e-8:
            score = np.clip(1.0 - (radius_arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        else:
            score.fill(0.5)
    slots = _coerce_slot_arrays(residue_arrays, count)
    turnover = np.zeros(count, dtype=np.float32)
    if count > 1:
        changes = sum((slots[index][1:] != slots[index][:-1]).astype(np.float32) for index in range(4))
        turnover[1:] = changes / 4.0
    return {
        "frame": np.full(count, float(frame_id), dtype=np.float32).tolist(),
        "path_progress": progress.tolist(),
        "bottleneck_score": score.tolist(),
        "residue_turnover": turnover.tolist(),
        "throughput": throughput_arr.tolist(),
    }


def encode_materialized_properties(properties: Mapping[str, Iterable[object]], point_count: int) -> bytes:
    """Encode the persisted per-point property matrix as compressed float32."""
    count = max(0, int(point_count))
    matrix = np.zeros((count, len(MATERIALIZED_PROPERTY_KEYS)), dtype=np.float32)
    for column, key in enumerate(MATERIALIZED_PROPERTY_KEYS):
        values = np.asarray(properties.get(key, []), dtype=np.float32).reshape(-1)
        copy_count = min(count, len(values))
        if copy_count:
            matrix[:copy_count, column] = values[:copy_count]
    return zlib.compress(matrix.tobytes(order="C"), level=6)


def decode_materialized_properties(
    payload: bytes,
    point_count: int,
    schema: str = MATERIALIZED_PROPERTY_SCHEMA,
) -> Dict[str, list[float]]:
    """Decode persisted properties, returning an empty mapping if incompatible."""
    if not payload:
        return {}
    keys = tuple(item.strip() for item in str(schema or "").split(",") if item.strip())
    if not keys:
        return {}
    count = max(0, int(point_count))
    try:
        raw = zlib.decompress(bytes(payload))
        values = np.frombuffer(raw, dtype=np.float32)
    except (TypeError, ValueError, zlib.error):
        return {}
    expected = count * len(keys)
    if len(values) != expected:
        return {}
    matrix = values.reshape(count, len(keys))
    result = {key: matrix[:, column].tolist() for column, key in enumerate(keys)}
    # Keep the explicit chart-facing aliases synchronized with short API keys.
    for short_key in ("polarity", "charge", "hbond_donor", "hbond_acceptor"):
        if short_key in result:
            result[f"residue_{short_key}"] = result[short_key]
    return result
