"""
Main application window.
Replaces App.vue - QMainWindow with toolbar, side panel, 3D viewer, charts.
"""
import os
import json
import tempfile
import time
import re
import csv
import pickle
import math
import html
from collections import OrderedDict, defaultdict
from itertools import combinations
from typing import Optional, Callable
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QDockWidget, QTabWidget, QToolBar, QLabel, QPushButton,
    QComboBox, QSlider, QStatusBar, QFileDialog, QMessageBox,
    QDialog, QLineEdit, QDialogButtonBox, QApplication,
    QToolButton, QMenu, QWidgetAction, QListView, QCheckBox, QScrollArea, QProgressBar, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QAbstractItemView, QStyledItemDelegate,
    QStackedWidget, QSpinBox, QDoubleSpinBox,
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QObject, QEvent, QVariantAnimation, QEasingCurve,
    QAbstractAnimation,
)
from PySide6.QtGui import (
    QAction, QShortcut, QKeySequence, QFontMetrics, QColor, QBrush, QPainter, QPen, QCursor,
)

from TopoTunnel_UI.app.widgets.residue_panel import ResiduePanel
from TopoTunnel_UI.app.widgets.path_panel import PathPanel
from TopoTunnel_UI.app.widgets.dataset_panel import DatasetPanel
from TopoTunnel_UI.app.widgets.residue_compare_panel import ResidueComparePanel
from TopoTunnel_UI.app.widgets.residue_combination_panel import ResidueCombinationPanel
from TopoTunnel_UI.app.widgets.dataset_compare_panel import DatasetComparePanel
from TopoTunnel_UI.app.widgets.evidence_panel import EvidencePanel
from TopoTunnel_UI.app.widgets.protein_viewer_panel import ProteinViewerPanel
from TopoTunnel_UI.app.models.path_model import PathTableModel
from TopoTunnel_UI.db.multi_dataset_db import MultiDatasetDatabase
from TopoTunnel_UI.core.residue_combination import (
    bottleneck_point_indices,
    combination_rows_from_profiles,
    combination_centroid,
    combination_property_trend,
    combination_property_value,
    compare_combination_row_sets,
    exit_region_from_path_coords,
    filter_path_ids_by_exit_region,
    filter_profiles_by_residue_subset,
    profile_residue_set,
    residue_set_from_profiles,
    trimmed_exit_region_from_path_coords,
    trimmed_length_range,
)
from TopoTunnel_UI.core.dataset_compare import (
    annotate_remote_hotspots,
    build_residue_change_profile,
    detect_residue_substitutions,
    match_reference_cluster_paths,
)
from TopoTunnel_UI.core.evidence import (
    DEFAULT_RMSD_FAMILY_MARGIN,
    DEFAULT_RMSD_MAX_THRESHOLD,
    ensemble_consistency_points,
    hotspot_residue_deltas,
    mapping_sensitivity,
    split_path_blocks_by_frame,
    summarize_ca_distance_frames,
)
from TopoTunnel_UI.core.residue_properties import load_residue_name_map_from_pdb
from TopoTunnel_UI.core.state import AppState
from TopoTunnel_UI.core.exporter import export_paths_to_pdb
from TopoTunnel_UI.vis.tunnel_viewer import DEFAULT_PATH_COORDINATE_MODE, TunnelViewer3D
from TopoTunnel_UI.vis.protein_model import (
    compose_pipeline_row_transforms,
    kabsch_pipeline_row_transform,
    read_pipeline_backbone_map,
)
from TopoTunnel_UI.vis.pse_parser import parse_pse_file, pse_protein_to_pdb_string
from TopoTunnel_UI.vis.charts import (
    ProfileChart,
    ResidueCombinationRegionChart,
    StatisticsChart,
)


class DataLoader(QObject):
    """Background worker for loading 3D render data."""
    finished = Signal(list, str, int)  # data, coordinate_mode, request_id
    failed = Signal(str, str, int)  # error, coordinate_mode, request_id
    progress = Signal(int)

    def __init__(self, db, frame_min=None, frame_max=None, coordinate_mode="current", request_id=0):
        super().__init__()
        self.db = db
        self.frame_min = frame_min
        self.frame_max = frame_max
        self.coordinate_mode = coordinate_mode if coordinate_mode in {"current", "original"} else "current"
        self.request_id = int(request_id)

    def run(self):
        try:
            if self.coordinate_mode == "original":
                data = self.db.get_all_original_render_coords(self.frame_min, self.frame_max)
            else:
                data = self.db.get_all_render_coords(self.frame_min, self.frame_max)
        except Exception as exc:
            self.failed.emit(str(exc), self.coordinate_mode, self.request_id)
            return
        self.finished.emit(data, self.coordinate_mode, self.request_id)


class DistanceEvidenceTaskWorker(QObject):
    """Read framewise C-alpha distances without blocking the Qt event loop."""

    finished = Signal(object, object, object)
    failed = Signal(object, str)

    def __init__(
        self,
        cache_key: tuple,
        reference_paths: list[str],
        target_paths: list[str],
        anchor_sequence_id: int,
        residue_sequence_ids: tuple[int, ...],
    ):
        super().__init__()
        self.cache_key = tuple(cache_key)
        self.reference_paths = list(reference_paths)
        self.target_paths = list(target_paths)
        self.anchor_sequence_id = int(anchor_sequence_id)
        self.residue_sequence_ids = tuple(int(value) for value in residue_sequence_ids)

    def run(self):
        try:
            reference = summarize_ca_distance_frames(
                self.reference_paths,
                self.anchor_sequence_id,
                self.residue_sequence_ids,
                max_frames=500,
            )
            target = summarize_ca_distance_frames(
                self.target_paths,
                self.anchor_sequence_id,
                self.residue_sequence_ids,
                max_frames=500,
            )
        except Exception as exc:
            self.failed.emit(self.cache_key, str(exc))
            return
        self.finished.emit(self.cache_key, reference, target)


class ResidueCombinationTaskWorker(QObject):
    """Background worker for residue-combination queries/calculations."""

    finished = Signal(dict)

    def __init__(self, task: str, payload: dict):
        super().__init__()
        self.task = str(task)
        self.payload = dict(payload or {})

    def run(self):
        try:
            if self.task == "query_match":
                result = self._run_query_match()
            elif self.task == "compute_match":
                result = self._run_compute_match()
            else:
                result = {"ok": False, "error": f"unknown_task:{self.task}"}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        result["task"] = self.task
        result["request_id"] = int(self.payload.get("request_id", 0) or 0)
        self.finished.emit(result)

    def _run_query_match(self) -> dict:
        db = self.payload["db"]
        source_key = str(self.payload.get("source_key") or "")
        target_key = str(self.payload.get("target_key") or "")
        source_path_ids = [int(path_id) for path_id in self.payload.get("source_path_ids", [])]
        keep_ratio = float(self.payload.get("keep_ratio", 1.0) or 1.0)
        frame_min = self.payload.get("frame_min")
        frame_max = self.payload.get("frame_max")
        datasets = list(self.payload.get("datasets", []) or [])

        source_binding = getattr(db, "_datasets_by_key", {}).get(source_key)
        target_binding = getattr(db, "_datasets_by_key", {}).get(target_key)
        source_db = source_binding.db if source_binding is not None else None
        target_db = target_binding.db if target_binding is not None else None
        if source_db is None or target_db is None:
            return {"ok": False, "error": "Dataset binding is unavailable, so matched paths cannot be queried"}

        source_local_path_ids = db.get_local_path_ids_for_dataset(source_key, source_path_ids)
        source_profiles = source_db.get_path_profiles(source_local_path_ids)
        source_residue_ids = residue_set_from_profiles(source_profiles)
        if not source_residue_ids:
            return {"ok": False, "error": "No valid residues were extracted from the source paths, so matched paths cannot be queried"}

        source_exit_region = trimmed_exit_region_from_path_coords(
            source_db.get_render_coords(source_local_path_ids),
            keep_ratio,
        )
        if not source_exit_region:
            return {"ok": False, "error": "No valid exit positions were extracted from the source paths, so matched paths cannot be queried"}

        source_length_map = source_db.get_path_length_map(source_local_path_ids)
        source_length_region = trimmed_length_range(source_length_map, keep_ratio)
        if not source_length_region:
            return {"ok": False, "error": "No valid path lengths were extracted from the source paths, so matched paths cannot be queried"}
        source_length_min = float(source_length_region["min"])
        source_length_max = float(source_length_region["max"])

        target_global_path_ids = db.get_dataset_path_ids(target_key, frame_min, frame_max)
        target_local_path_ids = db.get_local_path_ids_for_dataset(target_key, target_global_path_ids)
        target_local_path_ids = filter_path_ids_by_exit_region(
            target_db.get_render_coords(target_local_path_ids),
            source_exit_region,
        )
        target_length_map = target_db.get_path_length_map(target_local_path_ids)
        target_local_path_ids = [
            int(path_id)
            for path_id, path_length in target_length_map.items()
            if float(path_length) >= source_length_min and float(path_length) <= source_length_max
        ]
        target_local_path_ids.sort()
        target_profiles = target_db.get_path_profiles(target_local_path_ids)
        matched_target_profiles = filter_profiles_by_residue_subset(target_profiles, source_residue_ids)
        matched_target_local_ids = [
            int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
            for profile in matched_target_profiles
            if int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1) > 0
        ]
        target_path_id_base = int(getattr(target_binding, "path_id_base", 0) or 0)
        matched_target_global_ids = [
            target_path_id_base + int(local_id)
            for local_id in matched_target_local_ids
        ]
        source_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == source_key), {})
        target_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == target_key), {})
        summary = (
            f"Match query completed | Source [{source_dataset.get('prefix', source_key)}]: {len(source_profiles)} paths | "
            f"Keep ratio {keep_ratio:.2f} | "
            f"Exit radius {float(source_exit_region.get('radius', 0.0)):.3f} | "
            f"Length range [{source_length_min:.3f}, {source_length_max:.3f}] | "
            f"Target [{target_dataset.get('prefix', target_key)}] prefiltered: {len(target_profiles)} paths | "
            f"Residue-matched: {len(matched_target_profiles)} paths"
        )
        return {
            "ok": True,
            "source_profiles_count": len(source_profiles),
            "target_profiles_count": len(target_profiles),
            "matched_target_profiles_count": len(matched_target_profiles),
            "matched_target_global_ids": matched_target_global_ids,
            "target_key": target_key,
            "summary": summary,
        }

    def _run_compute_match(self) -> dict:
        db = self.payload["db"]
        file_a_key = str(self.payload.get("file_a_key") or "")
        file_b_key = str(self.payload.get("file_b_key") or "")
        using_matched_paths = bool(self.payload.get("using_matched_paths"))
        matched_target_key = str(self.payload.get("matched_target_key") or "")
        source_key = str(self.payload.get("source_key") or "")
        matched_target_path_ids = {int(path_id) for path_id in self.payload.get("matched_target_path_ids", [])}
        effective_path_ids = [int(path_id) for path_id in self.payload.get("effective_path_ids", [])]

        profiles_by_dataset_key: dict[str, list[dict]] = {}
        if using_matched_paths:
            target_path_ids = [
                int(path_id)
                for path_id in effective_path_ids
                if db.get_dataset_key_for_path_id(int(path_id)) == matched_target_key
                and int(path_id) in matched_target_path_ids
            ]
            source_path_ids = [
                int(path_id)
                for path_id in effective_path_ids
                if db.get_dataset_key_for_path_id(int(path_id)) == source_key
            ]
            source_binding = getattr(db, "_datasets_by_key", {}).get(source_key)
            target_binding = getattr(db, "_datasets_by_key", {}).get(matched_target_key)
            source_db = source_binding.db if source_binding is not None else None
            target_db = target_binding.db if target_binding is not None else None
            if source_db is None or target_db is None:
                return {"ok": False, "error": "Dataset binding is unavailable, so differences cannot be computed"}
            if not source_path_ids:
                return {"ok": False, "error": "No source-dataset paths are currently selected, so differences cannot be computed"}
            source_local_path_ids = db.get_local_path_ids_for_dataset(source_key, source_path_ids)
            target_local_path_ids = db.get_local_path_ids_for_dataset(matched_target_key, target_path_ids)
            profiles_by_dataset_key = {
                str(source_key): source_db.get_path_profiles(source_local_path_ids),
                str(matched_target_key): target_db.get_path_profiles(target_local_path_ids),
            }
            if file_a_key not in profiles_by_dataset_key or file_b_key not in profiles_by_dataset_key:
                return {"ok": False, "error": "Matched-path source/target datasets do not match the current File A/File B selection"}
            summary = (
                f"{self.payload.get('last_compare_summary', '')} | "
                f"Target paths kept after manual filtering: {len(profiles_by_dataset_key.get(matched_target_key, []))} | "
                f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
            )
        else:
            if not effective_path_ids:
                return {"ok": False, "error": "No paths are currently selected, so differences cannot be computed"}
            for dataset_key in (file_a_key, file_b_key):
                dataset_binding = getattr(db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    return {"ok": False, "error": "Dataset binding is unavailable, so differences cannot be computed"}
                local_path_ids = db.get_local_path_ids_for_dataset(dataset_key, effective_path_ids)
                profiles_by_dataset_key[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
            if not profiles_by_dataset_key.get(file_a_key) and not profiles_by_dataset_key.get(file_b_key):
                return {"ok": False, "error": "The current selection does not contain any File A or File B paths to compare"}
            summary = (
                f"Current selection compare | "
                f"File A: {len(profiles_by_dataset_key.get(file_a_key, []))} paths | "
                f"File B: {len(profiles_by_dataset_key.get(file_b_key, []))} paths | "
                f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
            )

        file_a_profiles = profiles_by_dataset_key.get(file_a_key, [])
        file_b_profiles = profiles_by_dataset_key.get(file_b_key, [])
        compare_rows = compare_combination_row_sets(
            combination_rows_from_profiles(file_a_profiles, bottleneck_only=False),
            combination_rows_from_profiles(file_b_profiles, bottleneck_only=False),
            include_right_only=True,
            left_total_paths=len(file_a_profiles),
            right_total_paths=len(file_b_profiles),
        )
        bottleneck_compare_rows = compare_combination_row_sets(
            combination_rows_from_profiles(file_a_profiles, bottleneck_only=True),
            combination_rows_from_profiles(file_b_profiles, bottleneck_only=True),
            include_right_only=True,
            left_total_paths=len(file_a_profiles),
            right_total_paths=len(file_b_profiles),
        )
        return {
            "ok": True,
            "file_a_key": file_a_key,
            "file_b_key": file_b_key,
            "file_a_profiles_count": len(file_a_profiles),
            "file_b_profiles_count": len(file_b_profiles),
            "profiles_by_dataset_key": profiles_by_dataset_key,
            "compare_rows": compare_rows,
            "bottleneck_compare_rows": bottleneck_compare_rows,
            "summary": summary,
        }


class DockResizeHandle(QFrame):
    """Slim drag handle shown at the right edge of the control panel."""

    def __init__(
        self,
        get_width: Callable[[], int],
        apply_width: Callable[[int], None],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._get_width = get_width
        self._apply_width = apply_width
        self._dragging = False
        self._start_global_x = 0
        self._start_width = 0
        self.setCursor(Qt.SizeHorCursor)
        self.setFixedWidth(8)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Drag to resize Control Panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            "QFrame{background:#f8fafc;border-left:1px solid #e5e7eb;}"
            "QFrame:hover{background:#e2e8f0;border-left:1px solid #cbd5e1;}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_global_x = int(event.globalPosition().x())
            self._start_width = int(self._get_width())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = int(event.globalPosition().x()) - self._start_global_x
            self._apply_width(self._start_width + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DatasetMultiSelectButton(QToolButton):
    """Compact checkable dataset dropdown that keeps the menu open."""

    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._choices: list[tuple[str, str]] = []
        self._checkboxes: dict[str, QCheckBox] = {}
        self._menu = QMenu(self)
        self._menu_widget = QWidget(self._menu)
        self._menu_layout = QVBoxLayout(self._menu_widget)
        self._menu_layout.setContentsMargins(8, 6, 8, 6)
        self._menu_layout.setSpacing(3)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(self._menu_widget)
        self._menu.addAction(action)
        self.setMenu(self._menu)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setText("Select datasets")

    def set_choices(self, choices, *, checked_keys=None) -> None:
        checked = set(str(key) for key in (checked_keys or self.checked_keys()) if str(key))
        while self._menu_layout.count():
            item = self._menu_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes.clear()
        self._choices = [(str(label), str(key)) for label, key in (choices or []) if str(key)]
        for label, key in self._choices:
            checkbox = QCheckBox(label, self._menu_widget)
            checkbox.setChecked(key in checked)
            checkbox.toggled.connect(lambda _checked, dataset_key=key: self._on_checkbox_toggled(dataset_key))
            self._menu_layout.addWidget(checkbox)
            self._checkboxes[key] = checkbox
        self._update_text()

    def checked_keys(self) -> list[str]:
        return [key for _label, key in self._choices if self._checkboxes.get(key) is not None and self._checkboxes[key].isChecked()]

    def set_checked_keys(self, keys, *, emit: bool = True) -> None:
        wanted = set(str(key) for key in (keys or []) if str(key))
        for key, checkbox in self._checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key in wanted)
            checkbox.blockSignals(False)
        self._update_text()
        if emit:
            self.selection_changed.emit(self.checked_keys())

    def _on_checkbox_toggled(self, _dataset_key: str) -> None:
        self._update_text()
        self.selection_changed.emit(self.checked_keys())

    def _update_text(self) -> None:
        labels = [label for label, key in self._choices if key in set(self.checked_keys())]
        if not labels:
            self.setText("Select datasets")
        elif len(labels) <= 2:
            self.setText(", ".join(labels))
        else:
            self.setText(f"{', '.join(labels[:2])} +{len(labels) - 2}")
        self.setToolTip("Selected MD datasets: " + (", ".join(labels) if labels else "none"))


class ObserverTimelineWidget(QWidget):
    """Compact frame timeline for the residue observer."""

    frame_clicked = Signal(str, int)
    zoom_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._zoom = 1.0
        self._hit_targets: list[dict] = []
        self._timeline_signature = None
        self.setMinimumHeight(72)
        self.setMaximumHeight(180)
        self.setToolTip("Each row is one selected MD dataset; click a highlighted frame to synchronize all observers.")

    def set_zoom_percent(self, value: int) -> None:
        self._zoom = max(1.0, min(100.0, int(value) / 100.0))
        self.update()

    def zoom_percent(self) -> int:
        return int(round(float(self._zoom) * 100.0))

    def set_timelines(
        self,
        top_frames,
        top_highlighted_frames,
        top_current_frame: int | None,
        bottom_frames,
        bottom_highlighted_frames,
        bottom_current_frame: int | None,
    ) -> None:
        self.set_dataset_timelines([
            {
                "key": "top",
                "label": "A",
                "frames": top_frames,
                "highlighted_frames": top_highlighted_frames,
                "current_frame": top_current_frame,
            },
            {
                "key": "bottom",
                "label": "B",
                "frames": bottom_frames,
                "highlighted_frames": bottom_highlighted_frames,
                "current_frame": bottom_current_frame,
            },
        ])

    def set_dataset_timelines(self, rows) -> None:
        normalized = []
        signature_rows = []
        for index, row in enumerate(rows or []):
            frames = tuple(sorted({int(frame) for frame in (row.get("frames") or [])}))
            highlighted = tuple(sorted({int(frame) for frame in (row.get("highlighted_frames") or set())}))
            current = int(row.get("current_frame")) if row.get("current_frame") is not None else None
            color = QColor(str(row.get("color") or "#16A34A"))
            key = str(row.get("key") or f"dataset_{index}")
            label = str(row.get("label") or key)
            signature_rows.append((key, label, frames, highlighted, current, color.name()))
            normalized.append({
                "key": key,
                "label": label,
                "frames": list(frames),
                "highlighted_frames": set(highlighted),
                "current_frame": current,
                "accent": color,
            })
        signature = tuple(signature_rows)
        if signature == self._timeline_signature:
            self.update()
            return
        self._timeline_signature = signature
        self._rows = normalized
        target_height = max(72, min(180, 48 + len(self._rows) * 18))
        self.setMinimumHeight(target_height)
        self.setMaximumHeight(target_height)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._hit_targets = []

        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#E5E7EB"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        if not self._rows:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignCenter, "Timeline: select MD datasets with readable frames")
            painter.end()
            return

        left = rect.left() + 112
        right = rect.right() - 8
        width = max(1, right - left)
        all_frames = sorted({
            int(frame)
            for row in self._rows
            for frame in list(row.get("frames") or [])
        })
        if not all_frames:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignCenter, "Timeline: selected datasets have no readable frames")
            painter.end()
            return

        current_candidates = [
            int(row.get("current_frame"))
            for row in self._rows
            if row.get("current_frame") is not None
        ]
        current_frame = current_candidates[0] if current_candidates else None
        full_min = min(all_frames)
        full_max = max(all_frames)
        frame_min, frame_max = self._visible_frame_range(full_min, full_max, current_frame)
        span = max(1, frame_max - frame_min)
        lane_top = rect.top() + 20
        lane_step = 16
        axis_y = min(rect.bottom() - 20, lane_top + len(self._rows) * lane_step + 4)

        def x_for_frame(frame: int) -> int:
            return int(round(left + ((int(frame) - frame_min) / span) * width))

        painter.setPen(QPen(QColor("#9CA3AF"), 2))
        painter.drawLine(left, axis_y, right, axis_y)

        tick_count = min(7, max(2, len(all_frames)))
        if frame_max == frame_min:
            tick_frames = [frame_min]
        else:
            tick_frames = [
                int(round(frame_min + (frame_max - frame_min) * idx / max(1, tick_count - 1)))
                for idx in range(tick_count)
            ]
        painter.setPen(QPen(QColor("#9CA3AF"), 1))
        for frame in tick_frames:
            x = x_for_frame(frame)
            painter.drawLine(x, axis_y - 5, x, axis_y + 5)
            painter.setPen(QColor("#6B7280"))
            painter.drawText(x - 24, axis_y + 7, 48, 12, Qt.AlignCenter, str(frame))
            painter.setPen(QPen(QColor("#9CA3AF"), 1))

        painter.setPen(QColor("#6B7280"))
        painter.drawText(
            left,
            rect.top() + 2,
            width,
            14,
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Frames {full_min}-{full_max} ({len(all_frames)})",
        )

        for row_index, row in enumerate(self._rows):
            side = str(row.get("key") or f"dataset_{row_index}")
            lane_y = lane_top + row_index * lane_step
            accent = row.get("accent") or QColor("#16A34A")
            painter.setPen(QColor("#374151"))
            painter.drawText(rect.left(), lane_y - 7, 100, 14, Qt.AlignRight | Qt.AlignVCenter, str(row.get("label") or side))
            painter.setPen(QPen(QColor("#E5E7EB"), 1))
            painter.drawLine(left, lane_y, right, lane_y)
            painter.setPen(QPen(accent, 2))
            painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue())))
            for frame in sorted(row.get("highlighted_frames") or set()):
                if frame < frame_min or frame > frame_max:
                    continue
                x = x_for_frame(frame)
                painter.drawLine(x, lane_y - 5, x, lane_y + 5)
                painter.drawEllipse(x - 3, lane_y - 3, 6, 6)
                self._hit_targets.append(
                    {
                        "side": side,
                        "frame": int(frame),
                        "x": int(x),
                        "y": int(lane_y),
                    }
                )

            row_current = row.get("current_frame")
            if row_current is not None and frame_min <= int(row_current) <= frame_max:
                x = x_for_frame(int(row_current))
                painter.setPen(QPen(QColor("#2563EB"), 2))
                painter.setBrush(QBrush(QColor("#2563EB")))
                painter.drawEllipse(x - 4, lane_y - 4, 8, 8)

        if current_frame is not None and frame_min <= int(current_frame) <= frame_max:
            x = x_for_frame(int(current_frame))
            painter.setPen(QPen(QColor("#1D4ED8"), 2))
            painter.setBrush(QBrush(QColor("#2563EB")))
            painter.drawLine(x, lane_top - 6, x, axis_y + 4)

        painter.end()

    def _visible_frame_range(self, frame_min: int, frame_max: int, current_frame) -> tuple[int, int]:
        if self._zoom <= 1.01 or frame_max <= frame_min:
            return int(frame_min), int(frame_max)
        full_span = max(1, int(frame_max) - int(frame_min))
        visible_span = max(1, int(round(full_span / self._zoom)))
        center = int(current_frame) if current_frame is not None else int(round((frame_min + frame_max) / 2))
        start = center - visible_span // 2
        end = start + visible_span
        if start < frame_min:
            start = int(frame_min)
            end = min(int(frame_max), start + visible_span)
        if end > frame_max:
            end = int(frame_max)
            start = max(int(frame_min), end - visible_span)
        return int(start), int(end)

    def mousePressEvent(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x = float(pos.x())
        y = float(pos.y())
        best = None
        best_dist2 = 999999.0
        for target in self._hit_targets:
            dx = x - float(target.get("x", 0))
            dy = y - float(target.get("y", 0))
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best = target
                best_dist2 = dist2
        if best is not None and best_dist2 <= 144.0:
            self.frame_clicked.emit(str(best.get("side") or "top"), int(best.get("frame", 0)))
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        current = self.zoom_percent()
        direction = 1 if delta > 0 else -1
        # Multiplicative zoom keeps wheel control responsive at both low and high zoom.
        factor = 1.25 if direction > 0 else 0.80
        next_value = int(round(current * factor))
        next_value = max(100, min(5000, next_value))
        if next_value != current:
            self.set_zoom_percent(next_value)
            self.zoom_changed.emit(next_value)
        event.accept()


class ChartModeStack(QWidget):
    """Segmented Profile/Statistics switch that preserves both chart widgets."""

    mode_changed = Signal(str)

    def __init__(self, profile_chart: ProfileChart, statistics_chart: StatisticsChart, parent=None):
        super().__init__(parent)
        self._profile_chart = profile_chart
        self._statistics_chart = statistics_chart

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        segment = QWidget()
        segment.setObjectName("chartModeSegment")
        segment.setStyleSheet(
            "QWidget#chartModeSegment{background:#f3f6fa;border:1px solid #d9e0ea;border-radius:6px;}"
            "QWidget#chartModeSegment QPushButton{border:none;background:transparent;color:#606266;"
            "padding:3px 14px;min-width:82px;font-size:11px;border-radius:5px;}"
            "QWidget#chartModeSegment QPushButton:checked{background:#409eff;color:#ffffff;font-weight:600;}"
        )
        segment_layout = QHBoxLayout(segment)
        segment_layout.setContentsMargins(3, 3, 3, 3)
        segment_layout.setSpacing(0)
        self._profile_btn = QPushButton("Profile")
        self._statistics_btn = QPushButton("Statistics")
        for button in (self._profile_btn, self._statistics_btn):
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setFixedHeight(26)
            segment_layout.addWidget(button)
        compact_controls = profile_chart.compact_control_widgets()
        if compact_controls:
            segment_layout.addSpacing(8)
            for control in compact_controls:
                segment_layout.addWidget(control)
        self._profile_btn.clicked.connect(lambda: self.set_mode("profile"))
        self._statistics_btn.clicked.connect(lambda: self.set_mode("statistics"))

        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.addWidget(segment, 0)
        selector_row.addSpacing(8)
        # Keep the active chart title on the same visual row as the mode
        # buttons.  These are the charts' real title labels (rather than
        # copies), so their metric-dependent text continues to update in one
        # place when the display configuration changes.
        self._profile_title = getattr(self._profile_chart, "_title", None)
        self._statistics_title = getattr(self._statistics_chart, "_title", None)
        if self._profile_title is not None:
            selector_row.addWidget(self._profile_title, 0)
        if self._statistics_title is not None:
            selector_row.addWidget(self._statistics_title, 0)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._profile_chart)
        self._stack.addWidget(self._statistics_chart)
        layout.addWidget(self._stack, 1)
        self.set_mode("profile", emit_signal=False)

    def mode(self) -> str:
        return "statistics" if self._stack.currentWidget() is self._statistics_chart else "profile"

    def set_mode(self, mode: str, *, emit_signal: bool = True):
        normalized = "statistics" if str(mode or "").strip().lower() == "statistics" else "profile"
        previous = self.mode()
        target = self._statistics_chart if normalized == "statistics" else self._profile_chart
        self._stack.setCurrentWidget(target)
        if self._profile_title is not None:
            self._profile_title.setVisible(normalized == "profile")
        if self._statistics_title is not None:
            self._statistics_title.setVisible(normalized == "statistics")
        for button, checked in (
            (self._profile_btn, normalized == "profile"),
            (self._statistics_btn, normalized == "statistics"),
        ):
            was_blocked = button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(was_blocked)
        if emit_signal and previous != normalized:
            self.mode_changed.emit(normalized)


class ChartWorkspaceDialog(QDialog):
    """Detached, larger chart workspace linked to the main window state."""

    closed = Signal()
    path_filter_apply_requested = Signal(dict)
    path_filter_revert_requested = Signal(list)

    def __init__(
        self,
        parent=None,
        *,
        shared_chart_container=None,
        profile_chart=None,
        statistics_chart=None,
        combination_region_chart=None,
        chart_mode_stack=None,
    ):
        super().__init__(parent)
        self.db = getattr(parent, "db", None)
        self._uses_shared_charts = shared_chart_container is not None
        self._shared_chart_container = shared_chart_container
        self.setWindowTitle("Chart Workspace")
        self.setModal(False)
        self.setMinimumSize(1180, 920)
        self.resize(1260, 980)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("Chart Workspace")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        subtitle = QLabel("Expanded, synchronized view for profile and statistics charts")
        subtitle.setStyleSheet("color: #909399; font-size: 11px;")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch()
        self._close_btn = QPushButton("Close")
        self._close_btn.setFixedHeight(28)
        self._close_btn.clicked.connect(self.close)
        header.addWidget(self._close_btn)
        layout.addLayout(header)

        self._content_splitter = QSplitter(Qt.Horizontal)
        self._content_splitter.setChildrenCollapsible(False)

        chart_column = QWidget()
        chart_layout = QVBoxLayout(chart_column)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        self._shared_chart_layout = chart_layout
        if self._uses_shared_charts:
            self._splitter = None
            self._profile_chart = profile_chart
            self._statistics_chart = statistics_chart
            self._combination_region_chart = combination_region_chart
            self._chart_mode_stack = chart_mode_stack
            chart_layout.addWidget(self._shared_chart_container, 1)
        else:
            self._splitter = QSplitter(Qt.Vertical)
            self._splitter.setChildrenCollapsible(False)
            self._profile_chart = ProfileChart(workspace_mode=True)
            self._statistics_chart = StatisticsChart(show_bottleneck_points=True)
            self._combination_region_chart = ResidueCombinationRegionChart()
            self._statistics_chart.set_display_config(*self._profile_chart.display_config())
            self._profile_chart.dataset_filter_changed.connect(self._statistics_chart.set_dataset_filter)
            self._profile_chart.dataset_filter_changed.connect(self._combination_region_chart.set_dataset_filter)
            self._profile_chart.residue_stats_updated.connect(self._update_residue_summary_panel)
            self._chart_mode_stack = ChartModeStack(
                self._profile_chart,
                self._statistics_chart,
            )
            self._splitter.addWidget(self._chart_mode_stack)
            self._splitter.addWidget(self._combination_region_chart)
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 1)
            self._splitter.setSizes([425, 425])
            chart_layout.addWidget(self._splitter, 1)

        side_panel = QWidget()
        side_panel.setMinimumWidth(320)
        side_panel.setMaximumWidth(460)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        side_title = QLabel("Residue Summary")
        side_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #1f2937;")
        side_layout.addWidget(side_title)

        side_hint = QLabel(
            "All visible paths are summarized by default. Use `Residue Stats` mode "
            "and drag a selection box to summarize only selected chart points."
        )
        side_hint.setWordWrap(True)
        side_hint.setStyleSheet("color: #909399; font-size: 11px;")
        side_layout.addWidget(side_hint)

        summary_toolbar = QHBoxLayout()
        summary_toolbar.setContentsMargins(0, 0, 0, 0)
        summary_toolbar.setSpacing(6)
        self._residue_summary_view_label = QLabel("View")
        summary_toolbar.addWidget(self._residue_summary_view_label)
        self._residue_summary_mode_combo = QComboBox()
        self._residue_summary_mode_combo.addItem("Path Residues", "path")
        self._residue_summary_mode_combo.addItem("Bottleneck", "bottleneck")
        self._residue_summary_mode_combo.currentIndexChanged.connect(self._refresh_residue_summary_table)
        summary_toolbar.addWidget(self._residue_summary_mode_combo)
        self._residue_summary_heatmap_btn = QPushButton("Compare")
        self._residue_summary_heatmap_btn.setFixedHeight(28)
        self._residue_summary_heatmap_btn.clicked.connect(self._open_residue_summary_heatmap)
        summary_toolbar.addWidget(self._residue_summary_heatmap_btn)
        summary_toolbar.addStretch()
        side_layout.addLayout(summary_toolbar)

        self._residue_summary_scroll = QScrollArea()
        self._residue_summary_scroll.setWidgetResizable(True)
        self._residue_summary_scroll.setFrameShape(QScrollArea.NoFrame)
        self._residue_summary_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._residue_summary_scroll.viewport().installEventFilter(self)
        self._residue_summary_container = QWidget()
        self._residue_summary_layout = QVBoxLayout(self._residue_summary_container)
        self._residue_summary_layout.setContentsMargins(0, 0, 0, 0)
        self._residue_summary_layout.setSpacing(10)
        self._residue_summary_layout.addStretch()
        self._residue_summary_scroll.setWidget(self._residue_summary_container)
        side_layout.addWidget(self._residue_summary_scroll, 1)
        self._residue_summary_rows: list[dict] = []
        self._residue_summary_sort_state: dict[str, tuple[str, bool]] = {}
        self._residue_summary_sections: dict[str, QFrame] = {}
        self._residue_summary_tables: dict[str, QTableWidget] = {}
        self._residue_summary_heatmap_dialog = None

        filter_frame = QFrame()
        filter_frame.setStyleSheet("QFrame{border:1px solid #e5ecf6;border-radius:10px;background:#ffffff;}")
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setContentsMargins(12, 12, 12, 12)
        filter_layout.setSpacing(8)

        filter_title = QLabel("Path Filter")
        filter_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #1f2937;")
        filter_layout.addWidget(filter_title)

        self._chart_selection_info = QLabel("No chart-path selection.")
        self._chart_selection_info.setWordWrap(True)
        self._chart_selection_info.setStyleSheet("color: #6b7280; font-size: 11px;")
        filter_layout.addWidget(self._chart_selection_info)

        hydro_row = QHBoxLayout()
        hydro_row.setSpacing(6)
        hydro_row.addWidget(QLabel("Hydro"))
        self._hydro_min_edit = QLineEdit()
        self._hydro_min_edit.setPlaceholderText("min")
        self._hydro_max_edit = QLineEdit()
        self._hydro_max_edit.setPlaceholderText("max")
        self._hydro_min_edit.textChanged.connect(self._update_filtered_selection_preview)
        self._hydro_max_edit.textChanged.connect(self._update_filtered_selection_preview)
        hydro_row.addWidget(self._hydro_min_edit)
        hydro_row.addWidget(self._hydro_max_edit)
        filter_layout.addLayout(hydro_row)

        frame_row = QHBoxLayout()
        frame_row.setSpacing(6)
        frame_row.addWidget(QLabel("Frame"))
        self._frame_min_edit = QLineEdit()
        self._frame_min_edit.setPlaceholderText("min")
        self._frame_max_edit = QLineEdit()
        self._frame_max_edit.setPlaceholderText("max")
        self._frame_min_edit.textChanged.connect(self._update_filtered_selection_preview)
        self._frame_max_edit.textChanged.connect(self._update_filtered_selection_preview)
        frame_row.addWidget(self._frame_min_edit)
        frame_row.addWidget(self._frame_max_edit)
        filter_layout.addLayout(frame_row)

        self._filtered_selection_info = QLabel("Filtered result: 0 paths")
        self._filtered_selection_info.setStyleSheet("color: #6b7280; font-size: 11px;")
        filter_layout.addWidget(self._filtered_selection_info)

        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(6)
        self._apply_filter_btn = QPushButton("Select Filtered")
        self._apply_filter_btn.setFixedHeight(28)
        self._apply_filter_btn.clicked.connect(self._emit_path_filter_request)
        filter_actions.addWidget(self._apply_filter_btn)
        self._revert_filter_btn = QPushButton("Revert")
        self._revert_filter_btn.setFixedHeight(28)
        self._revert_filter_btn.setEnabled(False)
        self._revert_filter_btn.clicked.connect(self._emit_revert_request)
        filter_actions.addWidget(self._revert_filter_btn)
        self._clear_filter_btn = QPushButton("Clear")
        self._clear_filter_btn.setFixedHeight(28)
        self._clear_filter_btn.clicked.connect(self._clear_path_filter_inputs)
        filter_actions.addWidget(self._clear_filter_btn)
        filter_actions.addStretch()
        filter_layout.addLayout(filter_actions)

        side_layout.addWidget(filter_frame)

        self._content_splitter.addWidget(chart_column)
        self._content_splitter.addWidget(side_panel)
        self._content_splitter.setStretchFactor(0, 5)
        self._content_splitter.setStretchFactor(1, 2)
        if self._uses_shared_charts:
            side_panel.setVisible(False)
            self._content_splitter.setSizes([1260, 0])
        else:
            self._content_splitter.setSizes([920, 340])
        layout.addWidget(self._content_splitter, 1)
        self._update_residue_summary_panel(None, False)
        self._selected_chart_path_ids: list[int] = []
        self._selection_history: list[list[int]] = []

    @property
    def profile_chart(self) -> ProfileChart:
        return self._profile_chart

    @property
    def uses_shared_charts(self) -> bool:
        return bool(self._uses_shared_charts)

    def attach_shared_charts(self) -> None:
        """Move the already-rendered main chart surface into this window."""
        if not self._uses_shared_charts or self._shared_chart_container is None:
            return
        if self._shared_chart_layout.indexOf(self._shared_chart_container) < 0:
            self._shared_chart_layout.addWidget(self._shared_chart_container, 1)
        self._shared_chart_container.setVisible(True)

    def release_shared_charts(self) -> None:
        """Detach the shared surface without clearing or rebuilding its plots."""
        if not self._uses_shared_charts or self._shared_chart_container is None:
            return
        self._shared_chart_layout.removeWidget(self._shared_chart_container)

    @property
    def statistics_chart(self) -> StatisticsChart:
        return self._statistics_chart

    def set_display_config(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
    ):
        self._profile_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
            emit_signal=False,
        )
        self._statistics_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
        )

    def set_profiles(self, profiles, path_color_map=None):
        self._profile_chart.set_profiles(profiles, replot=False)
        if path_color_map is not None:
            self._profile_chart.set_path_color_map(path_color_map, replot=False)
        selected_prefixes = self._profile_chart.selected_dataset_prefixes()
        self._statistics_chart.set_dataset_filter(selected_prefixes, replot=False)
        self._statistics_chart.set_profiles(profiles, replot=False)
        if path_color_map is not None:
            self._statistics_chart.set_path_color_map(path_color_map, replot=False)
        self._combination_region_chart.set_profiles(profiles, replot=False)
        self._combination_region_chart.set_dataset_filter(selected_prefixes, replot=False)
        self._profile_chart.refresh()
        self._statistics_chart.refresh()
        self._combination_region_chart.refresh()

    def set_path_color_map(self, path_color_map):
        self._profile_chart.set_path_color_map(path_color_map)
        self._statistics_chart.set_path_color_map(path_color_map)

    def set_appearance(self, alpha_scale: float, depth_scale: float):
        self._profile_chart.set_appearance(
            alpha_scale,
            depth_scale,
            emit_signal=False,
        )

    def appearance(self):
        return self._profile_chart.appearance()

    def chart_mode(self) -> str:
        return self._chart_mode_stack.mode()

    def set_chart_mode(self, mode: str):
        self._chart_mode_stack.set_mode(mode, emit_signal=False)

    def clear_charts(self):
        self._profile_chart.clear()
        self._statistics_chart.clear()
        self._combination_region_chart.clear()
        self._update_residue_summary_panel(None, False)
        self.set_chart_selected_paths([])

    def show_and_focus(self):
        self.attach_shared_charts()
        self.show()
        self.raise_()
        self.activateWindow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_residue_summary_sections()
        QTimer.singleShot(0, self._resize_residue_summary_sections)
        if not self._uses_shared_charts:
            self._refresh_render_surfaces()

    def eventFilter(self, watched, event):
        summary_viewport = getattr(self, "_residue_summary_scroll", None)
        if (
            summary_viewport is not None
            and watched is summary_viewport.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            QTimer.singleShot(0, self._resize_residue_summary_sections)
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and not self._uses_shared_charts:
            QTimer.singleShot(0, self._refresh_render_surfaces)

    def _refresh_render_surfaces(self):
        for widget in (
            getattr(self, "_profile_chart", None),
            getattr(self, "_statistics_chart", None),
            getattr(self, "_combination_region_chart", None),
            getattr(self, "_content_splitter", None),
            getattr(self, "_splitter", None),
        ):
            if widget is None:
                continue
            try:
                widget.update()
                widget.repaint()
            except Exception:
                pass

    @staticmethod
    def _blend_with_white(base: tuple[int, int, int], strength: float) -> QColor:
        strength = max(0.0, min(1.0, strength))
        red = int(round(255 + (base[0] - 255) * strength))
        green = int(round(255 + (base[1] - 255) * strength))
        blue = int(round(255 + (base[2] - 255) * strength))
        return QColor(red, green, blue)

    @classmethod
    def _rank_change_color(cls, rank_delta: int, max_abs_delta: int) -> QColor:
        if max_abs_delta <= 0 or rank_delta == 0:
            return QColor("#FFFFFF")
        strength = abs(rank_delta) / max_abs_delta
        strength = 0.28 + strength * 0.68
        if rank_delta > 0:
            return cls._blend_with_white((211, 76, 76), strength)
        return cls._blend_with_white((70, 116, 196), strength)

    @classmethod
    def _new_rank_color(cls) -> QColor:
        return cls._blend_with_white((103, 194, 58), 0.72)

    @classmethod
    def _metric_heatmap_color(cls, value: float, max_value: float) -> QColor:
        if max_value <= 1e-12 or value <= 1e-12:
            return QColor("#FFFFFF")
        strength = value / max_value
        strength = 0.20 + strength * 0.72
        return cls._blend_with_white((64, 158, 255), strength)

    def _sort_residue_summary_rows(self, rows, sort_key: str, descending: bool):
        def sort_value(row):
            if sort_key == "residue_id":
                return int(row.get("localResidueId", row.get("residueId", 0)))
            if sort_key == "path_count":
                return int(row.get("pathCount", 0))
            if sort_key == "avg_radius":
                value = row.get("avgRadius")
                return float(value) if value is not None else float("-inf")
            if sort_key == "change":
                return str(row.get("rank_change_text", "") or "")
            return int(row.get("localResidueId", row.get("residueId", 0)))

        sorted_rows = list(rows)
        sorted_rows.sort(
            key=lambda row: (
                sort_value(row),
                int(row.get("localResidueId", row.get("residueId", 0))),
                int(row.get("pathCount", 0)),
            ),
            reverse=descending,
        )
        return sorted_rows

    def _local_residue_id(self, residue_id: int) -> int:
        try:
            residue_id = int(residue_id)
        except (TypeError, ValueError):
            return 0
        if self.db is not None and hasattr(self.db, "_decode_residue_id"):
            dataset, local_id = self.db._decode_residue_id(residue_id)
            if dataset is not None and local_id is not None:
                return int(local_id)
        return residue_id

    def _ordered_summary_dataset_prefixes(self, grouped_rows: dict[str, list[dict]]) -> list[str]:
        prefixes = set(grouped_rows.keys())
        ordered: list[str] = []
        if self.db is not None and hasattr(self.db, "list_datasets"):
            for dataset in self.db.list_datasets():
                prefix = str(dataset.get("prefix", "") or "")
                if prefix in prefixes and prefix not in ordered:
                    ordered.append(prefix)
        for prefix in sorted(prefixes):
            if prefix not in ordered:
                ordered.append(prefix)
        return ordered

    def _annotate_residue_summary_ranks(self, grouped_rows: dict[str, list[dict]], ordered_prefixes: list[str]):
        rank_maps: dict[str, dict[int, int]] = {}
        max_rank_delta = 0
        for dataset_prefix in ordered_prefixes:
            rank_rows = sorted(
                grouped_rows.get(dataset_prefix, []),
                key=lambda row: (-int(row.get("pathCount", 0)), int(row.get("localResidueId", row.get("residueId", 0)))),
            )
            dataset_rank_map: dict[int, int] = {}
            last_path_count = None
            current_rank = 0
            for index, row in enumerate(rank_rows, start=1):
                path_count = int(row.get("pathCount", 0))
                if last_path_count is None or path_count != last_path_count:
                    current_rank = index
                    last_path_count = path_count
                dataset_rank_map[int(row.get("localResidueId", row.get("residueId", 0)))] = current_rank
            rank_maps[dataset_prefix] = dataset_rank_map

        compare_targets: dict[str, str] = {}
        if len(ordered_prefixes) == 2:
            left, right = ordered_prefixes
            compare_targets[left] = right
            compare_targets[right] = left

        for dataset_prefix in ordered_prefixes:
            target_prefix = compare_targets.get(dataset_prefix, "")
            target_ranks = rank_maps.get(target_prefix, {})
            for row in grouped_rows.get(dataset_prefix, []):
                local_residue_id = int(row.get("localResidueId", row.get("residueId", 0)))
                rank = int(rank_maps.get(dataset_prefix, {}).get(local_residue_id, 0))
                row["rank_change_delta"] = 0
                row["rank_change_text"] = ""
                row["rank_change_kind"] = "none"
                if not target_prefix:
                    continue
                other_rank = target_ranks.get(local_residue_id)
                if other_rank is None:
                    row["rank_change_text"] = "NEW"
                    row["rank_change_kind"] = "new"
                elif rank < other_rank:
                    delta = other_rank - rank
                    row["rank_change_delta"] = delta
                    row["rank_change_text"] = f"UP{delta}"
                    row["rank_change_kind"] = "up"
                    max_rank_delta = max(max_rank_delta, delta)
                elif rank > other_rank:
                    delta = rank - other_rank
                    row["rank_change_delta"] = -delta
                    row["rank_change_text"] = f"DOWN{delta}"
                    row["rank_change_kind"] = "down"
                    max_rank_delta = max(max_rank_delta, delta)
                else:
                    row["rank_change_text"] = "SAME"
                    row["rank_change_kind"] = "same"

        for dataset_prefix in ordered_prefixes:
            for row in grouped_rows.get(dataset_prefix, []):
                change_kind = str(row.get("rank_change_kind", "none") or "none")
                if change_kind == "new":
                    row["rank_change_color"] = self._new_rank_color()
                else:
                    row["rank_change_color"] = self._rank_change_color(
                        int(row.get("rank_change_delta", 0)),
                        max_rank_delta,
                    )

    def _create_residue_summary_table(self, dataset_prefix: str, selected_kind: str | None = None) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.setWordWrap(False)
        table.setStyleSheet(
            "QTableWidget {"
            "background: #fbfcfe; border: 1px solid #e5ecf6; border-radius: 10px; "
            "color: #303133; font-size: 11px;"
            "}"
        )
        table.setHorizontalHeaderLabels(["Residue", "Paths", "Mean Radius", "Change"])
        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(
            lambda section, prefix=dataset_prefix, kind=selected_kind: self._on_residue_summary_header_clicked(prefix, section, kind)
        )
        return table

    def _populate_residue_summary_table(self, table: QTableWidget, rows):
        table.clearContents()
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            rank_change = str(row.get("rank_change_text", "") or "")
            values = (
                str(int(row.get("localResidueId", row.get("residueId", 0)))),
                str(int(row.get("pathCount", 0))),
                f"{float(row.get('avgRadius', 0.0)):.2f} Å" if row.get("avgRadius") is not None else "-",
                rank_change or "-",
            )
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col_idx == 3:
                    background = row.get("rank_change_color")
                    if isinstance(background, QColor):
                        item.setBackground(QBrush(background))
                table.setItem(row_idx, col_idx, item)
        table.resizeRowsToContents()
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _resize_residue_summary_sections(self):
        section_count = len(self._residue_summary_sections)
        if section_count <= 0:
            return
        viewport_height = max(0, self._residue_summary_scroll.viewport().height())
        spacing = self._residue_summary_layout.spacing()
        visible_count = min(2, section_count)
        total_spacing = max(0, visible_count - 1) * spacing
        target_height = max(220, (viewport_height - total_spacing) // visible_count)
        for section in self._residue_summary_sections.values():
            section.setMinimumHeight(target_height)
            section.setMaximumHeight(target_height)

    def _current_residue_summary_grouped_rows(self) -> tuple[list[str], dict[str, list[dict]]]:
        return self._residue_summary_grouped_rows_for_kind(
            str(self._residue_summary_mode_combo.currentData() or "path")
        )

    def _residue_summary_dataset_prefixes(self) -> list[str]:
        prefixes = {
            str(row.get("dataset_prefix") or "Dataset")
            for row in self._residue_summary_rows
        }
        ordered: list[str] = []
        if self.db is not None and hasattr(self.db, "list_datasets"):
            for dataset in self.db.list_datasets():
                prefix = str(dataset.get("prefix", "") or "")
                if prefix in prefixes and prefix not in ordered:
                    ordered.append(prefix)
        for prefix in sorted(prefixes):
            if prefix not in ordered:
                ordered.append(prefix)
        return ordered

    def _residue_summary_grouped_rows_for_kind(self, selected_kind: str) -> tuple[list[str], dict[str, list[dict]]]:
        grouped_rows: dict[str, list[dict]] = {}
        for row in self._residue_summary_rows:
            if str(row.get("summary_kind") or "path") != selected_kind:
                continue
            copied = dict(row)
            copied["localResidueId"] = self._local_residue_id(copied.get("residueId", 0))
            dataset_prefix = str(copied.get("dataset_prefix") or "Dataset")
            grouped_rows.setdefault(dataset_prefix, []).append(copied)
        ordered_prefixes = self._ordered_summary_dataset_prefixes(grouped_rows)
        self._annotate_residue_summary_ranks(grouped_rows, ordered_prefixes)
        return ordered_prefixes, grouped_rows

    def _build_residue_summary_heatmap_payload(self) -> dict:
        ordered_prefixes, grouped_rows = self._current_residue_summary_grouped_rows()
        selected_kind = str(self._residue_summary_mode_combo.currentData() or "path")
        if not ordered_prefixes:
            return {"datasets": [], "rows": [], "kind": selected_kind}

        dataset_maps = {
            prefix: {
                int(row.get("localResidueId", row.get("residueId", 0))): row
                for row in grouped_rows.get(prefix, [])
            }
            for prefix in ordered_prefixes
        }
        compare_prefixes = ordered_prefixes[:2]
        if len(compare_prefixes) < 2:
            compare_prefixes = ordered_prefixes[:1]
        axis_residues = sorted(
            {
                residue_id
                for prefix in compare_prefixes
                for residue_id in dataset_maps.get(prefix, {}).keys()
            }
        )
        if not axis_residues:
            return {"datasets": [], "rows": [], "kind": selected_kind}
        dataset_totals = {
            prefix: max(
                [
                    int(row.get("totalPaths", 0) or 0)
                    for row in grouped_rows.get(prefix, [])
                ]
                or [0]
            )
            for prefix in ordered_prefixes
        }

        rows = []
        base_prefix = compare_prefixes[0]
        compare_prefix = compare_prefixes[-1]
        base_map = dataset_maps.get(base_prefix, {})
        compare_map = dataset_maps.get(compare_prefix, {})
        rank_maps = {}
        for prefix in compare_prefixes:
            ranked_rows = sorted(
                dataset_maps.get(prefix, {}).values(),
                key=lambda row: (-int(row.get("pathCount", 0) or 0), int(row.get("localResidueId", row.get("residueId", 0)))),
            )
            ranks = {}
            last_count = None
            current_rank = 0
            for index, row in enumerate(ranked_rows, start=1):
                path_count = int(row.get("pathCount", 0) or 0)
                if last_count is None or path_count != last_count:
                    current_rank = index
                    last_count = path_count
                ranks[int(row.get("localResidueId", row.get("residueId", 0)))] = current_rank
            rank_maps[prefix] = ranks
        base_total = max(0, int(dataset_totals.get(base_prefix, 0) or 0))
        compare_total = max(0, int(dataset_totals.get(compare_prefix, 0) or 0))
        delta_rows = []
        for residue_id in axis_residues:
            base_row = base_map.get(residue_id)
            compare_row = compare_map.get(residue_id)
            base_count = int(base_row.get("pathCount", 0) or 0) if base_row else 0
            compare_count = int(compare_row.get("pathCount", 0) or 0) if compare_row else 0
            base_percent = (base_count / base_total * 100.0) if base_total > 0 else 0.0
            compare_percent = (compare_count / compare_total * 100.0) if compare_total > 0 else 0.0
            delta_count = compare_count - base_count
            delta_percent = compare_percent - base_percent
            base_rank = rank_maps.get(base_prefix, {}).get(residue_id)
            compare_rank = rank_maps.get(compare_prefix, {}).get(residue_id)
            if base_row is None and compare_row is not None:
                status = "new"
            elif base_rank is not None and compare_rank is not None and compare_rank < base_rank:
                status = "up"
            elif base_rank is not None and compare_rank is not None and compare_rank > base_rank:
                status = "down"
            elif base_row is not None and compare_row is None:
                status = "down"
            else:
                status = "same"
            delta_rows.append(
                {
                    "residue_id": residue_id,
                    "base_count": base_count,
                    "base_percent": base_percent,
                    "compare_count": compare_count,
                    "compare_percent": compare_percent,
                    "delta_count": delta_count,
                    "delta_percent": delta_percent,
                    "status": status,
                    "base_rank": base_rank,
                    "compare_rank": compare_rank,
                }
            )
        delta_rows.sort(
            key=lambda row: (
                0 if row["status"] == "new" else 1,
                -abs(float(row["delta_percent"])),
                -abs(int(row["delta_count"])),
                int(row["residue_id"]),
            )
        )
        for row_residue_id in axis_residues:
            cells = {}
            for col_residue_id in axis_residues:
                if row_residue_id != col_residue_id:
                    cells[col_residue_id] = {"blank": True}
                    continue
                base_row = base_map.get(row_residue_id)
                compare_row = compare_map.get(col_residue_id)
                base_count = int(base_row.get("pathCount", 0) or 0) if base_row else 0
                compare_count = int(compare_row.get("pathCount", 0) or 0) if compare_row else 0
                percent = (compare_count / compare_total * 100.0) if compare_total > 0 else 0.0
                delta = compare_count - base_count
                if base_row is None and compare_row is not None:
                    change_kind = "new"
                elif delta > 0:
                    change_kind = "up"
                else:
                    change_kind = "down"
                cells[col_residue_id] = {
                    "count": compare_count,
                    "percent": percent,
                    "change_kind": change_kind,
                    "change_delta": delta,
                    "base_count": base_count,
                    "compare_count": compare_count,
                    "total_paths": compare_total,
                }
            rows.append({"residue_id": row_residue_id, "cells": cells})
        return {
            "datasets": compare_prefixes,
            "axis_residues": axis_residues,
            "delta_rows": delta_rows,
            "rows": rows,
            "kind": selected_kind,
        }

    def _refresh_residue_summary_table(self):
        while self._residue_summary_layout.count():
            item = self._residue_summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._residue_summary_sections = {}
        self._residue_summary_tables = {}
        dataset_prefixes = self._residue_summary_dataset_prefixes()
        single_dataset_mode = len(dataset_prefixes) == 1
        self._residue_summary_view_label.setVisible(not single_dataset_mode)
        self._residue_summary_mode_combo.setVisible(not single_dataset_mode)
        self._residue_summary_heatmap_btn.setVisible(not single_dataset_mode)

        if single_dataset_mode:
            dataset_prefix = dataset_prefixes[0]
            for selected_kind, title_suffix in (("path", "Path Residues"), ("bottleneck", "Bottleneck")):
                _ordered_prefixes, grouped_rows = self._residue_summary_grouped_rows_for_kind(selected_kind)
                rows = grouped_rows.get(dataset_prefix, [])
                if not rows:
                    continue
                self._add_residue_summary_section(
                    dataset_prefix,
                    selected_kind,
                    rows,
                    title=f"{dataset_prefix} - {title_suffix}",
                )
        else:
            selected_kind = str(self._residue_summary_mode_combo.currentData() or "path")
            ordered_prefixes, grouped_rows = self._current_residue_summary_grouped_rows()
            for dataset_prefix in ordered_prefixes:
                self._add_residue_summary_section(
                    dataset_prefix,
                    selected_kind,
                    grouped_rows[dataset_prefix],
                    title=dataset_prefix,
                )

        self._residue_summary_layout.addStretch()
        self._resize_residue_summary_sections()
        QTimer.singleShot(0, self._resize_residue_summary_sections)
        self._refresh_residue_summary_heatmap_dialog()

    def _add_residue_summary_section(self, dataset_prefix: str, selected_kind: str, rows: list[dict], *, title: str):
        sort_state_key = f"{selected_kind}:{dataset_prefix}"
        if sort_state_key not in self._residue_summary_sort_state:
            self._residue_summary_sort_state[sort_state_key] = ("path_count", True)
        sort_key, descending = self._residue_summary_sort_state[sort_state_key]
        sorted_rows = self._sort_residue_summary_rows(rows, sort_key, descending)

        section = QFrame()
        section.setStyleSheet("QFrame{border:1px solid #e5ecf6;border-radius:10px;background:#ffffff;}")
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(10, 10, 10, 10)
        section_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #1f2937;")
        section_layout.addWidget(title_label)

        table = self._create_residue_summary_table(dataset_prefix, selected_kind)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._populate_residue_summary_table(table, sorted_rows)
        section_key = f"{selected_kind}:{dataset_prefix}"
        self._residue_summary_sections[section_key] = section
        self._residue_summary_tables[section_key] = table
        section_layout.addWidget(table)
        self._residue_summary_layout.addWidget(section)

    def _on_residue_summary_header_clicked(self, dataset_prefix: str, section: int, selected_kind: str | None = None):
        sort_keys = {
            0: "residue_id",
            1: "path_count",
            2: "avg_radius",
            3: "change",
        }
        next_key = sort_keys.get(int(section), "residue_id")
        selected_kind = str(selected_kind or self._residue_summary_mode_combo.currentData() or "path")
        sort_state_key = f"{selected_kind}:{dataset_prefix}"
        current_key, current_desc = self._residue_summary_sort_state.get(sort_state_key, ("residue_id", False))
        descending = not current_desc if current_key == next_key else False
        self._residue_summary_sort_state[sort_state_key] = (next_key, descending)
        self._refresh_residue_summary_table()

    def _update_residue_summary_panel(self, rows, is_active: bool):
        if is_active:
            self._residue_summary_rows = list(rows or [])
        else:
            profile_chart = getattr(self, "_profile_chart", None)
            if profile_chart is not None and hasattr(profile_chart, "all_residue_summary_rows"):
                self._residue_summary_rows = profile_chart.all_residue_summary_rows()
            else:
                self._residue_summary_rows = []
        self._refresh_residue_summary_table()

    def _open_residue_summary_heatmap(self):
        if self._residue_summary_heatmap_dialog is None:
            self._residue_summary_heatmap_dialog = _ResidueSummaryHeatmapDialog(self)
            self._residue_summary_heatmap_dialog.destroyed.connect(self._on_residue_summary_heatmap_destroyed)
        self._refresh_residue_summary_heatmap_dialog()
        self._residue_summary_heatmap_dialog.show()
        self._residue_summary_heatmap_dialog.raise_()
        self._residue_summary_heatmap_dialog.activateWindow()

    def _refresh_residue_summary_heatmap_dialog(self):
        if self._residue_summary_heatmap_dialog is None:
            return
        self._residue_summary_heatmap_dialog.refresh(self._build_residue_summary_heatmap_payload())

    def _on_residue_summary_heatmap_destroyed(self, *_args):
        self._residue_summary_heatmap_dialog = None

    def set_chart_selected_paths(self, path_ids):
        self._selected_chart_path_ids = [int(path_id) for path_id in (path_ids or [])]
        if self._selected_chart_path_ids:
            self._chart_selection_info.setText(f"Current box selection: {len(self._selected_chart_path_ids)} paths")
        else:
            self._chart_selection_info.setText("No chart-path selection.")
        self._update_filtered_selection_preview()


    def reset_filter_history(self):
        self._selection_history = []
        self._revert_filter_btn.setEnabled(False)

    def _parse_optional_float(self, text: str) -> Optional[float]:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_optional_int(self, text: str) -> Optional[int]:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _current_filter_payload(self) -> dict:
        return {
            "hydrophobicity_min": self._parse_optional_float(self._hydro_min_edit.text()),
            "hydrophobicity_max": self._parse_optional_float(self._hydro_max_edit.text()),
            "frame_min": self._parse_optional_int(self._frame_min_edit.text()),
            "frame_max": self._parse_optional_int(self._frame_max_edit.text()),
        }

    def _update_filtered_selection_preview(self):
        if not self._selected_chart_path_ids:
            self._filtered_selection_info.setText("Filtered result: 0 paths")
            return
        filtered_ids = self._profile_chart.filter_path_ids(
            self._selected_chart_path_ids,
            **self._current_filter_payload(),
        )
        self._filtered_selection_info.setText(f"Filtered result: {len(filtered_ids)} paths")

    def _emit_path_filter_request(self):
        if self._selected_chart_path_ids:
            current = [int(path_id) for path_id in self._selected_chart_path_ids]
            if not self._selection_history or self._selection_history[-1] != current:
                self._selection_history.append(current)
        self._revert_filter_btn.setEnabled(bool(self._selection_history))
        payload = self._current_filter_payload()
        payload["path_ids"] = self._profile_chart.filter_path_ids(
            self._selected_chart_path_ids,
            **payload,
        )
        self.path_filter_apply_requested.emit(payload)
        self._filtered_selection_info.setText(f"Filtered result: {len(payload['path_ids'])} paths")

    def _emit_revert_request(self):
        if not self._selection_history:
            return
        previous = self._selection_history.pop()
        self._revert_filter_btn.setEnabled(bool(self._selection_history))
        self.path_filter_revert_requested.emit(previous)

    def _clear_path_filter_inputs(self):
        for edit in (
            self._hydro_min_edit,
            self._hydro_max_edit,
            self._frame_min_edit,
            self._frame_max_edit,
        ):
            edit.clear()
        self._update_filtered_selection_preview()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, db_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Tuner")
        self.setMinimumSize(1200, 990)
        self.resize(1400, 990)

        # Core objects
        initial_paths = [db_path] if db_path else []
        self.db = MultiDatasetDatabase(initial_paths)
        self.db.connect()
        self.state = AppState()
        self._protein_path: Optional[str] = None
        self._pse_path: Optional[str] = None
        self._pse_extracted_pdb_path: Optional[str] = None
        self._protein_observer_path: Optional[str] = None
        self._observer_pending_payload: dict = {}
        self._observer_change_events: list[dict] = []
        self._observer_path_focus_interval: tuple[float, float, float] | None = None
        self._observer_alignment_metadata_cache: dict[tuple[str, str], dict | None] = {}
        self._observer_frame_transform_cache: dict[tuple[str, str, int], dict | None] = {}
        self._observer_sequence_source_cache: dict[str, tuple[tuple[str, str], ...]] = {}
        self._observer_dataset_layout_signature: tuple = ()
        self._observer_load_generation = 0
        self._observer_load_queue: list[str] = []
        self._observer_loading_dataset_key = ""
        self._observer_compare_dataset_keys: list[str] = []
        self._protein_viewer_panel = None
        self._protein_viewer_panel_top = None
        self._protein_viewer_panel_bottom = None
        self._protein_observer_slots_by_dataset: dict[str, dict] = {}
        self._chart_profiles = []
        # Chart profiles are immutable for the lifetime of an opened database.
        # Keep a small LRU so repeatedly revisiting a selection does not hit the
        # database again.  Rendering is still refreshed when chart settings
        # change; this cache only avoids duplicate profile materialization.
        self._chart_profile_cache: OrderedDict[tuple, list] = OrderedDict()
        self._chart_profile_cache_limit = 8
        self._chart_profile_cache_max_paths = 6_000
        self._tunnel_properties_render_signature = None
        self._chart_workspace: Optional[ChartWorkspaceDialog] = None
        self._charts_detached = False
        self._attached_chart_sizes = [560, 280]
        # Start in the raw exported-coordinate view.  Pipeline/current
        # coordinates remain available through the viewer toggle, but an
        # imported dataset should immediately present its Original Paths.
        self._path_coordinate_mode = DEFAULT_PATH_COORDINATE_MODE
        self._path_coordinate_load_busy = False
        self._path_coordinate_reload_pending = False
        self._last_3d_load_signature = None
        self._active_3d_load_signature = None
        self._path_change_overlay_active = False
        self._path_change_metric_cache_signature: tuple = ()
        self._path_change_metric_cache: dict[int, float] = {}
        self._exit_delta_overlay_active = False
        self._exit_delta_metric_cache_signature: tuple = ()
        self._exit_delta_metric_cache: dict[int, float] = {}
        self._path_delta_excluded_path_ids: set[int] = set()
        self._path_delta_filter_threshold: float = 0.0
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids: set[int] = set()
        self._contained_match_last_compare_summary = ""
        self._suppress_next_path_preview = False
        self._3d_load_request_id = 0
        self._residue_combo_task_request_id = 0
        self._residue_combo_task_thread: QThread | None = None
        self._residue_combo_task_worker: ResidueCombinationTaskWorker | None = None
        self._dataset_path_cache = {}
        self._dataset_residue_pair_csv_path_cache: dict[tuple[str, str], str] = {}
        self._residue_combo_present_frame_cache = {}
        self._primary_residue_position_cache = {}
        self._dataset_residue_position_cache: dict[str, dict[int, tuple[float, float, float]]] = {}
        self._dataset_compare_active: Optional[dict] = None
        self._dataset_compare_mapping_locks: dict[tuple[str, str], dict[int, tuple[int, ...]]] = {}
        self._dataset_compare_rmsd_distance_threshold = float(DEFAULT_RMSD_FAMILY_MARGIN)
        self._dataset_compare_rmsd_max_threshold = float(DEFAULT_RMSD_MAX_THRESHOLD)
        self._dataset_compare_residue_similarity_threshold = 0.0
        self._dataset_compare_residue_profile_cache: dict[str, dict[int, dict[int, float]]] = {}
        self._dataset_compare_path_scope_cache: dict[tuple[str, int, str, float], dict] = {}
        self._dataset_compare_temporal_evidence_cache: dict[tuple, dict] = {}
        self._dataset_compare_temporal_block_cache: dict[tuple, dict] = {}
        self._dataset_compare_distance_evidence_cache: dict[tuple, dict] = {}
        self._dataset_compare_distance_summary_cache: dict[tuple, dict] = {}
        self._dataset_compare_distance_evidence_context: dict[tuple, dict] = {}
        self._dataset_compare_frame_source_cache: dict[str, tuple[str, list[str]]] = {}
        self._dataset_compare_distance_evidence_thread: QThread | None = None
        self._dataset_compare_distance_evidence_worker: DistanceEvidenceTaskWorker | None = None
        self._last_3d_overlay_signature = None
        self._last_effective_render_signature = None
        self._effective_refresh_requested_at = 0.0
        self._last_entry_exit_signature = None
        self._last_observer_overlay_signature = None
        self._top_controls_widget: Optional[QWidget] = None
        self._top_controls_layout: Optional[QGridLayout] = None
        self._top_controls_items: list[QWidget] = []
        self._color_label: Optional[QLabel] = None
        self._bg_label: Optional[QLabel] = None
        self._loading_depth = 0
        self._chart_operation_history: list[list[int]] = []
        self._statistics_last_rows: list[dict] = []
        self._statistics_last_headers: list[str] = []
        # Retired views remain as compatibility objects only. These switches
        # prevent their hidden tables/charts from querying data or repainting.
        self._legacy_path_panel_enabled = False
        self._legacy_residue_panel_enabled = False
        self._legacy_residue_compare_enabled = False
        self._legacy_residue_combination_enabled = False
        # Profile/Statistics and Combination Regions are exposed together in
        # the detail-on-demand Tunnel Properties page. They remain lazy while
        # Overview is active.
        self._legacy_charts_enabled = True
        self._tunnel_properties_enabled = True
        # Derived from the Datasets tab once the control-panel tabs are built.
        self._side_dock_min_width = 0
        self._side_dock_width = 0
        self._side_dock_collapsed_width = 28
        self._side_dock_auto_expanded = False
        self._side_dock_animation: Optional[QVariantAnimation] = None

        # Build UI
        self._build_toolbar()
        self._build_central()
        self._build_side_panel()
        self._relocate_top_controls()
        self._build_bottom_panel()
        self._build_statusbar()
        self._apply_readable_typography()

        # Wire signals
        self._connect_signals()

        # Load initial data
        QTimer.singleShot(100, self._init_data)

    @staticmethod
    def _scaled_inline_font_sizes(stylesheet: str) -> str:
        """Increase explicit pixel fonts while preserving every other style rule."""
        return re.sub(
            r"font-size\s*:\s*(\d+(?:\.\d+)?)px",
            lambda match: (
                f"font-size:{float(match.group(1)) + 2:g}px"
            ),
            str(stylesheet or ""),
        )

    def _apply_readable_typography(self) -> None:
        """Apply one moderate readability increase across the constructed UI."""
        widgets = [self, *self.findChildren(QWidget)]
        for widget in widgets:
            stylesheet = widget.styleSheet()
            if (
                stylesheet
                and "font-size" in stylesheet
                and widget.property("stage_index") is None
            ):
                widget.setStyleSheet(self._scaled_inline_font_sizes(stylesheet))

            font = widget.font()
            point_size = font.pointSizeF()
            if 0 < point_size < 10.5:
                font.setPointSizeF(10.5)
                widget.setFont(font)

    def _invalidate_render_signatures(self) -> None:
        self._last_3d_overlay_signature = None
        self._last_effective_render_signature = None
        self._last_entry_exit_signature = None
        self._last_observer_overlay_signature = None
        self._path_change_metric_cache_signature = ()
        self._exit_delta_metric_cache_signature = ()

    # ─── Toolbar ────────────────────────────────────────
    def _configure_toolbar_combo(
        self,
        combo: QComboBox,
        *,
        width: Optional[int] = None,
        min_contents: int = 0,
    ):
        """Display-only tuning for compact toolbar combos and their popups."""
        combo.setView(QListView(combo))
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        if min_contents > 0:
            combo.setMinimumContentsLength(min_contents)
        if width is not None:
            combo.setFixedWidth(width)
        view = combo.view()
        if view is not None:
            view.setTextElideMode(Qt.ElideNone)
            view.setMinimumWidth(max(combo.width(), combo.sizeHint().width() + 24))

    def _refresh_combo_popup_width(self, combo: QComboBox, extra: int = 40):
        """Ensure popup is wide enough to show full option text."""
        view = combo.view()
        if view is None:
            return
        metrics = QFontMetrics(view.font())
        content_width = 0
        for i in range(combo.count()):
            content_width = max(content_width, metrics.horizontalAdvance(combo.itemText(i)))
        popup_width = max(combo.width(), content_width + extra)
        view.setMinimumWidth(popup_width)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar{spacing:3px;padding:2px 4px;}")
        self.addToolBar(tb)
        self._main_toolbar = tb

        _lbl = "font-size:11px;color:#909399;padding:0 1px;"
        _sm = "padding:3px 7px;font-size:11px;min-height:18px;"

        # ── 1  Display ───────────────────────────────────
        lbl = QLabel("Color")
        lbl.setStyleSheet(_lbl); lbl.setToolTip("Background path color (light gray ↔ black)")
        tb.addWidget(lbl)
        self._path_color_slider = QSlider(Qt.Horizontal)
        self._path_color_slider.setRange(0, 100); self._path_color_slider.setValue(35)
        self._path_color_slider.setFixedWidth(55)
        self._path_color_slider.setToolTip("Background path color: left=light gray, right=black")
        self._path_color_slider.valueChanged.connect(self._on_path_color_changed)
        tb.addWidget(self._path_color_slider)

        lbl = QLabel("BG")
        lbl.setStyleSheet(_lbl); lbl.setToolTip("Background opacity")
        tb.addWidget(lbl)
        self._bg_opacity_slider = QSlider(Qt.Horizontal)
        self._bg_opacity_slider.setRange(1, 50); self._bg_opacity_slider.setValue(15)
        self._bg_opacity_slider.setFixedWidth(55); self._bg_opacity_slider.setToolTip("Background opacity")
        self._bg_opacity_slider.valueChanged.connect(self._on_bg_opacity_changed)
        tb.addWidget(self._bg_opacity_slider)

        self._path_change_btn = QPushButton("Path Δ")
        self._path_change_btn.setCheckable(True)
        self._path_change_btn.setStyleSheet(_sm)
        self._path_change_btn.setToolTip("Color paths by original/current coordinate distance; red means larger change")
        self._path_change_btn.toggled.connect(self._toggle_path_change_overlay)
        tb.addWidget(self._path_change_btn)

        self._exit_delta_btn = QPushButton("Exit Δ")
        self._exit_delta_btn.setCheckable(True)
        self._exit_delta_btn.setStyleSheet(_sm)
        self._exit_delta_btn.setToolTip(
            "Color paths by the difference between current and original exit-to-cluster-center distances"
        )
        self._exit_delta_btn.toggled.connect(self._toggle_exit_delta_overlay)
        tb.addWidget(self._exit_delta_btn)

        self._path_delta_filter_slider = QSlider(Qt.Horizontal)
        self._path_delta_filter_slider.setRange(0, 100)
        self._path_delta_filter_slider.setValue(0)
        self._path_delta_filter_slider.setFixedWidth(84)
        self._path_delta_filter_slider.setEnabled(False)
        self._path_delta_filter_slider.setTracking(False)
        self._path_delta_filter_slider.setToolTip("Exclude paths with the active Δ value at or below this threshold")
        self._path_delta_filter_slider.valueChanged.connect(self._on_path_delta_filter_changed)
        tb.addWidget(self._path_delta_filter_slider)

        self._path_delta_filter_label = QLabel("Δ off")
        self._path_delta_filter_label.setStyleSheet(_lbl)
        self._path_delta_filter_label.setMinimumWidth(84)
        self._path_delta_filter_label.setToolTip("Active Δ exclusion threshold")
        tb.addWidget(self._path_delta_filter_label)

        tb.addSeparator()

        # ── 2  Layers ────────────────────────────────────
        self._residue_toggle_btn = QPushButton("Residues")
        self._residue_toggle_btn.setCheckable(True); self._residue_toggle_btn.setStyleSheet(_sm)
        self._residue_toggle_btn.setToolTip("Show/hide residue 3D positions")
        self._residue_toggle_btn.toggled.connect(self._toggle_residue_positions)
        tb.addWidget(self._residue_toggle_btn)

        self._residue_labels_btn = QPushButton("Res Labels")
        self._residue_labels_btn.setCheckable(True); self._residue_labels_btn.setStyleSheet(_sm)
        self._residue_labels_btn.setToolTip("Show/hide labels for currently selected residues")
        self._residue_labels_btn.toggled.connect(self._toggle_residue_labels)
        tb.addWidget(self._residue_labels_btn)

        self._entry_exit_btn = QPushButton("Entry/Exit")
        self._entry_exit_btn.setCheckable(True); self._entry_exit_btn.setStyleSheet(_sm)
        self._entry_exit_btn.setToolTip("Show/hide entrance (green) and exit (red) points")
        self._entry_exit_btn.toggled.connect(self._toggle_entry_exit_points)
        tb.addWidget(self._entry_exit_btn)

        self._entry_exit_scope_btn = QPushButton("EE: All")
        self._entry_exit_scope_btn.setCheckable(True); self._entry_exit_scope_btn.setStyleSheet(_sm)
        self._entry_exit_scope_btn.setToolTip("Toggle entry/exit scope: all paths or current paths")
        self._entry_exit_scope_btn.toggled.connect(self._toggle_entry_exit_scope)
        tb.addWidget(self._entry_exit_scope_btn)

        self._exit_cluster_btn = QPushButton("Exit Cluster")
        self._exit_cluster_btn.setCheckable(True); self._exit_cluster_btn.setStyleSheet(_sm)
        self._exit_cluster_btn.setToolTip("Click an exit sphere to select all visually connected exit spheres at the current zoom")
        self._exit_cluster_btn.toggled.connect(self._toggle_exit_cluster_select)
        tb.addWidget(self._exit_cluster_btn)

        self._focus_btn = QPushButton("Focus")
        self._focus_btn.setCheckable(True); self._focus_btn.setStyleSheet(_sm)
        self._focus_btn.setToolTip("Focus: hide background, show only selected paths")
        self._focus_btn.toggled.connect(self._toggle_focus_mode)
        tb.addWidget(self._focus_btn)

        tb.addSeparator()

        # ── 3  Protein (compact) ─────────────────────────
        self._load_protein_btn = QPushButton("Protein…")
        self._load_protein_btn.setStyleSheet(_sm)
        self._load_protein_btn.setToolTip("Load protein PDB overlay")
        self._load_protein_btn.clicked.connect(self._on_load_protein)
        tb.addWidget(self._load_protein_btn)

        self._main_view_btn = QPushButton("Main View")
        self._main_view_btn.setCheckable(True)
        self._main_view_btn.setChecked(True)
        self._main_view_btn.setStyleSheet(_sm)
        self._main_view_btn.setToolTip("Show the aligned tunnel/path workspace (Ctrl+1)")
        self._main_view_btn.clicked.connect(lambda _checked=False: self._set_main_workspace("main"))
        tb.addWidget(self._main_view_btn)

        self._load_observer_protein_btn = QPushButton("Residue Observer")
        self._load_observer_protein_btn.setCheckable(True)
        self._load_observer_protein_btn.setChecked(False)
        self._load_observer_protein_btn.setStyleSheet(_sm)
        self._load_observer_protein_btn.setToolTip("Show linked structural residue comparisons (Ctrl+2)")
        self._load_observer_protein_btn.clicked.connect(lambda _checked=False: self._set_main_workspace("observer"))
        tb.addWidget(self._load_observer_protein_btn)

        self._evidence_view_btn = QPushButton("Evidence")
        self._evidence_view_btn.setCheckable(True)
        self._evidence_view_btn.setChecked(False)
        self._evidence_view_btn.setStyleSheet(_sm)
        self._evidence_view_btn.setToolTip(
            "Show the linked evidence overview and detail views (Ctrl+3)"
        )
        self._evidence_view_btn.clicked.connect(
            lambda checked=False: self._set_main_workspace("evidence" if checked else "main")
        )
        tb.addWidget(self._evidence_view_btn)

        self._protein_toggle_btn = QPushButton("Reference PDB")
        self._protein_toggle_btn.setCheckable(True)
        self._protein_toggle_btn.setEnabled(False)
        self._protein_toggle_btn.setStyleSheet(_sm)
        self._protein_toggle_btn.setToolTip("Show/hide the loaded reference PDB overlay")
        self._protein_toggle_btn.toggled.connect(self._toggle_protein_visible)
        tb.addWidget(self._protein_toggle_btn)

        self._protein_style_combo = QComboBox()
        self._protein_style_combo.addItems(["cartoon", "backbone", "tube", "ca_spheres"])
        self._protein_style_combo.setCurrentText("cartoon")
        self._protein_style_combo.setFixedHeight(24)
        self._protein_style_combo.setMaxVisibleItems(10)
        self._configure_toolbar_combo(self._protein_style_combo, width=108, min_contents=10)
        self._refresh_combo_popup_width(self._protein_style_combo)
        self._protein_style_combo.setEnabled(False)
        self._protein_style_combo.setToolTip("Protein style")
        self._protein_style_combo.currentTextChanged.connect(self._on_protein_style_changed)
        tb.addWidget(self._protein_style_combo)

        self._protein_align_combo = QComboBox()
        self._protein_align_combo.addItem("kabsch", "auto_kabsch")
        self._protein_align_combo.addItem("translate", "translate")
        self._protein_align_combo.addItem("none", "none")
        self._protein_align_combo.setCurrentIndex(0)
        self._protein_align_combo.setFixedHeight(24)
        self._protein_align_combo.setMaxVisibleItems(10)
        self._configure_toolbar_combo(self._protein_align_combo, width=104, min_contents=10)
        self._refresh_combo_popup_width(self._protein_align_combo)
        self._protein_align_combo.setToolTip("Alignment mode")
        self._protein_align_combo.currentIndexChanged.connect(self._on_protein_align_mode_changed)
        tb.addWidget(self._protein_align_combo)

        self._protein_opacity_slider = QSlider(Qt.Horizontal)
        self._protein_opacity_slider.setRange(0, 100)
        self._protein_opacity_slider.setValue(100)
        self._protein_opacity_slider.setFixedWidth(40)
        self._protein_opacity_slider.setEnabled(False)
        self._protein_opacity_slider.setToolTip("Protein opacity")
        self._protein_opacity_slider.valueChanged.connect(self._on_protein_opacity_changed)
        tb.addWidget(self._protein_opacity_slider)

        self._clear_protein_btn = QPushButton("Clear")
        self._clear_protein_btn.setProperty("flat", True)
        self._clear_protein_btn.setEnabled(False)
        self._clear_protein_btn.setStyleSheet(_sm)
        self._clear_protein_btn.setToolTip("Remove protein overlay")
        self._clear_protein_btn.clicked.connect(self._on_clear_protein)
        tb.addWidget(self._clear_protein_btn)

        self._binding_sites_btn = QPushButton("Sites")
        self._binding_sites_btn.setCheckable(True)
        self._binding_sites_btn.setChecked(True)
        self._binding_sites_btn.setEnabled(False)
        self._binding_sites_btn.setStyleSheet(_sm)
        self._binding_sites_btn.setToolTip("Show/hide binding site surfaces")
        self._binding_sites_btn.toggled.connect(self._toggle_binding_sites)
        tb.addWidget(self._binding_sites_btn)

        tb.addSeparator()

        # ── 4  Frame (collapsed popup) ───────────────────
        self._frame_menu_btn = QToolButton()
        self._frame_menu_btn.setText("Frame ▾")
        self._frame_menu_btn.setFixedHeight(24)
        self._frame_menu_btn.setToolTip("Set frame range filter")
        self._frame_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._frame_menu_btn.setStyleSheet(
            "QToolButton{padding:3px 7px;font-size:11px;}"
            "QToolButton::menu-indicator{image:none;}"
        )
        _fm = QMenu(self._frame_menu_btn)
        _fw = QWidget()
        _fl = QHBoxLayout(_fw); _fl.setContentsMargins(8, 6, 8, 6); _fl.setSpacing(4)
        self._frame_min_combo = QComboBox()
        self._frame_min_combo.setEditable(True)
        self._frame_min_combo.setFixedHeight(24)
        self._frame_min_combo.setMaxVisibleItems(30)
        self._configure_toolbar_combo(self._frame_min_combo, width=70, min_contents=6)
        _fl.addWidget(self._frame_min_combo)
        _fl.addWidget(QLabel("–"))
        self._frame_max_combo = QComboBox()
        self._frame_max_combo.setEditable(True)
        self._frame_max_combo.setFixedHeight(24)
        self._frame_max_combo.setMaxVisibleItems(30)
        self._configure_toolbar_combo(self._frame_max_combo, width=70, min_contents=6)
        _fl.addWidget(self._frame_max_combo)
        self._apply_frame_btn = QPushButton("Apply")
        self._apply_frame_btn.setFixedHeight(24); self._apply_frame_btn.setStyleSheet(_sm)
        self._apply_frame_btn.clicked.connect(self._apply_frame_range)
        _fl.addWidget(self._apply_frame_btn)
        self._clear_frame_btn = QPushButton("Clear")
        self._clear_frame_btn.setProperty("flat", True)
        self._clear_frame_btn.setFixedHeight(24); self._clear_frame_btn.setStyleSheet(_sm)
        self._clear_frame_btn.clicked.connect(self._clear_frame_range)
        _fl.addWidget(self._clear_frame_btn)
        _fa = QWidgetAction(_fm); _fa.setDefaultWidget(_fw)
        _fm.addAction(_fa)
        self._frame_menu_btn.setMenu(_fm)
        tb.addWidget(self._frame_menu_btn)

        tb.addSeparator()

        # ── 5  Selection ─────────────────────────────────
        self._point_select_btn = QPushButton("Point Select")
        self._point_select_btn.setCheckable(True); self._point_select_btn.setStyleSheet(_sm)
        self._point_select_btn.setToolTip(
            "ON: lasso selects entry/exit points (coarse)\n"
            "OFF: lasso selects paths (fine)")
        self._point_select_btn.toggled.connect(self._toggle_selection_mode)
        tb.addWidget(self._point_select_btn)

        self._lasso_btn = QPushButton("Lasso")
        self._lasso_btn.setCheckable(True); self._lasso_btn.setStyleSheet(_sm)
        self._lasso_btn.setToolTip(
            "Force-lasso mode (L)\nShift+Drag: lasso\nAlt+Shift+Drag: remove\nEsc: clear")
        self._lasso_btn.toggled.connect(self._toggle_lasso)
        tb.addWidget(self._lasso_btn)

        self._clear_sel_btn = QPushButton("Clear Sel")
        self._clear_sel_btn.setProperty("flat", True)
        self._clear_sel_btn.setStyleSheet(_sm)
        self._clear_sel_btn.setToolTip("Clear selection (Esc)")
        self._clear_sel_btn.clicked.connect(self._clear_selection)
        tb.addWidget(self._clear_sel_btn)

    # ─── Central Widget ─────────────────────────────────
    def _build_observer_dataset_panel(self, parent: QWidget, fallback_title: str) -> dict:
        container = QFrame(parent)
        container.setStyleSheet("QFrame{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;}")
        container.setMinimumWidth(340)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel(str(fallback_title))
        title.setStyleSheet("font-size:11px;font-weight:600;color:#4b5563;")
        layout.addWidget(title)

        residue_summary = QLabel("")
        residue_summary.setWordWrap(True)
        residue_summary.setTextFormat(Qt.RichText)
        residue_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        residue_summary.setStyleSheet(
            "font-size:11px;color:#374151;background:#ffffff;"
            "border:1px solid #e5e7eb;border-radius:4px;padding:4px;"
        )
        residue_summary.setVisible(False)
        layout.addWidget(residue_summary)

        panel = ProteinViewerPanel(container, compact=True)
        panel.sequence_prepare_completed.connect(
            lambda _generation, _index, _payload, error, viewer_panel=panel: (
                self._on_observer_panel_prepare_completed(viewer_panel, str(error or ""))
            )
        )
        panel.setMinimumWidth(300)
        layout.addWidget(panel, 1)
        return {
            "container": container,
            "label": title,
            "residue_summary": residue_summary,
            "panel": panel,
            "fallback_title": str(fallback_title),
            "source_name": str(fallback_title),
        }

    def _protein_observer_slots(self) -> list[dict]:
        dynamic = getattr(self, "_protein_observer_slots_by_dataset", None)
        if isinstance(dynamic, dict) and dynamic:
            return [slot for slot in dynamic.values() if isinstance(slot, dict)]
        slots: list[dict] = []
        for slot in (
            getattr(self, "_protein_viewer_panel_top", None),
            getattr(self, "_protein_viewer_panel_bottom", None),
        ):
            if isinstance(slot, dict):
                slots.append(slot)
        return slots

    def _iter_protein_observer_panels(self) -> list[ProteinViewerPanel]:
        return [
            slot["panel"]
            for slot in self._protein_observer_slots()
            if slot.get("panel") is not None
        ]

    def _set_observer_highlight_style(self, style: str) -> None:
        for panel in self._iter_protein_observer_panels():
            panel.set_highlight_style(style)

    def _set_observer_model_style(self, style: str) -> None:
        for panel in self._iter_protein_observer_panels():
            if hasattr(panel, "set_model_style"):
                panel.set_model_style(style)

    def _set_observer_protein_opacity(self, opacity: float) -> None:
        for panel in self._iter_protein_observer_panels():
            panel.set_protein_opacity(opacity)

    def _set_protein_observer_slot_title(self, slot: dict | None, title: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        label = slot.get("label")
        if label is None:
            return
        text = str(title or slot.get("fallback_title") or "").strip()
        label.setText(text or str(slot.get("fallback_title") or "Observer"))

    def _observer_slot_key(self, slot: dict | None) -> str:
        if isinstance(slot, dict) and str(slot.get("dataset_key") or ""):
            return str(slot.get("dataset_key"))
        if slot is getattr(self, "_protein_viewer_panel_top", None):
            return "top"
        if slot is getattr(self, "_protein_viewer_panel_bottom", None):
            return "bottom"
        return ""

    def _observer_slot_for_key(self, key: str) -> dict | None:
        value = str(key or "")
        if value == "top":
            return getattr(self, "_protein_viewer_panel_top", None)
        if value == "bottom":
            return getattr(self, "_protein_viewer_panel_bottom", None)
        slot = getattr(self, "_protein_observer_slots_by_dataset", {}).get(value)
        return slot if isinstance(slot, dict) else None

    def _observer_frame_value(self) -> int:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return 0
        return int(spin.value())

    def _update_observer_slot_title(self, slot: dict | None, dataset_label: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        panel = slot.get("panel")
        fallback_title = str(slot.get("role_title") or slot.get("fallback_title") or "Observer")
        label = str(dataset_label or slot.get("source_name") or fallback_title).strip()
        if panel is not None and hasattr(panel, "has_sequence") and panel.has_sequence():
            current_frame = panel.current_sequence_frame()
            if current_frame is not None:
                self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label} | frame {current_frame}")
                return
            self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label}")
            return
        self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label}")

    def _observer_sequence_slots(self) -> list[dict]:
        result: list[dict] = []
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if panel is not None and hasattr(panel, "has_sequence") and panel.has_sequence():
                result.append(slot)
        return result

    def _observer_slot_combo(self, side: str):
        # Dataset choice is now controlled by the multi-select dropdown.
        return None

    def _observer_dataset_choices(self) -> list[tuple[str, str]]:
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        choices: list[tuple[str, str]] = []
        for dataset in datasets:
            key = str(dataset.get("key") or "").strip()
            if not key:
                continue
            label = self._dataset_display_name(key, str(dataset.get("name") or key))
            choices.append((label, key))
        return choices

    def _selected_observer_dataset_keys(self) -> list[str]:
        selector = getattr(self, "_observer_dataset_multiselect", None)
        if selector is not None and hasattr(selector, "checked_keys"):
            return list(selector.checked_keys())
        return [str(slot.get("dataset_key") or "") for slot in self._protein_observer_slots() if str(slot.get("dataset_key") or "")]

    def _refresh_observer_dataset_multiselect(self, preferred_keys=None) -> None:
        selector = getattr(self, "_observer_dataset_multiselect", None)
        if selector is None:
            return
        choices = self._observer_dataset_choices()
        available = {key for _label, key in choices}
        selected = [
            str(key) for key in (
                preferred_keys
                if preferred_keys is not None
                else self._selected_observer_dataset_keys()
            )
            if str(key) in available
        ]
        if not selected:
            pending = dict(getattr(self, "_observer_pending_payload", {}) or {})
            for field in ("left_dataset_key", "right_dataset_key"):
                key = str(pending.get(field) or "")
                if key in available and key not in selected:
                    selected.append(key)
        if not selected:
            selected = [key for _label, key in choices[:2]]
        signature = (tuple(choices), tuple(selected))
        if (
            signature == getattr(self, "_observer_dataset_layout_signature", ())
            and tuple(getattr(self, "_protein_observer_slots_by_dataset", {}).keys()) == tuple(selected)
        ):
            return
        selector.set_choices(choices, checked_keys=selected)
        selector.set_checked_keys(selected, emit=False)
        self._rebuild_observer_dataset_slots(selected)
        self._observer_dataset_layout_signature = signature

    def _set_checked_observer_datasets(self, dataset_keys) -> None:
        available = {key for _label, key in self._observer_dataset_choices()}
        wanted = [str(key) for key in (dataset_keys or []) if str(key) in available]
        selector = getattr(self, "_observer_dataset_multiselect", None)
        if selector is not None:
            selector.set_checked_keys(wanted, emit=False)
        self._rebuild_observer_dataset_slots(wanted)
        choices = self._observer_dataset_choices()
        self._observer_dataset_layout_signature = (tuple(choices), tuple(wanted))

    def _on_observer_dataset_multiselect_changed(self, dataset_keys) -> None:
        self._observer_compare_dataset_keys = [
            str(key) for key in (dataset_keys or []) if str(key)
        ]
        self._rebuild_observer_dataset_slots(dataset_keys)
        self._observer_dataset_layout_signature = (
            tuple(self._observer_dataset_choices()),
            tuple(str(key) for key in (dataset_keys or [])),
        )
        if self._protein_observer_page_active():
            if self._load_uninitialized_observer_datasets():
                return
            self._finalize_observer_dataset_loads()
            return
        self._refresh_observer_frame_controls(preferred_frame=self._observer_frame_value())
        self._refresh_observer_timeline()

    def _load_uninitialized_observer_datasets(self) -> bool:
        """Load Observer panels progressively instead of starting them together."""
        unloaded: list[str] = []
        for slot in self._protein_observer_slots():
            dataset_key = str(slot.get("dataset_key") or "")
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if (
                dataset_key
                and panel is not None
                and hasattr(panel, "has_loaded_protein")
                and not panel.has_loaded_protein()
            ):
                unloaded.append(dataset_key)
        if not unloaded:
            return False

        active = [str(getattr(self, "_observer_loading_dataset_key", "") or "")]
        active.extend(str(key) for key in getattr(self, "_observer_load_queue", []))
        active = [key for key in active if key]
        if active == unloaded:
            return True

        self._observer_load_generation += 1
        generation = int(self._observer_load_generation)
        self._observer_load_queue = list(unloaded)
        self._observer_loading_dataset_key = ""
        for slot in self._protein_observer_slots():
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is not None and hasattr(panel, "set_sequence_prefetch_enabled"):
                panel.set_sequence_prefetch_enabled(False)
        # Give the stacked widget one display frame before any filesystem or
        # VTK work starts, so the click itself always receives immediate visual
        # feedback.
        QTimer.singleShot(16, lambda token=generation: self._load_next_observer_dataset(token))
        return True

    def _load_next_observer_dataset(self, generation: int) -> None:
        if int(generation) != int(getattr(self, "_observer_load_generation", 0)):
            return
        if not self._protein_observer_page_active():
            self._observer_load_queue = []
            self._observer_loading_dataset_key = ""
            return
        while self._observer_load_queue:
            dataset_key = str(self._observer_load_queue.pop(0) or "")
            slot = self._observer_slot_for_key(dataset_key)
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is None or panel.has_loaded_protein():
                continue
            self._observer_loading_dataset_key = dataset_key
            label = self._dataset_display_name(dataset_key, dataset_key)
            self._set_protein_observer_slot_title(slot, f"{label} | loading first frame…")
            pending = self._on_observer_sequence_source_changed(
                dataset_key,
                refresh_linked_views=False,
            )
            if pending:
                return
            self._observer_loading_dataset_key = ""
            QTimer.singleShot(0, lambda token=generation: self._load_next_observer_dataset(token))
            return
        self._observer_loading_dataset_key = ""
        self._finalize_observer_dataset_loads()
        QTimer.singleShot(
            650,
            lambda token=generation: self._enable_observer_sequence_prefetch(token),
        )

    def _on_observer_panel_prepare_completed(self, panel, error: str) -> None:
        dataset_key = str(getattr(self, "_observer_loading_dataset_key", "") or "")
        if not dataset_key:
            return
        slot = self._observer_slot_for_key(dataset_key)
        if not isinstance(slot, dict) or slot.get("panel") is not panel:
            return
        if not str(error or "") and not panel.has_loaded_protein():
            return
        self._observer_loading_dataset_key = ""
        generation = int(getattr(self, "_observer_load_generation", 0))
        QTimer.singleShot(0, lambda token=generation: self._load_next_observer_dataset(token))

    def _enable_observer_sequence_prefetch(self, generation: int) -> None:
        if int(generation) != int(getattr(self, "_observer_load_generation", 0)):
            return
        if not self._protein_observer_page_active():
            return
        for panel in self._iter_protein_observer_panels():
            if hasattr(panel, "set_sequence_prefetch_enabled"):
                panel.set_sequence_prefetch_enabled(True)

    def _finalize_observer_dataset_loads(self) -> None:
        """Refresh linked Observer views once after the staged first load."""
        self._refresh_observer_frame_controls(preferred_frame=self._observer_frame_value())
        self._refresh_observer_timeline()
        pending = dict(getattr(self, "_observer_pending_payload", {}) or {})
        if pending.get("source") == "dataset_compare_motif":
            self._update_combination_observer_views(pending)
            return
        self._refresh_observer_path_overlays()
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        if 1 <= len(residue_ids) <= 4:
            self._apply_observer_debug_selection()

    def _rebuild_observer_dataset_slots(self, dataset_keys) -> None:
        content = getattr(self, "_observer_panels_content", None)
        panels_layout = getattr(self, "_observer_panels_layout", None)
        if content is None or panels_layout is None:
            return
        choices = dict((key, label) for label, key in self._observer_dataset_choices())
        ordered_keys = [str(key) for key in (dataset_keys or []) if str(key) in choices]
        existing = dict(getattr(self, "_protein_observer_slots_by_dataset", {}) or {})
        for key, slot in list(existing.items()):
            if key in ordered_keys:
                continue
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is not None:
                try:
                    panel.shutdown()
                except Exception:
                    pass
            container = slot.get("container") if isinstance(slot, dict) else None
            if container is not None:
                panels_layout.removeWidget(container)
                container.deleteLater()
            existing.pop(key, None)

        rebuilt: dict[str, dict] = {}
        for key in ordered_keys:
            slot = existing.get(key)
            if not isinstance(slot, dict):
                label = choices.get(key, key)
                slot = self._build_observer_dataset_panel(content, label)
                slot["dataset_key"] = key
                slot["role_title"] = label
                slot["source_name"] = label
                panels_layout.addWidget(slot["container"], 1)
                panel = slot.get("panel")
                if panel is not None:
                    panel.set_model_style(str(self._observer_model_style_combo.currentData() or "cartoon"))
                    panel.set_highlight_style(str(self._observer_highlight_style_combo.currentData() or "ball_and_stick"))
                    panel.set_protein_opacity(int(self._observer_protein_opacity_slider.value()) / 100.0)
            rebuilt[key] = slot
        if tuple(existing.keys()) != tuple(ordered_keys):
            for key in ordered_keys:
                container = rebuilt[key].get("container")
                if container is not None:
                    panels_layout.removeWidget(container)
                    panels_layout.addWidget(container, 1)
        self._protein_observer_slots_by_dataset = rebuilt
        slots = list(rebuilt.values())
        self._protein_viewer_panel_top = slots[0] if slots else None
        self._protein_viewer_panel_bottom = slots[1] if len(slots) > 1 else None
        self._protein_viewer_panel = self._protein_viewer_panel_top.get("panel") if isinstance(self._protein_viewer_panel_top, dict) else None
        self._last_observer_overlay_signature = None

    def _protein_observer_page_active(self) -> bool:
        stack = getattr(self, "_main_view_stack", None)
        container = getattr(self, "_protein_observer_container", None)
        return bool(stack is not None and container is not None and stack.currentWidget() is container)

    def _preferred_observer_dataset_key(self, side: str) -> str:
        slot = self._observer_slot_for_key(side)
        if isinstance(slot, dict):
            key = str(slot.get("dataset_key") or "").strip()
            if key:
                return key
        combo = self._observer_slot_combo(side)
        if combo is not None:
            key = str(combo.currentData() or "").strip()
            if key:
                return key
        pending = dict(getattr(self, "_observer_pending_payload", {}) or {})
        side_key = "left_dataset_key" if str(side) == "top" else "right_dataset_key"
        key = str(pending.get(side_key, "") or "").strip()
        if key:
            return key
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is not None:
            if str(side) == "top" and hasattr(panel, "left_dataset_key"):
                key = str(panel.left_dataset_key() or "").strip()
            elif str(side) == "bottom" and hasattr(panel, "right_dataset_key"):
                key = str(panel.right_dataset_key() or "").strip()
            else:
                key = ""
            if key:
                return key
        choices = self._observer_dataset_choices()
        if not choices:
            return ""
        if str(side) == "bottom" and len(choices) >= 2:
            return str(choices[1][1])
        return str(choices[0][1])

    def _observer_sequence_source_candidates(self, dataset_key: str) -> list[tuple[str, str]]:
        dataset_key = str(dataset_key or "").strip()
        cached = getattr(self, "_observer_sequence_source_cache", {}).get(dataset_key)
        if cached is not None:
            return list(cached)
        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset is None:
            return []
        md_file_sources: list[str] = []
        md_root_sources: list[str] = []
        md_root_path = str(getattr(dataset, "md_root_path", "") or "").strip()
        if md_root_path:
            normalized_md_root = os.path.abspath(md_root_path)
            if os.path.isfile(normalized_md_root) and normalized_md_root.lower().endswith(".pdb"):
                normalized_md_root = os.path.dirname(normalized_md_root)
            if os.path.isdir(normalized_md_root):
                md_root_sources.append(normalized_md_root)
        dataset_info = self._dataset_info_for_statistics(str(dataset_key or "").strip())
        md_path_file = self._find_md_path_file_near_dataset(dataset_info)
        if md_path_file:
            try:
                with open(md_path_file, "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        raw_value = str(line).strip().strip("\"'")
                        if not raw_value:
                            continue
                        candidate = (
                            os.path.abspath(raw_value)
                            if os.path.isabs(raw_value)
                            else os.path.abspath(os.path.join(os.path.dirname(md_path_file), raw_value))
                        )
                        if os.path.isfile(candidate) and candidate.lower().endswith(".pdb"):
                            candidate = os.path.dirname(candidate)
                        if candidate and os.path.isdir(candidate):
                            md_file_sources.append(candidate)
            except OSError:
                pass
        roots: list[str] = []
        root_candidates = (
            md_file_sources
            if md_file_sources
            else (md_root_sources if md_root_sources else [getattr(dataset, "folder", "")])
        )
        for candidate in root_candidates:
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and normalized not in roots and os.path.isdir(normalized):
                roots.append(normalized)

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for root in roots:
            direct_has_pdb = False
            try:
                direct_has_pdb = any(name.lower().endswith(".pdb") for name in os.listdir(root))
            except OSError:
                direct_has_pdb = False
            if direct_has_pdb:
                label = os.path.basename(root) or root
                candidates.append((label, root))
                seen.add(os.path.normcase(root))
                continue
            try:
                child_names = sorted(os.listdir(root))
            except OSError:
                child_names = []
            for child_name in child_names:
                child_path = os.path.abspath(os.path.join(root, child_name))
                if not os.path.isdir(child_path):
                    continue
                key = os.path.normcase(child_path)
                if key in seen:
                    continue
                try:
                    has_pdb = any(name.lower().endswith(".pdb") for name in os.listdir(child_path))
                except OSError:
                    has_pdb = False
                if not has_pdb:
                    continue
                label = f"{os.path.basename(root) or root} / {child_name}"
                candidates.append((label, child_path))
                seen.add(key)
            if candidates:
                continue
            try:
                for walk_root, _dirs, files in os.walk(root):
                    walk_path = os.path.abspath(walk_root)
                    if os.path.normcase(walk_path) == os.path.normcase(root):
                        continue
                    if not any(name.lower().endswith(".pdb") for name in files):
                        continue
                    key = os.path.normcase(walk_path)
                    if key in seen:
                        continue
                    label = os.path.relpath(walk_path, root)
                    candidates.append((f"{os.path.basename(root) or root} / {label}", walk_path))
                    seen.add(key)
                    break
            except OSError:
                pass
        self._observer_sequence_source_cache[dataset_key] = tuple(candidates)
        return list(candidates)

    def _observer_resolve_sequence_source(self, dataset_key: str) -> tuple[str, str]:
        dataset_key = str(dataset_key or "").strip()
        candidates = self._observer_sequence_source_candidates(dataset_key)
        if not candidates:
            return "", ""
        if len(candidates) == 1:
            return candidates[0]

        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        tokens: list[str] = []
        for raw_value in (
            getattr(dataset, "name", "") if dataset is not None else "",
            os.path.basename(str(getattr(dataset, "folder", "") or "").strip()) if dataset is not None else "",
            getattr(dataset, "prefix", "") if dataset is not None else "",
        ):
            token = str(raw_value or "").strip().lower()
            if token:
                tokens.append(token)

        ranked: list[tuple[int, str, str]] = []
        for label, path in candidates:
            basename = os.path.basename(str(path or "")).strip().lower()
            label_text = str(label or "").strip().lower()
            score = 0
            for token in tokens:
                if basename == token:
                    score += 8
                elif basename.startswith(token) or basename.endswith(token):
                    score += 5
                elif token in basename:
                    score += 3
                if label_text == token:
                    score += 4
                elif token in label_text:
                    score += 2
            ranked.append((score, label, path))
        ranked.sort(key=lambda item: (-int(item[0]), str(item[1]).lower(), str(item[2]).lower()))
        _score, label, path = ranked[0]
        return str(label), str(path)

    def _set_observer_dataset_selection(self, side: str, dataset_key: str) -> None:
        dataset_key = str(dataset_key or "").strip()
        if not dataset_key:
            return
        selected = self._selected_observer_dataset_keys()
        if dataset_key not in selected:
            selected.append(dataset_key)
            self._set_checked_observer_datasets(selected)

    def _refresh_observer_sequence_source_combo(self, side: str) -> None:
        self._refresh_observer_dataset_multiselect()

    def _on_observer_sequence_source_changed(
        self,
        side: str,
        *,
        refresh_linked_views: bool = True,
    ) -> bool:
        slot = self._observer_slot_for_key(side)
        if not isinstance(slot, dict):
            return False
        dataset_key = str(slot.get("dataset_key") or "").strip()
        slot["dataset_key"] = dataset_key
        panel = slot.get("panel")
        if panel is None:
            return False
        if hasattr(panel, "set_display_transform_resolver"):
            panel.set_display_transform_resolver(
                (
                    lambda frame_number, pdb_path, key=dataset_key: (
                        self._observer_pipeline_display_transform(key, frame_number, pdb_path)
                    )
                )
                if dataset_key
                else None
            )
        dataset_label = self._dataset_display_name(dataset_key, str(slot.get("fallback_title") or "Observer")) if dataset_key else str(slot.get("fallback_title") or "Observer")
        if not dataset_key:
            panel.clear_protein()
            slot["source_name"] = dataset_label
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | {dataset_label}",
            )
            if refresh_linked_views:
                self._refresh_observer_frame_controls()
                self._refresh_observer_timeline()
                self._refresh_observer_path_overlays()
            return False
        source_label, folder = self._observer_resolve_sequence_source(dataset_key)
        if not folder:
            panel.clear_protein()
            slot["source_name"] = dataset_label
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | {dataset_label} | no MD frames",
            )
            self._protein_observer_debug_label.setText(
                f"{slot.get('fallback_title', 'Observer')} has no readable MD frame folder"
            )
            if refresh_linked_views:
                self._refresh_observer_frame_controls()
                self._refresh_observer_timeline()
                self._refresh_observer_path_overlays()
            return False
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        preferred_frame = self._observer_frame_value()
        result = panel.load_pdb_sequence_folder(folder)
        if not bool(result.get("ok")):
            self._protein_observer_debug_label.setText(
                f"{slot.get('fallback_title', 'Observer')} sequence load failed: {result.get('error', 'unknown_error')}"
            )
            self._statusbar.showMessage("Observer frame folder load failed", 4000)
            return False
        slot["source_name"] = dataset_label
        slot["sequence_sources"] = self._observer_sequence_source_candidates(dataset_key)
        self._update_observer_slot_title(slot, dataset_label)
        if refresh_linked_views:
            self._refresh_observer_frame_controls(preferred_frame=preferred_frame or int(result.get("current_frame", 0) or 0))
        target_frame = (
            self._observer_frame_value()
            if refresh_linked_views and self._observer_frame_value() > 0
            else int(result.get("current_frame", 0) or 0)
        )
        current_frame_raw = result.get("current_frame", result.get("frame", target_frame))
        current_frame = int(target_frame if current_frame_raw is None else current_frame_raw)
        if target_frame != current_frame:
            frame_result = panel.set_sequence_frame(target_frame)
            if bool(frame_result.get("ok")):
                result = {**result, **frame_result}
        panel.set_highlight_residues(
            residue_ids,
            focus_loaded=True,
            refresh_sequence_cache=False,
            defer_render=bool(residue_ids),
        )
        self._protein_observer_debug_label.setText(
            f"{slot.get('fallback_title', 'Observer')} loaded {dataset_label} | "
            f"{source_label or os.path.basename(folder)} | "
            f"{int(result.get('frame_count', 0) or 0)} frames | "
            f"frame {result.get('frame', result.get('current_frame', target_frame))} | "
            f"{folder}"
        )
        self._statusbar.showMessage(
            f"{slot.get('fallback_title', 'Observer')} frame source loaded: {dataset_label}",
            4000,
        )
        if refresh_linked_views:
            self._refresh_observer_path_overlays()
        return bool(result.get("pending"))

    def _refresh_observer_frame_controls(self, preferred_frame: int | None = None) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        prev_btn = getattr(self, "_observer_prev_btn", None)
        next_btn = getattr(self, "_observer_next_btn", None)
        if spin is None:
            return

        sequence_slots = self._observer_sequence_slots()
        if not sequence_slots:
            spin.blockSignals(True)
            spin.setEnabled(False)
            spin.setRange(0, 0)
            spin.setValue(0)
            spin.blockSignals(False)
            if prev_btn is not None:
                prev_btn.setEnabled(False)
            if next_btn is not None:
                next_btn.setEnabled(False)
            return

        frame_ranges = []
        for slot in sequence_slots:
            panel = slot["panel"]
            frames = panel.available_sequence_frames()
            if not frames:
                continue
            frame_ranges.append((int(frames[0]), int(frames[-1])))
        if not frame_ranges:
            return

        frame_min = min(start for start, _end in frame_ranges)
        frame_max = max(end for _start, end in frame_ranges)

        current_value = int(preferred_frame if preferred_frame is not None else spin.value())
        current_value = max(frame_min, min(frame_max, current_value))
        spin.blockSignals(True)
        spin.setEnabled(True)
        spin.setRange(frame_min, frame_max)
        spin.setValue(current_value)
        spin.blockSignals(False)
        if prev_btn is not None:
            prev_btn.setEnabled(True)
        if next_btn is not None:
            next_btn.setEnabled(True)
        self._refresh_observer_timeline()

    def _observer_dataset_key(self, side: str) -> str:
        return self._preferred_observer_dataset_key(side)

    def _observer_selected_frames_for_dataset(self, dataset_key: str) -> set[int]:
        dataset_key = str(dataset_key or "").strip()
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not dataset_key or not selected_path_ids:
            return set()
        local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, selected_path_ids)
        if not local_ids:
            return set()
        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset is None:
            return set()
        try:
            return set(int(frame) for frame in dataset.db.get_frame_ids_for_path_ids(local_ids))
        except Exception:
            return set()

    def _preferred_observer_frame_from_selection(self) -> int | None:
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not selected_path_ids:
            return None
        frames_by_side: list[set[int]] = []
        for slot in self._protein_observer_slots():
            dataset_key = str(slot.get("dataset_key") or "")
            if not dataset_key:
                continue
            frames = self._observer_selected_frames_for_dataset(dataset_key)
            if frames:
                frames_by_side.append(frames)
        if not frames_by_side:
            return None
        combined = set().union(*frames_by_side)
        if len(combined) == 1:
            return int(next(iter(combined)))
        return None

    def _sync_observer_frame_to_selection(self) -> None:
        preferred_frame = self._preferred_observer_frame_from_selection()
        if preferred_frame is None:
            return
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        frame_min = int(spin.minimum())
        frame_max = int(spin.maximum())
        target_frame = max(frame_min, min(frame_max, int(preferred_frame)))
        if int(spin.value()) == target_frame:
            return
        spin.setValue(target_frame)

    def _observer_path_points_for_frame(
        self,
        dataset_key: str,
        frame_number: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        dataset_key = str(dataset_key or "").strip()
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not dataset_key or not selected_path_ids:
            return None, None, None
        local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, selected_path_ids)
        if not local_ids:
            return None, None, None
        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset is None:
            return None, None, None
        try:
            rows = dataset.db.get_path_point_rows(
                local_ids,
                frame_id=int(frame_number),
                original_coords=True,
            )
        except Exception:
            return None, None, None
        if not rows:
            return None, None, None
        path_ids = np.asarray(
            [int(row["path_id"]) for row in rows],
            dtype=np.int64,
        )
        points = np.asarray(
            [(float(row["x"]), float(row["y"]), float(row["z"])) for row in rows],
            dtype=np.float64,
        )
        radii = np.asarray(
            [max(0.1, float(row["radius"] or 0.1)) for row in rows],
            dtype=np.float64,
        )
        return points, radii, path_ids

    def _observer_path_overlay_color(self, dataset_key: str) -> str:
        key = str(dataset_key or "").strip()
        if not key:
            return self.state.get_auto_color(0)
        return str(
            self.state.dataset_colors.get(
                key,
                self.state.get_auto_color(0),
            )
        )

    def _refresh_observer_path_overlays(self, frame_number: int | None = None) -> None:
        frame = int(frame_number if frame_number is not None else self._observer_frame_value())
        slots = self._protein_observer_slots()
        focus_interval = getattr(self, "_observer_path_focus_interval", None)
        signature = (
            frame,
            tuple(
                (str(slot.get("dataset_key") or ""), self._observer_path_overlay_color(str(slot.get("dataset_key") or "")))
                for slot in slots
            ),
            tuple(sorted(int(path_id) for path_id in self.state.effective_path_ids)),
            tuple(float(value) for value in focus_interval) if focus_interval is not None else None,
        )
        if signature == self._last_observer_overlay_signature:
            return
        self._last_observer_overlay_signature = signature
        for slot in slots:
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is None or not hasattr(panel, "set_path_overlay"):
                continue
            dataset_key = str(slot.get("dataset_key") or "")
            points, radii, path_ids = self._observer_path_points_for_frame(
                dataset_key,
                frame,
            )
            if points is None:
                panel.clear_path_overlay()
            else:
                panel.set_path_overlay(
                    points,
                    radii,
                    path_ids,
                    color=self._observer_path_overlay_color(dataset_key),
                    focus_interval=focus_interval,
                )

    def _refresh_observer_timeline(self) -> None:
        timeline = getattr(self, "_observer_timeline", None)
        if timeline is None:
            return
        shared_current_frame = self._observer_frame_value()
        rows = []
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            dataset_key = str(slot.get("dataset_key") or "")
            frames = (
                list(panel.available_sequence_frames())
                if panel is not None and hasattr(panel, "available_sequence_frames")
                else []
            )
            rows.append({
                "key": dataset_key,
                "label": self._dataset_display_name(dataset_key, str(slot.get("fallback_title") or dataset_key)),
                "frames": frames,
                "highlighted_frames": self._observer_selected_frames_for_dataset(dataset_key),
                "current_frame": shared_current_frame,
                "color": self._observer_path_overlay_color(dataset_key),
            })
        timeline.set_dataset_timelines(rows)

    def _on_observer_timeline_zoom_changed(self, value: int) -> None:
        timeline = getattr(self, "_observer_timeline", None)
        if timeline is None:
            return
        timeline.set_zoom_percent(int(value))

    def _sync_observer_timeline_zoom_slider(self, value: int) -> None:
        slider = getattr(self, "_observer_timeline_zoom_slider", None)
        if slider is None:
            return
        value = max(int(slider.minimum()), min(int(slider.maximum()), int(value)))
        if int(slider.value()) == value:
            return
        was_blocked = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(was_blocked)

    def _on_observer_timeline_frame_clicked(self, side: str, frame: int) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        frame = int(frame)
        if int(spin.value()) != frame:
            spin.setValue(frame)
            self._refresh_observer_timeline()
            return
        self._apply_observer_frame_to_sequence_slots(
            frame,
            self._parse_observer_residue_ids(self._observer_residue_input.text()),
            focus_loaded=True,
        )

    def _apply_observer_frame_to_sequence_slots(
        self,
        frame_number: int,
        residue_ids,
        *,
        focus_loaded: bool,
    ) -> list[str]:
        messages: list[str] = []
        for slot in self._observer_sequence_slots():
            panel = slot.get("panel")
            if panel is None:
                continue
            panel.set_highlight_residues(
                residue_ids,
                focus_loaded=False,
                refresh_sequence_cache=False,
            )
            result = panel.set_sequence_frame(frame_number)
            if not bool(result.get("ok")):
                messages.append(f"{slot.get('fallback_title', 'Observer')}: {result.get('error', 'load_failed')}")
                continue
            panel.set_highlight_residues(residue_ids, focus_loaded=focus_loaded)
            slot["source_name"] = os.path.basename(panel.sequence_folder()) or str(slot.get("fallback_title") or "Observer")
            self._update_observer_slot_title(slot)
            messages.append(
                f"{slot.get('fallback_title', 'Observer')} frame {result.get('frame', frame_number)}"
            )
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays(frame_number)
        return messages

    def _choose_observer_sequence_folder(self, slot_key: str) -> None:
        self._refresh_observer_sequence_source_combo(slot_key)

    def _on_observer_frame_changed(self, value: int) -> None:
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        messages = self._apply_observer_frame_to_sequence_slots(
            int(value),
            residue_ids,
            focus_loaded=False,
        )
        if messages:
            self._protein_observer_debug_label.setText(" | ".join(messages))

    def _step_observer_frame(self, delta: int) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        spin.setValue(spin.value() + int(delta))

    def _dataset_display_name(self, dataset_key: str, fallback: str) -> str:
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or "").strip())
        if dataset is None:
            return str(fallback)
        return str(dataset.prefix or dataset.name or fallback)

    def _observer_alignment_metadata(self, dataset_key: str) -> dict | None:
        """Read a pipeline transform only when the loaded DB is aligned."""
        normalized_key = str(dataset_key or "").strip()
        binding = getattr(self.db, "_datasets_by_key", {}).get(normalized_key)
        if binding is None:
            return None
        binding_folder = os.path.abspath(str(getattr(binding, "folder", "") or ""))
        cache_key = (normalized_key, os.path.normcase(binding_folder))
        if cache_key in self._observer_alignment_metadata_cache:
            return self._observer_alignment_metadata_cache[cache_key]

        database = getattr(binding, "db", None)
        get_meta = getattr(database, "get_meta", None)
        if not callable(get_meta):
            self._observer_alignment_metadata_cache[cache_key] = None
            return None
        try:
            status = str(get_meta("alignment_status", "") or "").strip().lower()
            raw_transform = str(get_meta("alignment_transform_json", "") or "").strip()
            mobile_pdb = str(get_meta("alignment_mobile_pdb", "") or "").strip()
        except Exception:
            self._observer_alignment_metadata_cache[cache_key] = None
            return None
        if status not in {"reference", "aligned_to_reference"} or not raw_transform:
            self._observer_alignment_metadata_cache[cache_key] = None
            return None

        try:
            transform = json.loads(raw_transform)
            rotation = np.asarray(transform.get("rotation"), dtype=np.float64).reshape(3, 3)
            translation = np.asarray(transform.get("translation"), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
                raise ValueError("non-finite pipeline transform")
        except (TypeError, ValueError, json.JSONDecodeError):
            self._observer_alignment_metadata_cache[cache_key] = None
            return None

        mobile_pdb = os.path.abspath(mobile_pdb) if mobile_pdb else ""
        metadata = {
            "status": status,
            "rotation": rotation,
            "translation": translation,
            "mobile_pdb": mobile_pdb,
            "aligned_output_folder": binding_folder,
        }
        self._observer_alignment_metadata_cache[cache_key] = metadata
        return metadata

    def _observer_pipeline_display_transform(
        self,
        dataset_key: str,
        frame_number,
        pdb_path: str,
    ) -> dict | None:
        """Return the exact pipeline coordinate transform without editing PDBs."""
        metadata = self._observer_alignment_metadata(dataset_key)
        if not metadata:
            return None
        source_path = os.path.abspath(str(pdb_path or "").strip()) if str(pdb_path or "").strip() else ""
        if not source_path or not os.path.isfile(source_path):
            return None

        # The representative_frame.pdb written inside an aligned output folder
        # has already been transformed by stage 06 and must not be transformed twice.
        aligned_folder = str(metadata.get("aligned_output_folder") or "")
        try:
            inside_aligned_output = (
                aligned_folder
                and os.path.normcase(os.path.commonpath((source_path, aligned_folder)))
                == os.path.normcase(aligned_folder)
            )
        except ValueError:
            inside_aligned_output = False
        if inside_aligned_output:
            return None

        representative_rotation = np.asarray(metadata["rotation"], dtype=np.float64).reshape(3, 3)
        representative_translation = np.asarray(metadata["translation"], dtype=np.float64).reshape(3)
        mobile_pdb = str(metadata.get("mobile_pdb") or "")
        if mobile_pdb and os.path.normcase(source_path) == os.path.normcase(mobile_pdb):
            return {
                "rotation": representative_rotation,
                "translation": representative_translation,
                "reason": "pipeline_representative_to_reference",
            }
        if not mobile_pdb or not os.path.isfile(mobile_pdb):
            aligned_representative = os.path.join(
                aligned_folder, "representative_frame.pdb"
            )
            if os.path.isfile(aligned_representative):
                try:
                    rotation, translation, rmsd, matched_count = (
                        kabsch_pipeline_row_transform(
                            read_pipeline_backbone_map(source_path),
                            read_pipeline_backbone_map(aligned_representative),
                        )
                    )
                    return {
                        "rotation": rotation,
                        "translation": translation,
                        "rmsd": rmsd,
                        "matched_count": matched_count,
                        "frame": int(frame_number) if frame_number is not None else None,
                        "reason": "pipeline_frame_direct_to_aligned_representative",
                    }
                except (OSError, ValueError):
                    pass
            return None

        try:
            modified_ns = int(os.stat(source_path).st_mtime_ns)
        except OSError:
            modified_ns = 0
        cache_key = (
            str(dataset_key or "").strip(),
            os.path.normcase(source_path),
            modified_ns,
        )
        if cache_key in self._observer_frame_transform_cache:
            return self._observer_frame_transform_cache[cache_key]

        try:
            frame_backbone = read_pipeline_backbone_map(source_path)
            representative_backbone = metadata.get("mobile_backbone")
            if not isinstance(representative_backbone, dict):
                representative_backbone = read_pipeline_backbone_map(mobile_pdb)
                metadata["mobile_backbone"] = representative_backbone
            frame_rotation, frame_translation, frame_rmsd, matched_count = (
                kabsch_pipeline_row_transform(frame_backbone, representative_backbone)
            )
            rotation, translation = compose_pipeline_row_transforms(
                frame_rotation,
                frame_translation,
                representative_rotation,
                representative_translation,
            )
            result = {
                "rotation": rotation,
                "translation": translation,
                "rmsd": frame_rmsd,
                "matched_count": matched_count,
                "frame": int(frame_number) if frame_number is not None else None,
                "reason": "pipeline_frame_to_representative_to_reference",
            }
        except (OSError, ValueError):
            result = None
        self._observer_frame_transform_cache[cache_key] = result
        return result

    def _dataset_representative_pdb_path(self, dataset_key: str) -> str:
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or "").strip())
        search_roots: list[str] = []
        for candidate in (
            getattr(dataset, "folder", ""),
            os.path.dirname(str(getattr(dataset, "path", "") or "").strip()),
        ):
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and normalized not in search_roots and os.path.isdir(normalized):
                search_roots.append(normalized)

        for root in search_roots:
            direct = os.path.join(root, "representative_frame.pdb")
            if os.path.exists(direct):
                return direct

            for walk_root, _dirs, files in os.walk(root):
                if "representative_frame.pdb" in files:
                    return os.path.abspath(os.path.join(walk_root, "representative_frame.pdb"))

        for root in list(search_roots):
            current = root
            for _ in range(4):
                parent = os.path.dirname(current)
                if not parent or parent == current:
                    break
                direct = os.path.join(parent, "representative_frame.pdb")
                if os.path.exists(direct):
                    return direct
                current = parent

        dataset_name = str(getattr(dataset, "name", "") or "")
        dataset_folder = str(getattr(dataset, "folder", "") or "")
        fallback_tokens = [
            token for token in (
                dataset_name.split("_")[0] if dataset_name else "",
                os.path.basename(dataset_folder).split("_")[0] if dataset_folder else "",
            )
            if token
        ]
        database_root = os.path.abspath(os.path.join(os.getcwd(), "database"))
        for token in fallback_tokens:
            candidate = os.path.join(database_root, token, "representative_frame.pdb")
            if os.path.exists(candidate):
                return candidate
        return ""

    def _clear_protein_observer_slot(self, slot: dict | None, title: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        panel = slot.get("panel")
        if panel is not None:
            panel.clear_protein()
        self._set_protein_observer_slot_title(slot, title)

    def _load_combination_into_observer_panel(
        self,
        slot: dict | None,
        dataset_key: str,
        residue_ids,
        *,
        present: bool,
        title_prefix: str = "",
        label_shape_color: str = "#FFFFFF",
    ) -> None:
        if not isinstance(slot, dict):
            return
        requested_dataset_key = str(dataset_key or "").strip()
        current_dataset_key = str(slot.get("dataset_key") or "").strip()
        slot["dataset_key"] = requested_dataset_key
        default_title = str(
            title_prefix or slot.get("role_title") or slot.get("fallback_title") or "Observer"
        )
        dataset_label = self._dataset_display_name(requested_dataset_key, default_title) if requested_dataset_key else default_title
        panel = slot.get("panel")
        if panel is not None and hasattr(panel, "set_highlight_label_specs"):
            panel.set_highlight_label_specs({}, render=False)
        if panel is not None and hasattr(panel, "set_highlight_label_colors"):
            panel.set_highlight_label_colors(
                text_color="#FFFFFF",
                shape_color=label_shape_color,
                render=False,
            )
        if panel is not None and hasattr(panel, "set_display_transform_resolver"):
            panel.set_display_transform_resolver(
                lambda frame_number, pdb_path, key=requested_dataset_key: (
                    self._observer_pipeline_display_transform(key, frame_number, pdb_path)
                )
            )
        if (
            panel is not None
            and hasattr(panel, "has_sequence")
            and panel.has_sequence()
            and current_dataset_key == requested_dataset_key
        ):
            target_frame = self._observer_frame_value()
            current_frame = panel.current_sequence_frame()
            if current_frame != target_frame:
                result = panel.set_sequence_frame(target_frame)
                if not bool(result.get("ok")):
                    self._clear_protein_observer_slot(slot, f"{default_title} | {dataset_label}")
                    return
            wanted_highlights = tuple(sorted(set(residue_ids if present else ())))
            current_highlights = (
                panel.highlight_residue_ids()
                if hasattr(panel, "highlight_residue_ids")
                else None
            )
            if current_highlights != wanted_highlights:
                panel.set_highlight_residues(wanted_highlights, focus_loaded=True)
            slot["source_name"] = os.path.basename(panel.sequence_folder()) or dataset_label
            suffix = "" if present else " | absent"
            self._set_protein_observer_slot_title(
                slot,
                f"{default_title} | {dataset_label}{suffix} | {slot['source_name']} | frame {self._observer_frame_value()}",
            )
            return

        title = f"{default_title} | {dataset_label}"
        if not requested_dataset_key:
            self._clear_protein_observer_slot(slot, title)
            return
        if panel is None:
            return

        source_label, folder = self._observer_resolve_sequence_source(requested_dataset_key)
        if folder:
            result = panel.load_pdb_sequence_folder(folder)
            if bool(result.get("ok")):
                self._refresh_observer_frame_controls(
                    preferred_frame=self._observer_frame_value()
                    if self._observer_frame_value() is not None
                    else int(result.get("current_frame", 0) or 0)
                )
                panel.set_sequence_frame(self._observer_frame_value())
                panel.set_highlight_residues(residue_ids if present else (), focus_loaded=True)
                suffix = "" if present else " | no pair"
                slot["source_name"] = source_label or os.path.basename(folder) or dataset_label
                self._set_protein_observer_slot_title(
                    slot,
                    f"{title}{suffix} | {slot['source_name']} | {int(result.get('frame_count', 0) or 0)} frames",
                )
                self._refresh_observer_timeline()
                self._refresh_observer_path_overlays()
                return

        pdb_path = self._dataset_representative_pdb_path(requested_dataset_key)
        if not pdb_path:
            self._clear_protein_observer_slot(slot, title)
            return

        panel.load_pdb_file(pdb_path, first_model_only=True)
        panel.set_highlight_residues(residue_ids if present else (), focus_loaded=True)
        suffix = "" if present else " | no pair"
        slot["source_name"] = os.path.basename(os.path.dirname(pdb_path)) or dataset_label
        self._set_protein_observer_slot_title(
            slot,
            f"{title}{suffix} | {slot['source_name']}",
        )

    def _update_combination_observer_views(self, payload: dict | None) -> None:
        payload = payload or {}
        left_residue_ids = tuple(
            int(rid) for rid in payload.get("left_residue_ids", payload.get("residue_ids", ())) if int(rid) > 0
        )
        right_residue_ids = tuple(
            int(rid) for rid in payload.get("right_residue_ids", payload.get("residue_ids", ())) if int(rid) > 0
        )
        if len(left_residue_ids) < 1 and len(right_residue_ids) < 1:
            return
        left_dataset_key = str(payload.get("left_dataset_key", "") or "")
        right_dataset_key = str(payload.get("right_dataset_key", "") or "")
        selected_keys = self._selected_observer_dataset_keys()
        for dataset_key in (left_dataset_key, right_dataset_key):
            if dataset_key and dataset_key not in selected_keys:
                selected_keys.append(dataset_key)
        self._set_checked_observer_datasets(selected_keys)
        merged_residue_ids = tuple(dict.fromkeys((*left_residue_ids, *right_residue_ids)))
        residue_ids_by_dataset: dict[str, tuple[int, ...]] = {}
        for slot in self._protein_observer_slots():
            dataset_key = str(slot.get("dataset_key") or "")
            if dataset_key == left_dataset_key:
                residue_ids = left_residue_ids
                present = bool(payload.get("present_left")) and bool(residue_ids)
                title = str(payload.get("left_title", "") or "Reference")
                label_color = "#2563EB"
            elif dataset_key == right_dataset_key:
                residue_ids = right_residue_ids
                present = bool(payload.get("present_right")) and bool(residue_ids)
                title = str(payload.get("right_title", "") or "Target")
                label_color = "#EA580C"
            else:
                residue_ids = merged_residue_ids
                present = bool(residue_ids)
                title = self._dataset_display_name(dataset_key, dataset_key)
                label_color = self._observer_path_overlay_color(dataset_key)
            residue_ids_by_dataset[dataset_key] = tuple(residue_ids)
            slot["role_title"] = title
            self._load_combination_into_observer_panel(
                slot,
                dataset_key,
                residue_ids,
                present=present,
                title_prefix=title,
                label_shape_color=label_color,
            )
        self._apply_observer_residue_comparison_visuals(
            merged_residue_ids,
            residue_ids_by_dataset=residue_ids_by_dataset,
            forced_changed_ids=tuple(payload.get("changed_residue_ids", ()) or ()),
            comparison_label="Path residue change",
        )
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays()

    @staticmethod
    def _parse_observer_residue_ids(text: str) -> tuple[int, ...]:
        residue_ids: list[int] = []
        for token in re.findall(r"\d+", str(text or "")):
            try:
                residue_id = int(token)
            except ValueError:
                continue
            if residue_id > 0 and residue_id not in residue_ids:
                residue_ids.append(residue_id)
        return tuple(residue_ids)

    _OBSERVER_RESIDUE_COLORS = (
        "#2563EB", "#7C3AED", "#059669", "#D97706",
        "#DB2777", "#0891B2", "#65A30D", "#9333EA",
    )

    def _refresh_observer_change_event_selector(self, _index=None) -> None:
        combo = getattr(self, "_observer_change_event_combo", None)
        button = getattr(self, "_observer_show_change_btn", None)
        if combo is None:
            return
        selected_payload = combo.currentData()
        selected_id = str(selected_payload.get("event_id") or "") if isinstance(selected_payload, dict) else ""
        events = [dict(row) for row in (getattr(self, "_observer_change_events", []) or [])]
        combo.blockSignals(True)
        combo.clear()
        if not events:
            combo.addItem("No detected change events", None)
        else:
            for row in events:
                combo.addItem(str(row.get("display_label") or "Residue change"), row)
            selected_index = next(
                (
                    index
                    for index, row in enumerate(events)
                    if str(row.get("event_id") or "") == selected_id
                ),
                0,
            )
            combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)
        combo.setEnabled(bool(events))
        if button is not None:
            button.setEnabled(bool(events))
        explanation = getattr(self, "_observer_change_explanation", None)
        if explanation is not None and not events:
            explanation.setText(
                "No path-position changes are available. Select a mapped baseline and mutant cluster first."
            )

    def _show_selected_observer_change_event(self) -> None:
        combo = getattr(self, "_observer_change_event_combo", None)
        row = combo.currentData() if combo is not None else None
        if not isinstance(row, dict):
            self._statusbar.showMessage("No residue-change event is selected", 3000)
            return
        self._on_dataset_compare_motif_selected(dict(row))
        explanation = getattr(self, "_observer_change_explanation", None)
        if explanation is not None:
            explanation.setText(str(row.get("impact_text") or row.get("display_label") or "Residue change"))

    def _observer_residue_names_for_slot(self, slot: dict, residue_ids) -> dict[int, str]:
        wanted = tuple(int(rid) for rid in residue_ids if int(rid) > 0)
        dataset_key = str(slot.get("dataset_key") or "")
        binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        database = getattr(binding, "db", None)
        stored_names = getattr(database, "_residue_name_map", {}) or {}
        result = {
            int(rid): str(stored_names.get(int(rid), "") or "").strip().upper()
            for rid in wanted
        }
        panel = slot.get("panel") if isinstance(slot, dict) else None
        if panel is not None and hasattr(panel, "residue_name_map"):
            try:
                displayed_names = panel.residue_name_map(wanted)
            except Exception:
                displayed_names = {}
            for rid, name in dict(displayed_names or {}).items():
                if str(name or "").strip():
                    result[int(rid)] = str(name).strip().upper()
        return result

    def _observer_residue_centroids_for_dataset(
        self,
        dataset_key: str,
        residue_ids,
    ) -> dict[int, np.ndarray]:
        """Read aligned C-alpha anchors without requiring Observer to be open."""
        wanted = tuple(dict.fromkeys(int(rid) for rid in residue_ids if int(rid) > 0))
        if not wanted:
            return {}
        for slot in self._protein_observer_slots():
            if str(slot.get("dataset_key") or "") != str(dataset_key or ""):
                continue
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is None or not hasattr(panel, "residue_centroid_map"):
                continue
            try:
                displayed = {
                    int(rid): np.asarray(point, dtype=np.float64)
                    for rid, point in dict(panel.residue_centroid_map(wanted) or {}).items()
                }
                if displayed:
                    return displayed
            except Exception:
                pass

        # Evidence must not depend on a view having been opened first. The
        # representative PDB normally lives in pipeline-aligned coordinates;
        # an external mobile PDB receives the same display-only transform as
        # Residue Observer.
        pdb_path = self._dataset_representative_pdb_path(dataset_key)
        if not pdb_path or not os.path.isfile(pdb_path):
            return {}
        wanted_set = set(wanted)
        result: dict[int, np.ndarray] = {}
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line[:6].strip() not in {"ATOM", "HETATM"}:
                        continue
                    if line[12:16].strip().upper() != "CA":
                        continue
                    try:
                        residue_id = int(line[22:26])
                        point = np.asarray(
                            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            dtype=np.float64,
                        )
                    except (ValueError, IndexError):
                        continue
                    if residue_id in wanted_set:
                        result.setdefault(residue_id, point)
        except OSError:
            return {}
        transform = self._observer_pipeline_display_transform(dataset_key, None, pdb_path)
        if transform and result:
            rotation = np.asarray(transform["rotation"], dtype=np.float64).reshape(3, 3)
            translation = np.asarray(transform["translation"], dtype=np.float64).reshape(3)
            result = {
                residue_id: np.asarray(point, dtype=np.float64) @ rotation + translation
                for residue_id, point in result.items()
            }
        return result

    def _dataset_residue_sequence_offset(self, dataset_key: str) -> int:
        """Infer the database-to-PDB residue-number offset from residue names."""
        binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or ""))
        database_names = getattr(getattr(binding, "db", None), "_residue_name_map", {}) or {}
        pdb_path = self._dataset_representative_pdb_path(dataset_key)
        pdb_names = load_residue_name_map_from_pdb(pdb_path)
        if not database_names or not pdb_names:
            # Preserve the legacy convention when no structure can establish
            # an explicit numbering relationship.
            return 1
        ranked = []
        for offset in range(-2, 3):
            comparable = 0
            matches = 0
            for residue_id, residue_name in database_names.items():
                pdb_name = pdb_names.get(int(residue_id) + offset)
                if not pdb_name:
                    continue
                comparable += 1
                matches += int(str(pdb_name).upper() == str(residue_name).upper())
            ranked.append((matches, comparable, -abs(offset), int(offset)))
        best = max(ranked)
        return int(best[-1]) if best[1] > 0 else 1

    def _dataset_compare_allosteric_profile(
        self,
        reference_key: str,
        target_key: str,
        profile: dict,
        reference_center_points,
        target_center_points,
    ) -> dict:
        """Connect explicit substitutions to path-local and bottleneck changes."""
        reference_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        reference_names = getattr(getattr(reference_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        sequence_offset = self._dataset_residue_sequence_offset(reference_key)
        substitutions = detect_residue_substitutions(
            reference_names,
            target_names,
            sequence_offset=sequence_offset,
        )
        sequence_ids = tuple(int(row.get("sequence_id", 0) or 0) for row in substitutions)
        reference_centroids = self._observer_residue_centroids_for_dataset(reference_key, sequence_ids)
        target_centroids = self._observer_residue_centroids_for_dataset(target_key, sequence_ids)
        threshold = (
            self._dataset_compare_panel.remote_distance_threshold()
            if hasattr(self._dataset_compare_panel, "remote_distance_threshold")
            else 10.0
        )
        annotated = annotate_remote_hotspots(
            profile,
            substitutions,
            reference_center_points=reference_center_points,
            target_center_points=target_center_points,
            reference_points_by_sequence=reference_centroids,
            target_points_by_sequence=target_centroids,
            remote_distance_threshold=threshold,
        )
        annotated["sequence_offset"] = int(sequence_offset)
        if substitutions:
            annotated["perturbation_type"] = "sequence substitution"
        else:
            annotated["perturbation_type"] = "ensemble state contrast"
            annotated["perturbation_label"] = (
                f"{self._dataset_compare_label(reference_key)} → "
                f"{self._dataset_compare_label(target_key)}"
            )
        enriched_hotspots = []
        for hotspot in annotated.get("hotspots", []) or []:
            row = dict(hotspot or {})
            if row.get("mutation_distance") is not None:
                region_distance = float(row["mutation_distance"])
                active_site_distance = row.get("active_site_distance")
                distance_text = f"{row.get('nearest_mutation_label', 'mutation')} · region {region_distance:.1f} Å"
                if active_site_distance is not None:
                    distance_text += f" · active site {float(active_site_distance):.1f} Å"
                # Keep the 3D label compact. Both distances remain available
                # in the comparison detail and Evidence views.
                row["scene_label"] = str(row.get("scene_label") or row.get("display_label") or "R")
                row["display_label"] = str(row.get("display_label") or row.get("region_id") or "R") + f" | {distance_text}"
            enriched_hotspots.append(row)
        annotated["hotspots"] = enriched_hotspots
        return annotated

    def _observer_active_path_membership(self, residue_ids) -> dict[str, dict]:
        """Measure selected residue/group contact within the active mapped clusters."""
        selected = tuple(dict.fromkeys(int(rid) for rid in residue_ids if int(rid) > 0))
        active = dict(getattr(self, "_dataset_compare_active", {}) or {})
        if not selected or not active:
            return {}
        dataset_clusters = (
            (str(active.get("reference_key") or ""), active.get("reference_cluster")),
            (str(active.get("target_key") or ""), active.get("target_cluster")),
        )
        result: dict[str, dict] = {}
        for dataset_key, cluster_id in dataset_clusters:
            binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
            connection = getattr(getattr(binding, "db", None), "conn", None)
            if not dataset_key or cluster_id is None or connection is None:
                continue
            try:
                rows = connection.execute(
                    "SELECT p.id AS path_id, pp.res_1, pp.res_2, pp.res_3, pp.res_4 "
                    "FROM paths p JOIN path_points pp ON pp.path_id = p.id "
                    "WHERE p.cluster_id = ? ORDER BY p.id, pp.seq",
                    (int(cluster_id),),
                ).fetchall()
                total_row = connection.execute(
                    "SELECT COUNT(*) FROM paths WHERE cluster_id = ?",
                    (int(cluster_id),),
                ).fetchone()
            except Exception:
                continue
            residue_paths: dict[int, set[int]] = {rid: set() for rid in selected}
            group_paths: set[int] = set()
            selected_set = set(selected)
            for row in rows:
                path_id = int(row[0])
                point_residues = {
                    int(row[index] or 0)
                    for index in range(1, 5)
                    if int(row[index] or 0) > 0
                }
                for rid in selected:
                    if rid in point_residues:
                        residue_paths[rid].add(path_id)
                if selected_set.issubset(point_residues):
                    group_paths.add(path_id)
            total = int(total_row[0] if total_row else 0)
            result[dataset_key] = {
                "cluster_id": int(cluster_id),
                "path_count": total,
                "residue_counts": {rid: len(paths) for rid, paths in residue_paths.items()},
                "group_count": len(group_paths),
            }
        return result

    def _apply_observer_residue_comparison_visuals(
        self,
        residue_ids,
        *,
        residue_ids_by_dataset: dict[str, tuple[int, ...]] | None = None,
        path_membership_by_dataset: dict[str, dict] | None = None,
        forced_changed_ids=(),
        comparison_label: str = "Manual",
    ) -> None:
        """Show residue identities and substitutions directly in every 3D observer."""
        ordered_ids = tuple(dict.fromkeys(int(rid) for rid in residue_ids if int(rid) > 0))
        if not ordered_ids:
            return
        slots = self._protein_observer_slots()
        if not slots:
            return
        identities: dict[str, dict[int, str]] = {}
        relevant_by_dataset: dict[str, set[int]] = {}
        for slot in slots:
            dataset_key = str(slot.get("dataset_key") or "")
            relevant = set(
                int(rid)
                for rid in (
                    (residue_ids_by_dataset or {}).get(dataset_key, ordered_ids)
                )
                if int(rid) > 0
            )
            relevant_by_dataset[dataset_key] = relevant
            identities[dataset_key] = self._observer_residue_names_for_slot(slot, ordered_ids)

        changed_ids: set[int] = {
            int(rid) for rid in (forced_changed_ids or ()) if int(rid) > 0
        }
        identity_variants: dict[int, list[str]] = {}
        membership_variants: dict[int, set[bool]] = {rid: set() for rid in ordered_ids}
        for rid in ordered_ids:
            values = []
            missing = False
            for slot in slots:
                dataset_key = str(slot.get("dataset_key") or "")
                if rid not in relevant_by_dataset.get(dataset_key, set()):
                    missing = True
                    continue
                name = identities.get(dataset_key, {}).get(rid, "")
                if name and name not in values:
                    values.append(name)
                membership = (path_membership_by_dataset or {}).get(dataset_key)
                if membership is not None:
                    count = int((membership.get("residue_counts") or {}).get(rid, 0) or 0)
                    membership_variants[rid].add(count > 0)
            identity_variants[rid] = values
            if len(values) > 1 or missing or len(membership_variants[rid]) > 1:
                changed_ids.add(rid)

        for slot in slots:
            dataset_key = str(slot.get("dataset_key") or "")
            relevant = relevant_by_dataset.get(dataset_key, set())
            names = identities.get(dataset_key, {})
            specs: dict[int, dict] = {}
            badges = []
            membership = (path_membership_by_dataset or {}).get(dataset_key)
            if membership is not None:
                group_count = int(membership.get("group_count", 0) or 0)
                total_paths = int(membership.get("path_count", 0) or 0)
                group_state = "on path" if group_count > 0 else "off path"
                group_color = "#166534" if group_count > 0 else "#991B1B"
                badges.append(
                    f'<span style="color:#ffffff;background-color:{group_color};'
                    f'font-weight:700;padding:2px 5px;">Group {group_state} {group_count}/{total_paths}</span>'
                )
            for index, rid in enumerate(ordered_ids):
                color = self._OBSERVER_RESIDUE_COLORS[index % len(self._OBSERVER_RESIDUE_COLORS)]
                present = rid in relevant
                residue_name = names.get(rid, "") if present else ""
                identity = residue_name or ("absent" if not present else "UNK")
                changed = rid in changed_ids
                prefix = "Δ " if changed else ""
                label_text = f"{prefix}{rid} {identity}"
                if present:
                    specs[rid] = {
                        "text": label_text,
                        "text_color": "#FFFFFF",
                        # Position colors remain identical across datasets; Δ
                        # marks identity/presence changes without changing the
                        # visual identity of a selected residue.
                        "shape_color": color,
                    }
                path_suffix = ""
                if membership is not None:
                    contact_count = int((membership.get("residue_counts") or {}).get(rid, 0) or 0)
                    path_suffix = " · path" if contact_count > 0 else " · off path"
                badge_text = html.escape(label_text + path_suffix)
                emphasis = "font-weight:700;" if changed else "font-weight:600;"
                badges.append(
                    f'<span style="color:#ffffff;background-color:{color};'
                    f'{emphasis}padding:2px 5px;">{badge_text}</span>'
                )
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is not None and hasattr(panel, "set_highlight_label_specs"):
                panel.set_highlight_label_specs(specs)
            summary = slot.get("residue_summary") if isinstance(slot, dict) else None
            if summary is not None:
                summary.setText(" &nbsp; ".join(badges))
                summary.setToolTip("Δ marks a residue identity or presence difference across selected datasets")
                summary.setVisible(True)

        changed_text = []
        for rid in ordered_ids:
            variants = identity_variants.get(rid, [])
            if rid in changed_ids:
                reasons = []
                if len(variants) > 1:
                    reasons.append("/".join(variants))
                if len(membership_variants.get(rid, set())) > 1:
                    reasons.append("path contact changed")
                changed_text.append(f"{rid} ({'; '.join(reasons) or 'presence change'})")
        group_label = {1: "single residue", 2: "residue pair", 3: "residue triple", 4: "four-residue group"}.get(
            len(ordered_ids), f"{len(ordered_ids)}-residue group"
        )
        message = f"{str(comparison_label or 'Residue')} {group_label}: {', '.join(str(rid) for rid in ordered_ids)}"
        if changed_text:
            message += " | Changed: " + ", ".join(changed_text)
        else:
            message += " | identities are consistent across selected datasets"
        self._protein_observer_debug_label.setText(message)

    def _apply_observer_debug_selection(self) -> None:
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        if len(residue_ids) < 1:
            self._protein_observer_debug_label.setText("Enter 1–4 residue IDs, for example: 145 or 145,212")
            self._statusbar.showMessage("Residue comparison needs 1–4 residue IDs", 3000)
            return
        if len(residue_ids) > 4:
            self._protein_observer_debug_label.setText("Choose no more than four residues for a readable structural comparison.")
            self._statusbar.showMessage("Select 1–4 residues", 3000)
            return
        if not self._protein_observer_page_active():
            self._toggle_protein_observer_panel(True)
        slots = self._protein_observer_slots()
        if not slots:
            self._protein_observer_debug_label.setText("Select at least one MD dataset before comparing residues.")
            return
        frame_number = self._observer_frame_value()
        for slot in self._protein_observer_slots():
            dataset_key = str(slot.get("dataset_key") or "")
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is None:
                continue
            if hasattr(panel, "has_sequence") and panel.has_sequence():
                if panel.current_sequence_frame() != frame_number:
                    panel.set_sequence_frame(frame_number)
                if (
                    not hasattr(panel, "highlight_residue_ids")
                    or panel.highlight_residue_ids() != tuple(sorted(set(residue_ids)))
                ):
                    panel.set_highlight_residues(residue_ids, focus_loaded=True)
            elif hasattr(panel, "has_loaded_protein") and panel.has_loaded_protein():
                panel.set_highlight_residues(residue_ids, focus_loaded=True)
            else:
                self._load_combination_into_observer_panel(
                    slot,
                    dataset_key,
                    residue_ids,
                    present=True,
                    title_prefix=self._dataset_display_name(dataset_key, dataset_key),
                    label_shape_color="#2563EB",
                )
        self._apply_observer_residue_comparison_visuals(
            residue_ids,
            path_membership_by_dataset=self._observer_active_path_membership(residue_ids),
        )
        self._statusbar.showMessage(
            f"Comparing {len(residue_ids)} residue(s) across {len(slots)} dataset(s)",
            4000,
        )

    @staticmethod
    def _observer_control_group(parent: QWidget, *, expanding: bool = False) -> tuple[QWidget, QHBoxLayout]:
        group = QWidget(parent)
        group.setSizePolicy(
            QSizePolicy.Expanding if expanding else QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(4)
        return group, group_layout

    def _refresh_observer_controls_layout(self) -> None:
        """Keep compact control groups evenly spaced and wrap only when needed."""
        host = getattr(self, "_observer_controls_widget", None)
        layout = getattr(self, "_observer_controls_layout", None)
        groups = list(getattr(self, "_observer_control_groups", []) or [])
        if host is None or layout is None or len(groups) != 5:
            return

        available = max(1, int(host.width()))
        widths = [max(group.minimumSizeHint().width(), group.sizeHint().width()) for group in groups]
        gap = int(layout.horizontalSpacing() if layout.horizontalSpacing() >= 0 else 6)
        one_row_width = sum(widths) + gap * (len(groups) - 1)
        two_row_width = max(
            widths[0] + widths[2] + gap,
            widths[1] + widths[3] + widths[4] + gap * 2,
        )
        if available >= one_row_width:
            mode = "single"
            placements = [(index, 0, index, 1, 1) for index in range(5)]
        elif available >= two_row_width:
            mode = "double"
            # Navigation and the expanding dataset selector form the primary
            # row; visual appearance controls remain together below it.
            placements = [
                (0, 0, 0, 1, 1),
                (2, 0, 1, 1, 2),
                (1, 1, 0, 1, 1),
                (3, 1, 1, 1, 1),
                (4, 1, 2, 1, 1),
            ]
        else:
            mode = "triple"
            placements = [
                (0, 0, 0, 1, 3),
                (2, 1, 0, 1, 3),
                (1, 2, 0, 1, 1),
                (3, 2, 1, 1, 1),
                (4, 2, 2, 1, 1),
            ]
        if mode == getattr(self, "_observer_controls_layout_mode", ""):
            return
        self._observer_controls_layout_mode = mode

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(host)
        visual_host = getattr(self, "_observer_visual_controls_widget", None)
        visual_layout = getattr(self, "_observer_visual_controls_layout", None)
        if visual_layout is not None:
            while visual_layout.count():
                item = visual_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(host)
        for column in range(5):
            layout.setColumnStretch(column, 0)
        if mode == "single":
            if visual_host is not None:
                visual_host.hide()
            for group_index, row, column, row_span, column_span in placements:
                layout.addWidget(groups[group_index], row, column, row_span, column_span)
            layout.setColumnStretch(2, 1)
            return

        if visual_host is not None and visual_layout is not None:
            for group_index in (1, 3, 4):
                visual_layout.addWidget(groups[group_index])
            visual_layout.addStretch(1)
            visual_host.show()
        if mode == "double":
            layout.addWidget(groups[0], 0, 0)
            layout.addWidget(groups[2], 0, 1)
            if visual_host is not None:
                layout.addWidget(visual_host, 1, 0, 1, 2)
            layout.setColumnStretch(1, 1)
        else:
            layout.addWidget(groups[0], 0, 0)
            layout.addWidget(groups[2], 1, 0)
            if visual_host is not None:
                layout.addWidget(visual_host, 2, 0)
            layout.setColumnStretch(0, 1)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._central_layout = layout

        # Main splitter: 3D viewer (top) + charts (bottom)
        self._main_splitter = QSplitter(Qt.Vertical)

        # Main visual area: switch between the tunnel/path view and Residue Observer.
        self._main_view_stack = QStackedWidget()
        self._viewer = TunnelViewer3D()
        self._main_visual_splitter = None  # compatibility alias for older sessions
        self._main_view_stack.addWidget(self._viewer)

        self._protein_observer_container = QFrame()
        self._protein_observer_container.setObjectName("proteinObserverContainer")
        self._protein_observer_container.setStyleSheet(
            "#proteinObserverContainer{background:#ffffff;border-left:1px solid #e5e7eb;}"
        )
        observer_layout = QVBoxLayout(self._protein_observer_container)
        observer_layout.setContentsMargins(8, 8, 8, 8)
        observer_layout.setSpacing(6)

        observer_header = QHBoxLayout()
        self._protein_observer_title = QLabel("Residue Observer")
        self._protein_observer_title.setStyleSheet("font-size:12px;font-weight:600;color:#1f2937;")
        observer_header.addWidget(self._protein_observer_title)
        observer_header.addStretch(1)
        self._observer_back_to_main_btn = QPushButton("Back to Main View")
        self._observer_back_to_main_btn.setFixedHeight(26)
        self._observer_back_to_main_btn.clicked.connect(
            lambda: self._toggle_protein_observer_panel(False)
        )
        observer_header.addWidget(self._observer_back_to_main_btn)
        observer_layout.addLayout(observer_header)

        # Keep path-position events available to linked Compare interactions,
        # but remove the redundant manual selector from the Observer UI.
        self._observer_change_controls = QWidget(self._protein_observer_container)
        observer_debug_row = QHBoxLayout(self._observer_change_controls)
        observer_debug_row.setContentsMargins(0, 0, 0, 0)
        observer_debug_row.setSpacing(6)
        self._observer_change_label = QLabel("Path Position Change")
        observer_debug_row.addWidget(self._observer_change_label)
        self._observer_change_type_combo = QComboBox()
        self._observer_change_type_combo.addItem("Path position", "path_position")
        self._observer_change_type_combo.setVisible(False)
        self._observer_change_event_combo = QComboBox()
        self._observer_change_event_combo.setMinimumContentsLength(34)
        self._observer_change_event_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._observer_change_event_combo.activated.connect(
            lambda _index: self._show_selected_observer_change_event()
        )
        observer_debug_row.addWidget(self._observer_change_event_combo, 1)
        self._observer_show_change_btn = QPushButton("Show Change")
        self._observer_show_change_btn.setFixedHeight(24)
        self._observer_show_change_btn.clicked.connect(self._show_selected_observer_change_event)
        observer_debug_row.addWidget(self._observer_show_change_btn)

        # Retained as an internal compatibility buffer for existing residue-
        # combination callbacks. Manual selection is now event-based.
        self._observer_residue_input = QLineEdit()
        self._observer_residue_input.returnPressed.connect(self._apply_observer_debug_selection)
        self._observer_residue_input.setVisible(False)
        self._observer_apply_btn = self._observer_show_change_btn
        self._observer_change_controls.setVisible(False)
        observer_layout.addWidget(self._observer_change_controls)

        # One compact row controls synchronized frames and rendering for every
        # horizontally arranged Observer dataset.
        self._observer_controls_widget = QWidget(self._protein_observer_container)
        self._observer_controls_layout = QGridLayout(self._observer_controls_widget)
        self._observer_controls_layout.setContentsMargins(0, 0, 0, 0)
        self._observer_controls_layout.setHorizontalSpacing(8)
        self._observer_controls_layout.setVerticalSpacing(4)

        navigation_group, navigation_layout = self._observer_control_group(
            self._observer_controls_widget
        )
        self._observer_prev_btn = QPushButton("Prev")
        self._observer_prev_btn.setFixedHeight(24)
        self._observer_prev_btn.setMinimumWidth(54)
        self._observer_prev_btn.setEnabled(False)
        self._observer_prev_btn.clicked.connect(lambda: self._step_observer_frame(-1))
        navigation_layout.addWidget(self._observer_prev_btn)
        navigation_layout.addWidget(QLabel("Frame"))
        self._observer_frame_spin = QSpinBox()
        self._observer_frame_spin.setEnabled(False)
        self._observer_frame_spin.setRange(0, 0)
        self._observer_frame_spin.setFixedHeight(24)
        self._observer_frame_spin.setMinimumWidth(78)
        self._observer_frame_spin.valueChanged.connect(self._on_observer_frame_changed)
        navigation_layout.addWidget(self._observer_frame_spin)
        self._observer_next_btn = QPushButton("Next")
        self._observer_next_btn.setFixedHeight(24)
        self._observer_next_btn.setMinimumWidth(54)
        self._observer_next_btn.setEnabled(False)
        self._observer_next_btn.clicked.connect(lambda: self._step_observer_frame(1))
        navigation_layout.addWidget(self._observer_next_btn)

        highlight_group, highlight_layout = self._observer_control_group(
            self._observer_controls_widget
        )
        highlight_layout.addWidget(QLabel("Highlight"))
        self._observer_highlight_style_combo = QComboBox()
        self._observer_highlight_style_combo.setFixedWidth(108)
        self._observer_highlight_style_combo.addItem("Ball-and-Stick", "ball_and_stick")
        self._observer_highlight_style_combo.addItem("Sticks", "sticks")
        self._observer_highlight_style_combo.addItem("Spheres", "spheres")
        self._observer_highlight_style_combo.currentIndexChanged.connect(
            lambda _idx: self._set_observer_highlight_style(
                str(self._observer_highlight_style_combo.currentData() or "ball_and_stick")
            )
        )
        highlight_layout.addWidget(self._observer_highlight_style_combo)

        dataset_group, dataset_layout = self._observer_control_group(
            self._observer_controls_widget,
            expanding=True,
        )
        dataset_layout.addWidget(QLabel("MD Datasets"))
        self._observer_dataset_multiselect = DatasetMultiSelectButton(
            self._protein_observer_container
        )
        self._observer_dataset_multiselect.setMinimumWidth(160)
        self._observer_dataset_multiselect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._observer_dataset_multiselect.selection_changed.connect(
            self._on_observer_dataset_multiselect_changed
        )
        dataset_layout.addWidget(self._observer_dataset_multiselect, 1)

        model_group, model_layout = self._observer_control_group(
            self._observer_controls_widget
        )
        model_layout.addWidget(QLabel("Model"))
        self._observer_model_style_combo = QComboBox()
        self._observer_model_style_combo.setFixedWidth(82)
        self._observer_model_style_combo.addItem("Cartoon", "cartoon")
        self._observer_model_style_combo.currentIndexChanged.connect(
            lambda _idx: self._set_observer_model_style(
                str(self._observer_model_style_combo.currentData() or "cartoon")
            )
        )
        model_layout.addWidget(self._observer_model_style_combo)

        opacity_group, opacity_layout = self._observer_control_group(
            self._observer_controls_widget
        )
        opacity_layout.addWidget(QLabel("Opacity"))
        self._observer_protein_opacity_slider = QSlider(Qt.Horizontal)
        self._observer_protein_opacity_slider.setRange(0, 100)
        self._observer_protein_opacity_slider.setValue(100)
        self._observer_protein_opacity_slider.setFixedWidth(92)
        self._observer_protein_opacity_slider.setToolTip("Protein opacity in observer panels")
        self._observer_protein_opacity_slider.valueChanged.connect(
            lambda value: self._set_observer_protein_opacity(int(value) / 100.0)
        )
        opacity_layout.addWidget(self._observer_protein_opacity_slider)

        self._observer_control_groups = [
            navigation_group,
            highlight_group,
            dataset_group,
            model_group,
            opacity_group,
        ]
        self._observer_visual_controls_widget = QWidget(self._observer_controls_widget)
        self._observer_visual_controls_layout = QHBoxLayout(
            self._observer_visual_controls_widget
        )
        self._observer_visual_controls_layout.setContentsMargins(0, 0, 0, 0)
        self._observer_visual_controls_layout.setSpacing(8)
        self._observer_visual_controls_widget.hide()
        self._observer_controls_layout_mode = ""
        observer_layout.addWidget(self._observer_controls_widget)
        self._observer_controls_widget.installEventFilter(self)
        QTimer.singleShot(0, self._refresh_observer_controls_layout)

        # The shared time axis sits directly below its frame controls.
        self._observer_timeline = ObserverTimelineWidget(self._protein_observer_container)
        self._observer_timeline.frame_clicked.connect(self._on_observer_timeline_frame_clicked)
        self._observer_timeline.zoom_changed.connect(self._sync_observer_timeline_zoom_slider)
        observer_layout.addWidget(self._observer_timeline)

        self._observer_change_explanation = QLabel(
            "Choose a physical arc-length position to compare its baseline and mutant residue environments."
        )
        self._observer_change_explanation.setWordWrap(True)
        self._observer_change_explanation.setStyleSheet(
            "font-size:11px;color:#1E3A8A;background:#EFF6FF;"
            "border:1px solid #BFDBFE;border-radius:5px;padding:5px;"
        )
        self._observer_change_explanation.setVisible(False)

        self._protein_observer_debug_label = QLabel(
            "Select a residue region above; both observers will focus on the same physical arc-length position."
        )
        self._protein_observer_debug_label.setWordWrap(True)
        self._protein_observer_debug_label.setStyleSheet("font-size:11px;color:#6b7280;")
        self._protein_observer_debug_label.setVisible(False)

        self._observer_panels_scroll = QScrollArea(self._protein_observer_container)
        self._observer_panels_scroll.setWidgetResizable(True)
        self._observer_panels_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._observer_panels_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._observer_panels_scroll.setFrameShape(QFrame.NoFrame)
        self._observer_panels_content = QWidget(self._observer_panels_scroll)
        self._observer_panels_layout = QHBoxLayout(self._observer_panels_content)
        self._observer_panels_layout.setContentsMargins(0, 0, 0, 0)
        self._observer_panels_layout.setSpacing(8)
        self._observer_panels_scroll.setWidget(self._observer_panels_content)
        observer_layout.addWidget(self._observer_panels_scroll, 1)

        self._refresh_observer_dataset_multiselect()
        self._refresh_observer_change_event_selector()
        self._set_observer_model_style("cartoon")
        self._set_observer_highlight_style("ball_and_stick")
        self._set_observer_protein_opacity(1.0)

        self._main_view_stack.addWidget(self._protein_observer_container)

        self._evidence_panel = EvidencePanel(self)
        self._evidence_panel.back_requested.connect(lambda: self._toggle_evidence_panel(False))
        self._evidence_panel.observer_requested.connect(lambda: self._toggle_protein_observer_panel(True))
        self._evidence_panel.marker_requested.connect(self._on_evidence_marker_requested)
        self._evidence_panel.dataset_selected.connect(self._on_evidence_dataset_selected)
        self._evidence_panel.region_step_requested.connect(self._step_dataset_compare_region)
        self._main_view_stack.setCurrentWidget(self._viewer)
        self._main_splitter.addWidget(self._main_view_stack)

        # Bottom charts
        charts_widget = QWidget()
        charts_outer = QVBoxLayout(charts_widget)
        charts_outer.setContentsMargins(4, 2, 4, 4)
        charts_outer.setSpacing(2)
        self._charts_outer_layout = charts_outer

        self._profile_chart = ProfileChart(external_compact_controls=True)
        self._statistics_chart = StatisticsChart()
        self._combination_region_chart = ResidueCombinationRegionChart(compact=True)
        self._statistics_chart.set_display_config(*self._profile_chart.display_config())

        # Sync button bar
        self._chart_toolbar_widget = QWidget(charts_widget)
        btn_bar = QHBoxLayout(self._chart_toolbar_widget)
        btn_bar.setContentsMargins(4, 0, 4, 0)
        btn_bar.setSpacing(5)
        self._chart_status_label = QLabel("")
        btn_bar.addWidget(self._chart_status_label)
        btn_bar.addStretch()

        for control in self._profile_chart.compact_control_widgets():
            btn_bar.addWidget(control)

        self._sync_charts_btn = QPushButton("Sync Charts")
        self._sync_charts_btn.setFixedHeight(24)
        self._sync_charts_btn.setFixedWidth(90)
        self._sync_charts_btn.clicked.connect(self._sync_charts)
        btn_bar.addWidget(self._sync_charts_btn)
        self._open_charts_btn = QPushButton("Chart Window")
        self._open_charts_btn.setFixedHeight(24)
        self._open_charts_btn.setFixedWidth(104)
        self._open_charts_btn.setToolTip("Open a larger synchronized chart workspace")
        self._open_charts_btn.clicked.connect(self._open_chart_workspace)
        btn_bar.addWidget(self._open_charts_btn)
        # Selection state and chart controls belong to Tunnel Properties rather
        # than floating above every Evidence tab.  The combination-size selector
        # remains inside Combination Regions, where its effect is visible.
        charts_outer.addWidget(self._chart_toolbar_widget)

        # Charts row
        self._detached_hint = QLabel(
            "Charts are currently displayed in Chart Workspace. "
            "Use the button above to reopen the detached window."
        )
        self._detached_hint.setStyleSheet(
            "color: #909399; font-size: 11px; padding: 10px 4px 6px 4px;"
        )
        self._detached_hint.setWordWrap(True)
        self._detached_hint.setVisible(False)
        charts_outer.addWidget(self._detached_hint)

        self._bottom_charts_container = QWidget()
        charts_layout = QHBoxLayout()
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(6)
        self._bottom_charts_container.setLayout(charts_layout)

        # Tunnel properties use a stable two-column layout: Profile and
        # Statistics share the left pane through a segmented switch, while
        # Combination Regions stays visible on the right for direct context.
        self._chart_mode_stack = ChartModeStack(
            self._profile_chart,
            self._statistics_chart,
        )
        self._chart_mode_stack.setMinimumWidth(0)
        self._chart_mode_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        charts_layout.addWidget(self._chart_mode_stack, 1)

        combination_host = QWidget(self._bottom_charts_container)
        combination_host.setMinimumWidth(0)
        combination_host.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        combination_layout = QVBoxLayout(combination_host)
        combination_layout.setContentsMargins(4, 0, 0, 0)
        combination_layout.setSpacing(4)
        combination_title = QLabel("Combination Regions")
        combination_title.setStyleSheet(
            "font-size:12px;font-weight:600;color:#374151;padding:4px 2px;"
        )
        combination_layout.addWidget(combination_title)
        self._combination_region_chart.set_embedded_mode(True)
        combination_layout.addWidget(self._combination_region_chart, 1)
        self._combination_region_host = combination_host
        charts_layout.addWidget(combination_host, 1)
        charts_outer.addWidget(self._bottom_charts_container, 1)

        # Overview remains the landing page; this page is opened explicitly by
        # its Detailed Properties button or the corresponding tab.
        self._bottom_charts_page = charts_widget
        self._evidence_panel.set_charts_page(charts_widget, visible=True)
        charts_widget.setUpdatesEnabled(self._legacy_charts_enabled)
        self._profile_chart.setUpdatesEnabled(self._legacy_charts_enabled)
        self._statistics_chart.setUpdatesEnabled(self._legacy_charts_enabled)
        self._combination_region_chart.setUpdatesEnabled(
            self._tunnel_properties_enabled
        )
        self._bottom_analysis_mode = "evidence"
        self._main_splitter.addWidget(self._evidence_panel)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([560, 280])
        self._bottom_chart_splitter_sizes = [560, 280]

        layout.addWidget(self._main_splitter)

    def _relocate_top_controls(self):
        self._main_toolbar.setVisible(False)

        self._protein_opacity_slider.setVisible(False)
        self._clear_protein_btn.setVisible(False)
        self._binding_sites_btn.setVisible(False)
        self._protein_align_combo.setVisible(False)
        self._lasso_btn.setVisible(False)

        top_controls = QWidget()
        top_controls.setObjectName("mainViewControls")
        top_controls.setStyleSheet(
            "#mainViewControls{background:#ffffff;border-bottom:1px solid #e5e7eb;}"
        )
        self._top_controls_widget = top_controls

        controls_layout = QGridLayout(top_controls)
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(0)
        self._top_controls_layout = controls_layout

        self._color_label = QLabel("Color")
        self._color_label.setStyleSheet("font-size:11px;color:#909399;")
        self._bg_label = QLabel("BG")
        self._bg_label.setStyleSheet("font-size:11px;color:#909399;")
        self._top_controls_items = [
            self._color_label,
            self._path_color_slider,
            self._bg_label,
            self._bg_opacity_slider,
            self._point_select_btn,
            self._clear_sel_btn,
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._protein_toggle_btn,
            self._main_view_btn,
            self._load_observer_protein_btn,
        ]

        self._central_layout.insertWidget(0, top_controls)
        # The main scene now owns its two essential actions as unobtrusive
        # overlays (Focus at top-left and Residue Observer at top-right).
        # Hide the former Color/BG toolbar row in every window size.
        top_controls.setVisible(False)
        self._refresh_top_controls_layout()

        self._load_protein_btn.setText("Load Protein")
        self._load_observer_protein_btn.setText("Residue Observer")

        button_row = self._dataset_panel._button_row
        for widget in (
            self._dataset_panel._add_btn,
            self._dataset_panel._prefix_btn,
            self._dataset_panel._color_btn,
            self._dataset_panel._reference_btn,
            self._dataset_panel._visibility_btn,
            self._dataset_panel._cluster_mode_btn,
            self._dataset_panel._cluster_labels_btn,
            self._dataset_panel._remove_btn,
        ):
            button_row.removeWidget(widget)

        metrics = QFontMetrics(self.font())
        shared_width = max(
            128,
            metrics.horizontalAdvance(self._dataset_panel._add_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._prefix_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._color_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._reference_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._visibility_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._cluster_mode_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._cluster_labels_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._remove_btn.text()),
        ) + 24
        for button in (
            self._dataset_panel._add_btn,
            self._dataset_panel._prefix_btn,
            self._dataset_panel._color_btn,
            self._dataset_panel._reference_btn,
            self._dataset_panel._visibility_btn,
            self._dataset_panel._cluster_mode_btn,
            self._dataset_panel._cluster_labels_btn,
            self._dataset_panel._remove_btn,
        ):
            button.setMinimumWidth(shared_width)
            button.setFixedHeight(self._dataset_panel._add_btn.sizeHint().height())
        button_row.addWidget(self._dataset_panel._add_btn, 0, 0)
        button_row.addWidget(self._dataset_panel._prefix_btn, 0, 1)
        button_row.addWidget(self._dataset_panel._color_btn, 0, 2)
        button_row.addWidget(self._dataset_panel._reference_btn, 0, 3)
        button_row.addWidget(self._dataset_panel._visibility_btn, 1, 0)
        button_row.addWidget(self._dataset_panel._cluster_mode_btn, 1, 1)
        button_row.addWidget(self._dataset_panel._cluster_labels_btn, 1, 2)
        button_row.addWidget(self._dataset_panel._remove_btn, 1, 3)

        for button in (
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._point_select_btn,
            self._clear_sel_btn,
            self._protein_toggle_btn,
        ):
            button.setMinimumWidth(max(button.minimumWidth(), 78 if button is not self._protein_toggle_btn else 118))
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        for widget in (
            self._path_color_slider,
            self._bg_opacity_slider,
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._protein_toggle_btn,
            self._load_observer_protein_btn,
            self._point_select_btn,
            self._clear_sel_btn,
        ):
            widget.setVisible(True)
        # Evidence now shares the lower analysis area with Tunnel Properties. Keep this
        # legacy toolbar button as a shortcut target for workspace state, but
        # avoid exposing a second, competing visual switch.
        self._evidence_view_btn.setVisible(False)
        for widget in (
            self._path_change_btn,
            self._exit_delta_btn,
            self._path_delta_filter_slider,
            self._path_delta_filter_label,
            self._residue_toggle_btn,
            self._residue_labels_btn,
            self._exit_cluster_btn,
            self._load_protein_btn,
        ):
            widget.setVisible(False)
        self._protein_style_combo.setVisible(False)
        self._frame_menu_btn.setVisible(False)

        self._residue_toggle_btn.blockSignals(True)
        self._residue_toggle_btn.setChecked(False)
        self._residue_toggle_btn.blockSignals(False)
        self._residue_labels_btn.blockSignals(True)
        self._residue_labels_btn.setChecked(False)
        self._residue_labels_btn.blockSignals(False)
        self.state.show_all_residues = False
        self._viewer.clear_residues()
        self._viewer.clear_residue_labels()

    def _refresh_top_controls_layout(self):
        if self._top_controls_layout is None or self._top_controls_widget is None:
            return
        layout = self._top_controls_layout

        # The main controls are a toolbar, not a responsive card grid.  The
        # previous width check ran before the first layout pass and therefore
        # put the controls into two rows until the user resized the window.
        # Keep one stable row from the initial frame onward.
        if layout.count() == len(self._top_controls_items):
            already_single_row = all(
                layout.itemAtPosition(0, column) is not None
                and layout.itemAtPosition(0, column).widget() is widget
                for column, widget in enumerate(self._top_controls_items)
            )
            if already_single_row:
                return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._top_controls_widget)

        for index, widget in enumerate(self._top_controls_items):
            layout.addWidget(widget, 0, index)
        columns = len(self._top_controls_items)
        for column in range(columns + 1):
            layout.setColumnStretch(column, 0)
        layout.setColumnStretch(columns, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_top_controls_layout()
        self._refresh_observer_controls_layout()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_observer_controls_widget", None):
            if event.type() == QEvent.Type.Resize:
                # Evaluate after Qt has committed the new available width.
                QTimer.singleShot(0, self._refresh_observer_controls_layout)
        auto_hide_widgets = {
            widget
            for widget in (
                getattr(self, "_side_dock", None),
                getattr(self, "_side_dock_container", None),
                getattr(self, "_side_dock_auto_hide_rail", None),
                getattr(self, "_side_dock_tabs", None),
                getattr(self, "_side_dock_resize_handle", None),
            )
            if widget is not None
        }
        if watched in auto_hide_widgets:
            event_type = event.type()
            # Control Panel visibility is click-driven. Pointer hover is
            # deliberately ignored so moving across the left edge cannot
            # unexpectedly shift the main canvas.
            if event_type == QEvent.Type.MouseButtonPress and watched is getattr(self, "_side_dock_auto_hide_rail", None):
                self._toggle_side_dock_from_rail()
                return True
        return super().eventFilter(watched, event)

    # ─── Side Panel (Dock) ──────────────────────────────
    def _build_side_panel(self):
        dock = QDockWidget("Control Panel", self)
        dock.setObjectName("ControlPanelDock")
        dock.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dock.setAutoFillBackground(True)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        title_bar = QWidget(dock)
        title_bar.setFixedHeight(0)
        dock.setTitleBarWidget(title_bar)
        tabs = QTabWidget()
        tabs.setObjectName("ControlPanelTabs")
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        tabs.tabBar().setExpanding(False)
        tabs.tabBar().setUsesScrollButtons(True)
        tabs.setDocumentMode(True)
        tabs.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tabs.setAutoFillBackground(True)

        # Hidden helper panels kept for existing selection / table logic.
        self._residue_panel = ResiduePanel(self.state, self.db)

        self._path_panel = PathPanel(self.state, self.db)

        # Visible tabs
        self._dataset_panel = DatasetPanel(self.db, self.state)
        tabs.addTab(self._dataset_panel, "Datasets")

        path_tab_index = tabs.addTab(self._path_panel, "Paths")

        self._residue_compare_panel = ResidueComparePanel()
        residue_compare_tab_index = tabs.addTab(self._residue_compare_panel, "Residue Compare")

        self._residue_combination_panel = ResidueCombinationPanel()
        residue_combination_tab_index = tabs.addTab(
            self._residue_combination_panel,
            "Residue Combination Compare",
        )
        self._dataset_compare_panel = DatasetComparePanel()
        if self._legacy_residue_combination_enabled:
            self._combination_region_chart.set_embedded_mode(True)
            self._dataset_compare_panel.set_combination_regions_widget(
                self._combination_region_chart
            )
        self._evidence_panel.set_path_region_widget(
            self._dataset_compare_panel.take_path_region_widget()
        )
        tabs.addTab(self._dataset_compare_panel, "Dataset Compare")
        # These panels still provide internal selection and comparison services,
        # but they are no longer primary workflow entrances.  Keep the widgets
        # alive for existing signal connections while presenting a focused
        # Datasets -> Dataset Compare navigation in the control panel.
        for hidden_tab_index in (
            path_tab_index,
            residue_compare_tab_index,
            residue_combination_tab_index,
        ):
            tabs.setTabVisible(hidden_tab_index, False)
        self._path_panel.set_active(self._legacy_path_panel_enabled)
        self._residue_panel.set_active(self._legacy_residue_panel_enabled)
        self._residue_compare_panel.setUpdatesEnabled(self._legacy_residue_compare_enabled)
        self._residue_combination_panel.setUpdatesEnabled(
            self._legacy_residue_combination_enabled
        )
        self._statistics_panel = None

        dataset_button_row_width = self._dataset_panel.primary_button_row_width()
        content_width = max(1, int(dataset_button_row_width) + 20)
        self._side_dock_width = content_width + self._side_dock_collapsed_width
        self._side_dock_min_width = self._side_dock_width

        dock_container = QWidget()
        dock_container.setObjectName("ControlPanelContainer")
        dock_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dock_container.setAutoFillBackground(True)
        dock_layout = QHBoxLayout(dock_container)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)

        auto_hide_rail = QFrame(dock_container)
        auto_hide_rail.setObjectName("ControlPanelAutoHideRail")
        auto_hide_rail.setFixedWidth(self._side_dock_collapsed_width)
        auto_hide_rail.setCursor(Qt.PointingHandCursor)
        auto_hide_rail.setToolTip("Click to show or hide the Control Panel")
        auto_hide_rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        rail_layout = QVBoxLayout(auto_hide_rail)
        rail_layout.setContentsMargins(2, 8, 2, 8)
        rail_layout.setSpacing(0)
        rail_indicator = QToolButton(auto_hide_rail)
        rail_indicator.setObjectName("ControlPanelAutoHideIndicator")
        rail_indicator.setArrowType(Qt.RightArrow)
        rail_indicator.setAutoRaise(True)
        rail_indicator.setFixedSize(24, 32)
        rail_indicator.setFocusPolicy(Qt.NoFocus)
        rail_indicator.clicked.connect(self._toggle_side_dock_from_rail)
        rail_layout.addWidget(rail_indicator, 0, Qt.AlignTop | Qt.AlignHCenter)
        rail_layout.addStretch(1)
        dock_layout.addWidget(auto_hide_rail, 0)
        dock_layout.addWidget(tabs, 1)
        self._side_dock_resize_handle = DockResizeHandle(
            lambda: int(self._side_dock.width()) if getattr(self, "_side_dock", None) is not None else int(self._side_dock_width),
            self._apply_side_dock_width,
            dock_container,
        )
        dock_layout.addWidget(self._side_dock_resize_handle, 0)

        dock.setWidget(dock_container)
        self._side_dock = dock
        self._side_dock_container = dock_container
        self._side_dock_tabs = tabs
        self._side_dock_auto_hide_rail = auto_hide_rail
        self._side_dock_auto_hide_indicator = rail_indicator
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self._side_dock_expand_timer = QTimer(self)
        self._side_dock_expand_timer.setSingleShot(True)
        self._side_dock_expand_timer.setInterval(80)
        self._side_dock_expand_timer.timeout.connect(
            lambda: self._set_side_dock_expanded(True)
        )
        self._side_dock_collapse_timer = QTimer(self)
        self._side_dock_collapse_timer.setSingleShot(True)
        self._side_dock_collapse_timer.setInterval(420)
        self._side_dock_collapse_timer.timeout.connect(
            self._collapse_side_dock_if_idle
        )
        for widget in (dock, dock_container, auto_hide_rail, tabs, self._side_dock_resize_handle):
            widget.setMouseTracking(True)
            widget.installEventFilter(self)

        self._set_side_dock_expanded(False, animate=False, force=True)
        QTimer.singleShot(0, self._apply_initial_side_dock_width)

    def _toggle_side_dock_from_rail(self):
        """Toggle the Control Panel only after an explicit rail click."""
        expand_timer = getattr(self, "_side_dock_expand_timer", None)
        collapse_timer = getattr(self, "_side_dock_collapse_timer", None)
        if expand_timer is not None:
            expand_timer.stop()
        if collapse_timer is not None:
            collapse_timer.stop()
        self._set_side_dock_expanded(not bool(self._side_dock_auto_expanded))

    def _apply_initial_side_dock_width(self):
        dataset_button_row_width = self._dataset_panel.primary_button_row_width()
        content_width = max(1, int(dataset_button_row_width) + 20)
        self._side_dock_width = content_width + self._side_dock_collapsed_width
        self._side_dock_min_width = self._side_dock_width
        if self._side_dock_auto_expanded:
            self._apply_side_dock_width(self._side_dock_width)
        else:
            self._apply_side_dock_runtime_width(self._side_dock_collapsed_width, locked=True)

    def _apply_side_dock_width(self, width: int):
        dock = getattr(self, "_side_dock", None)
        min_width = int(getattr(self, "_side_dock_min_width", 0))
        width_value = max(min_width, int(width))
        if self.width() > 0:
            width_value = min(width_value, max(min_width, int(self.width()) - 220))
        self._side_dock_width = width_value
        if dock is not None and self._side_dock_auto_expanded:
            self._apply_side_dock_runtime_width(width_value, locked=False)

    def _apply_side_dock_runtime_width(self, width: int, *, locked: bool):
        dock = getattr(self, "_side_dock", None)
        if dock is None:
            return
        width_value = max(1, int(width))
        minimum_width = width_value if locked else int(self._side_dock_min_width)
        maximum_width = width_value if locked else 16777215
        dock.setMinimumWidth(minimum_width)
        dock.setMaximumWidth(maximum_width)
        widget = dock.widget()
        if widget is not None:
            widget.setMinimumWidth(minimum_width)
            widget.setMaximumWidth(maximum_width)
        dock.resize(width_value, dock.height())
        try:
            self.resizeDocks([dock], [width_value], Qt.Horizontal)
        except Exception:
            pass
        if widget is not None:
            widget.updateGeometry()
            widget.update()
        dock.updateGeometry()
        dock.update()

    def _set_side_dock_expanded(
        self,
        expanded: bool,
        *,
        animate: bool = True,
        force: bool = False,
    ):
        dock = getattr(self, "_side_dock", None)
        if dock is None:
            return
        expanded = bool(expanded)
        animation = getattr(self, "_side_dock_animation", None)
        if not force and expanded == self._side_dock_auto_expanded and not (
            animation is not None
            and animation.state() == QAbstractAnimation.State.Running
        ):
            return

        self._side_dock_auto_expanded = expanded
        self._side_dock_auto_hide_indicator.setArrowType(
            Qt.LeftArrow if expanded else Qt.RightArrow
        )
        self._side_dock_auto_hide_rail.setToolTip(
            "Click to hide the Control Panel"
            if expanded
            else "Click to show the Control Panel"
        )
        if expanded:
            self._side_dock_tabs.setVisible(True)
            self._side_dock_resize_handle.setVisible(True)

        start_width = max(1, int(dock.width()))
        target_width = (
            int(self._side_dock_width)
            if expanded
            else int(self._side_dock_collapsed_width)
        )
        if animation is not None:
            animation.stop()
            animation.deleteLater()

        if not animate or start_width == target_width:
            self._finish_side_dock_transition(expanded)
            return

        animation = QVariantAnimation(self)
        animation.setStartValue(start_width)
        animation.setEndValue(target_width)
        animation.setDuration(150 if expanded else 130)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(
            lambda value: self._apply_side_dock_runtime_width(int(value), locked=True)
        )
        animation.finished.connect(
            lambda: self._finish_side_dock_transition(expanded)
        )
        self._side_dock_animation = animation
        animation.start()

    def _finish_side_dock_transition(self, expanded: bool):
        if bool(expanded) != self._side_dock_auto_expanded:
            return
        if expanded:
            self._side_dock_tabs.setVisible(True)
            self._side_dock_resize_handle.setVisible(True)
            # Finish at the requested width while locked. Unlocking before
            # resizeDocks() lets Qt assign the whole available window width.
            self._apply_side_dock_runtime_width(self._side_dock_width, locked=True)
            dock = self._side_dock
            widget = dock.widget()
            dock.setMinimumWidth(self._side_dock_min_width)
            dock.setMaximumWidth(16777215)
            if widget is not None:
                widget.setMinimumWidth(self._side_dock_min_width)
                widget.setMaximumWidth(16777215)
        else:
            self._side_dock_tabs.setVisible(False)
            self._side_dock_resize_handle.setVisible(False)
            self._apply_side_dock_runtime_width(self._side_dock_collapsed_width, locked=True)

    def _collapse_side_dock_if_idle(self):
        dock = getattr(self, "_side_dock", None)
        if dock is None or not self._side_dock_auto_expanded:
            return
        local_pos = dock.mapFromGlobal(QCursor.pos())
        pointer_inside = dock.rect().contains(local_pos)
        popup_open = QApplication.activePopupWidget() is not None
        dragging = bool(
            getattr(getattr(self, "_side_dock_resize_handle", None), "_dragging", False)
        )
        if pointer_inside or popup_open or dragging:
            self._side_dock_collapse_timer.start()
            return
        self._set_side_dock_expanded(False)

    def _build_statistics_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._statistics_selection_label = QLabel("Selected: 0 paths")
        self._statistics_selection_label.setStyleSheet("font-weight:600;color:#1f2937;")
        header_row.addWidget(self._statistics_selection_label)
        header_row.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.clicked.connect(self._refresh_statistics_selection_label)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self._stats_entrance_centroid_btn = QPushButton("Entrance Centroid")
        self._stats_entrance_centroid_btn.setFixedHeight(28)
        self._stats_entrance_centroid_btn.clicked.connect(self._compute_statistics_entrance_centroid)
        button_row.addWidget(self._stats_entrance_centroid_btn)
        self._stats_centerline_distance_btn = QPushButton("Centerline Distance")
        self._stats_centerline_distance_btn.setFixedHeight(28)
        self._stats_centerline_distance_btn.clicked.connect(self._compute_statistics_centerline_distance)
        button_row.addWidget(self._stats_centerline_distance_btn)
        self._stats_export_cluster_btn = QPushButton("Cluster Count Results")
        self._stats_export_cluster_btn.setFixedHeight(28)
        self._stats_export_cluster_btn.clicked.connect(self._compute_dataset_cluster_results_export)
        button_row.addWidget(self._stats_export_cluster_btn)
        self._stats_cluster_compactness_btn = QPushButton("Cluster Compactness")
        self._stats_cluster_compactness_btn.setFixedHeight(28)
        self._stats_cluster_compactness_btn.clicked.connect(self._compute_dataset_cluster_compactness)
        button_row.addWidget(self._stats_cluster_compactness_btn)
        self._stats_cluster_residue_similarity_btn = QPushButton("Cluster Residue Similarity")
        self._stats_cluster_residue_similarity_btn.setFixedHeight(28)
        self._stats_cluster_residue_similarity_btn.clicked.connect(self._compute_dataset_cluster_residue_similarity)
        button_row.addWidget(self._stats_cluster_residue_similarity_btn)
        layout.addLayout(button_row)

        export_row = QHBoxLayout()
        self._statistics_summary_label = QLabel("No statistics calculated.")
        self._statistics_summary_label.setWordWrap(True)
        self._statistics_summary_label.setStyleSheet("font-size:11px;color:#6b7280;")
        export_row.addWidget(self._statistics_summary_label, 1)
        self._stats_export_btn = QPushButton("Export CSV")
        self._stats_export_btn.setFixedHeight(26)
        self._stats_export_btn.setEnabled(False)
        self._stats_export_btn.clicked.connect(self._export_statistics_csv)
        export_row.addWidget(self._stats_export_btn)
        layout.addLayout(export_row)

        self._statistics_table = QTableWidget(0, 0)
        self._statistics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._statistics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._statistics_table.setAlternatingRowColors(True)
        self._statistics_table.verticalHeader().setVisible(False)
        self._statistics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._statistics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._statistics_table, 1)

        return panel

    # ─── Bottom Panel ───────────────────────────────────
    def _build_bottom_panel(self):
        pass  # Charts already in main splitter

    # ─── Status Bar ─────────────────────────────────────
    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_paths = QLabel("Paths: 0")
        self._status_residues = QLabel("Residues: 0")
        self._status_memory = QLabel("")
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedWidth(120)
        self._loading_bar.setFixedHeight(12)
        self._loading_bar.setVisible(False)
        self._loading_bar.setStyleSheet(
            "QProgressBar{border:1px solid #d1d5db;border-radius:6px;background:#f3f4f6;}"
            "QProgressBar::chunk{border-radius:6px;background:#22c55e;}"
        )
        self._statusbar.addWidget(self._status_paths)
        self._statusbar.addWidget(QLabel("|"))
        self._statusbar.addWidget(self._status_residues)
        self._statusbar.addWidget(QLabel("|"))
        self._status_selected = QLabel("Selected: 0 paths")
        self._statusbar.addWidget(self._status_selected)
        self._statusbar.addPermanentWidget(self._loading_bar)
        self._statusbar.addPermanentWidget(self._status_memory)

    def _begin_loading(self, message: str):
        self._loading_depth += 1
        self._loading_bar.setVisible(True)
        self._statusbar.showMessage(message)
        QApplication.processEvents()

    def _end_loading(self, message: str | None = None, timeout_ms: int = 3000):
        self._loading_depth = max(0, self._loading_depth - 1)
        if self._loading_depth == 0:
            self._loading_bar.setVisible(False)
            if message:
                self._statusbar.showMessage(message, timeout_ms)

    # ─── Signal wiring ──────────────────────────────────
    def _connect_signals(self):
        # Residue panel → update matched paths
        self._residue_panel.selection_changed.connect(self._on_residue_changed)

        # Path panel → show profiles, export
        self._path_panel.paths_selected.connect(self._on_paths_selected)
        self._path_panel.path_selected.connect(self._on_path_clicked)
        self._path_panel.export_requested.connect(self._on_export)
        self._path_panel.profile_requested.connect(self._on_load_selected_profiles)
        self._dataset_panel.datasets_changed.connect(self._on_datasets_changed)
        self._dataset_panel.dataset_color_changed.connect(self._on_dataset_color_changed)
        self._dataset_panel.dataset_visibility_changed.connect(self._on_dataset_visibility_changed)
        self._dataset_panel.dataset_visibility_batch_changed.connect(self._on_dataset_visibility_batch_changed)
        self._dataset_panel.dataset_selection_changed.connect(self._on_dataset_selection_changed)
        self._dataset_panel.dataset_cluster_mode_changed.connect(self._on_dataset_cluster_mode_changed)
        self._dataset_panel.dataset_cluster_labels_changed.connect(self._on_dataset_cluster_labels_changed)
        self._dataset_panel.cluster_paths_selected.connect(self._on_dataset_cluster_paths_selected)
        self._dataset_panel.reference_protein_requested.connect(self._on_load_protein)
        self._dataset_compare_panel.compare_requested.connect(self._on_dataset_compare_requested)
        self._dataset_compare_panel.source_dataset_changed.connect(self._on_dataset_compare_source_changed)
        self._dataset_compare_panel.target_datasets_changed.connect(
            self._on_dataset_compare_targets_changed
        )
        self._dataset_compare_panel.residue_threshold_changed.connect(self._on_dataset_compare_residue_threshold_changed)
        self._dataset_compare_panel.rmsd_threshold_changed.connect(self._on_dataset_compare_rmsd_threshold_changed)
        self._dataset_compare_panel.rmsd_max_threshold_changed.connect(self._on_dataset_compare_rmsd_max_threshold_changed)
        self._dataset_compare_panel.comparison_selected.connect(self._on_dataset_compare_selection)
        self._dataset_compare_panel.mapping_lock_requested.connect(self._on_dataset_compare_mapping_lock_requested)
        self._dataset_compare_panel.motif_selected.connect(
            self._on_dataset_compare_region_selected
        )
        self._dataset_compare_panel.evidence_view_requested.connect(
            lambda: self._set_main_workspace("evidence")
        )
        self._dataset_compare_panel.evidence_export_requested.connect(self._on_dataset_compare_export_evidence)
        self._dataset_compare_panel.clear_requested.connect(self._clear_dataset_compare)
        self._evidence_panel.properties_requested.connect(
            self._on_tunnel_properties_requested
        )

        # Shared chart config
        self._profile_chart.display_config_changed.connect(
            self._on_chart_display_config_changed
        )
        self._profile_chart.appearance_changed.connect(
            self._on_chart_appearance_changed
        )
        self._profile_chart.keep_only_requested.connect(
            self._on_chart_keep_only_requested
        )
        self._profile_chart.exclude_requested.connect(
            self._on_chart_exclude_requested
        )
        self._profile_chart.chart_selection_changed.connect(
            self._on_chart_selection_changed
        )
        self._profile_chart.undo_requested.connect(
            self._on_chart_undo_requested
        )

        # Lasso selection
        self._viewer.lasso_selected.connect(self._on_lasso_selected)
        self._viewer.exit_cluster_selected.connect(self._on_exit_cluster_selected)
        self._viewer.path_clicked.connect(self._on_viewer_path_clicked)
        self._viewer.path_coordinate_mode_changed.connect(
            self._on_viewer_coordinate_mode_changed
        )
        self._viewer.observer_requested.connect(
            lambda: self._set_main_workspace("observer")
        )
        self._viewer.focus_mode_changed.connect(self._focus_btn.setChecked)
        self._viewer.protein_opacity_changed.connect(
            self._sync_main_protein_opacity
        )

        # Residue 3D click
        self._viewer.residue_clicked.connect(self._on_residue_3d_clicked)
        self._viewer.combination_clicked.connect(self._on_combination_marker_clicked)
        self._residue_compare_panel.residue_selection_requested.connect(
            self._on_residue_compare_selection_requested
        )
        self._residue_compare_panel.residue_clear_requested.connect(
            self._on_residue_compare_clear_requested
        )
        self._residue_combination_panel.visualization_changed.connect(
            self._refresh_residue_combination_markers
        )
        self._residue_combination_panel.manual_update_requested.connect(
            self._manual_update_residue_combination_compare
        )
        self._residue_combination_panel.contained_match_query_requested.connect(
            self._query_residue_combination_contained_match
        )
        self._residue_combination_panel.contained_match_compute_requested.connect(
            self._compute_residue_combination_contained_match
        )
        self._residue_combination_panel.contained_match_clear_requested.connect(
            self._clear_residue_combination_contained_match
        )
        self._residue_combination_panel.combination_selection_requested.connect(
            self._on_combination_selection_requested
        )
        self._residue_combination_panel.left_residue_set_requested.connect(
            self._on_left_residue_set_requested
        )

        # State signals — unified effective set drives 3D + table + status
        self.state.frame_range_changed.connect(self._reload_3d)
        self.state.effective_paths_changed.connect(self._on_effective_paths_changed)
        self.state.residue_selection_changed.connect(self._refresh_residue_3d)
        self.state.residue_selection_changed.connect(self._sync_residue_compare_selection_state)
        self.state.category_visibility_changed.connect(self._on_category_visibility_changed)
        self.state.dataset_display_changed.connect(self._on_dataset_display_changed)

        # Selection mode + entry/exit display
        self.state.selection_mode_changed.connect(
            lambda mode: self._viewer.set_selection_mode(mode)
        )
        self.state.entry_exit_display_changed.connect(self._on_entry_exit_display_changed)
        self.state.entry_exit_scope_changed.connect(self._on_entry_exit_scope_changed)

        # Chart debounce timer
        self._chart_timer = QTimer()
        self._chart_timer.setSingleShot(True)
        self._chart_timer.setInterval(150)
        self._chart_timer.timeout.connect(self._do_chart_update)

        # Effective-path changes can arrive in bursts (table selection, lasso,
        # linked Compare/Observer updates).  Commit only the newest state on the
        # next display frame instead of rebuilding VTK actors for every signal.
        self._effective_render_timer = QTimer()
        self._effective_render_timer.setSingleShot(True)
        self._effective_render_timer.setInterval(24)
        self._effective_render_timer.timeout.connect(self._flush_effective_path_refresh)

        # Keyboard shortcuts
        QShortcut(QKeySequence("L"), self).activated.connect(
            lambda: self._lasso_btn.toggle()
        )
        QShortcut(QKeySequence("F"), self).activated.connect(
            lambda: self._focus_btn.toggle()
        )
        QShortcut(QKeySequence("Escape"), self).activated.connect(
            self._clear_selection
        )
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self._set_main_workspace("main")
        )
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self._set_main_workspace("observer")
        )
        QShortcut(QKeySequence("Ctrl+3"), self).activated.connect(
            lambda: self._set_main_workspace("evidence")
        )
        QShortcut(QKeySequence("Alt+Left"), self).activated.connect(
            lambda: self._step_dataset_compare_region(-1)
        )
        QShortcut(QKeySequence("Alt+Right"), self).activated.connect(
            lambda: self._step_dataset_compare_region(1)
        )

    def _on_chart_display_config_changed(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
    ):
        workspace_shared = bool(
            self._chart_workspace is not None
            and self._chart_workspace.uses_shared_charts
        )
        # ProfileChart emitted this signal after applying the configuration to
        # itself.  Only synchronize the companion chart here; applying the
        # same configuration back to the source chart caused a second rebuild.
        if not self._charts_detached or workspace_shared:
            self._statistics_chart.set_display_config(
                color_key,
                x_key,
                y_key,
                advanced_mode,
            )
        if self._chart_workspace is not None and not workspace_shared:
            self._chart_workspace.set_display_config(
                color_key,
                x_key,
                y_key,
                advanced_mode,
            )

    def _on_chart_appearance_changed(
        self,
        alpha_scale: float,
        depth_scale: float,
    ):
        workspace_shared = bool(
            self._chart_workspace is not None
            and self._chart_workspace.uses_shared_charts
        )
        # The emitting profile already owns the new appearance.  A shared
        # Chart Window therefore needs no synchronization or second repaint.
        if self._chart_workspace is not None and not workspace_shared:
            self._chart_workspace.set_appearance(
                alpha_scale,
                depth_scale,
            )

    def _on_chart_keep_only_requested(self, path_ids):
        selected = set(int(pid) for pid in path_ids)
        if not selected:
            return
        self._push_chart_operation_history()
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) in selected
        ]
        self._set_chart_profiles(profiles)
        self._chart_status_label.setText(f"Chart kept {len(selected)} paths")
        self._statusbar.showMessage(f"Chart kept {len(selected)} paths", 3000)

    def _on_chart_exclude_requested(self, path_ids):
        selected = set(int(pid) for pid in path_ids)
        if not selected:
            return
        self._push_chart_operation_history()
        self.state.apply_lasso(selected, "remove")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) not in selected
        ]
        if profiles:
            self._set_chart_profiles(profiles)
        else:
            self._clear_chart_views()
        self._chart_status_label.setText(f"Chart excluded {len(selected)} paths")
        self._statusbar.showMessage(f"Chart excluded {len(selected)} paths", 3000)

    def _on_chart_selection_changed(self, path_ids):
        if self._chart_workspace is not None:
            self._chart_workspace.reset_filter_history()
            self._chart_workspace.set_chart_selected_paths(path_ids)

    def _push_chart_operation_history(self):
        current = sorted(
            int(profile.get("pathIndex", -1))
            for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) > 0
        )
        if not current:
            return
        if self._chart_operation_history and self._chart_operation_history[-1] == current:
            return
        self._chart_operation_history.append(current)
        if len(self._chart_operation_history) > 5:
            self._chart_operation_history = self._chart_operation_history[-5:]
        self._sync_chart_undo_state()

    def _sync_chart_undo_state(self):
        available = bool(self._chart_operation_history)
        self._profile_chart.set_undo_available(available)
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.set_undo_available(available)

    def _restore_chart_path_ids(self, restored_ids: list[int]):
        selected = set(int(pid) for pid in restored_ids if int(pid) > 0)
        if not selected:
            return
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) in selected
        ]
        if not profiles:
            profiles, _cache_hit = self._cached_path_profiles(sorted(selected))
        self._set_chart_profiles(profiles)
        if not self._charts_detached:
            self._profile_chart.apply_filtered_path_selection(sorted(selected))
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(selected))
            self._chart_workspace.set_chart_selected_paths(sorted(selected))

    def _on_chart_undo_requested(self):
        if not self._chart_operation_history:
            self._statusbar.showMessage("No chart operation to undo", 3000)
            return
        restored = self._chart_operation_history.pop()
        self._restore_chart_path_ids(restored)
        self._sync_chart_undo_state()
        self._chart_status_label.setText(f"Chart undo restored {len(restored)} paths")
        self._statusbar.showMessage(f"Chart undo restored {len(restored)} paths", 3000)

    def _on_chart_workspace_filter_apply_requested(self, payload: dict):
        selected = set(int(pid) for pid in payload.get("path_ids", []) if int(pid) > 0)
        if not selected:
            self._statusbar.showMessage("No paths matched the current chart filter", 3000)
            return
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(selected))
            self._chart_workspace.set_chart_selected_paths(sorted(selected))
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        self._chart_status_label.setText(f"Chart filter selected {len(selected)} paths")
        self._statusbar.showMessage(f"Chart filter selected {len(selected)} paths", 3000)

    def _on_chart_workspace_filter_revert_requested(self, path_ids):
        restored = set(int(pid) for pid in (path_ids or []) if int(pid) > 0)
        if not restored:
            self._statusbar.showMessage("No previous chart selection to restore", 3000)
            return
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(restored))
            self._chart_workspace.set_chart_selected_paths(sorted(restored))
        self.state.apply_lasso(restored, "replace")
        self._path_panel.refresh()
        self._chart_status_label.setText(f"Chart selection restored to {len(restored)} paths")
        self._statusbar.showMessage(f"Chart selection restored to {len(restored)} paths", 3000)

    def _set_chart_profiles(self, profiles):
        if not self._legacy_charts_enabled:
            self._chart_profiles = []
            return
        self._tunnel_properties_render_signature = None
        self._chart_profiles = list(profiles)
        path_color_map = self._chart_path_color_map(self._chart_profiles)
        workspace_shared = bool(
            self._chart_workspace is not None
            and self._chart_workspace.uses_shared_charts
        )
        if not self._charts_detached or workspace_shared:
            self._profile_chart.set_profiles(self._chart_profiles, replot=False)
            self._profile_chart.set_path_color_map(path_color_map, replot=False)
            selected_prefixes = self._profile_chart.selected_dataset_prefixes()
            self._statistics_chart.set_dataset_filter(selected_prefixes, replot=False)
            self._statistics_chart.set_profiles(self._chart_profiles, replot=False)
            self._statistics_chart.set_path_color_map(path_color_map, replot=False)
            self._combination_region_chart.set_profiles(self._chart_profiles, replot=False)
            self._combination_region_chart.set_dataset_filter(selected_prefixes, replot=False)
            self._profile_chart.refresh()
            self._statistics_chart.refresh()
            self._combination_region_chart.refresh()
        if self._chart_workspace is not None and not workspace_shared:
            self._chart_workspace.set_profiles(self._chart_profiles, path_color_map)
            self._chart_workspace.set_chart_selected_paths(
                self._profile_chart.selected_path_ids() if not self._charts_detached else self._chart_workspace.profile_chart.selected_path_ids()
            )
        elif self._chart_workspace is not None:
            self._chart_workspace.set_chart_selected_paths(
                self._profile_chart.selected_path_ids()
            )
        self._sync_chart_undo_state()

    def _clear_chart_views(self):
        self._chart_profiles = []
        self._tunnel_properties_render_signature = None
        if not self._legacy_charts_enabled:
            return
        workspace_shared = bool(
            self._chart_workspace is not None
            and self._chart_workspace.uses_shared_charts
        )
        if not self._charts_detached or workspace_shared:
            self._profile_chart.clear()
            self._statistics_chart.clear()
            self._combination_region_chart.clear()
        if self._chart_workspace is not None and not workspace_shared:
            self._chart_workspace.clear_charts()
        self._sync_chart_undo_state()

    def _active_chart_display_config(self):
        if self._charts_detached and self._chart_workspace is not None:
            return self._chart_workspace.profile_chart.display_config()
        return self._profile_chart.display_config()

    def _active_chart_appearance(self):
        if self._charts_detached and self._chart_workspace is not None:
            return self._chart_workspace.appearance()
        return self._profile_chart.appearance()

    def _set_main_charts_detached(self, detached: bool):
        if self._charts_detached == detached:
            return
        if detached:
            self._charts_detached = True
            self._detached_hint.setVisible(True)
            current_sizes = self._main_splitter.sizes()
            if current_sizes and sum(current_sizes) > 0:
                self._attached_chart_sizes = current_sizes
            self._main_splitter.setSizes([760, 56])
            return

        self._charts_detached = False
        workspace = self._chart_workspace
        if workspace is not None and getattr(workspace, "_uses_shared_charts", False):
            workspace.release_shared_charts()
            self._charts_outer_layout.addWidget(self._bottom_charts_container, 1)
        self._bottom_charts_container.setVisible(True)
        self._detached_hint.setVisible(False)
        # The same chart widgets are moved back from the detached window. Their
        # pyqtgraph scenes remain intact, so close performs no data query and no
        # curve or combination-region rebuild.
        self._profile_chart.set_undo_available(bool(self._chart_operation_history))
        self._main_splitter.setSizes(self._attached_chart_sizes or [560, 280])

    def _open_chart_workspace(self):
        if self._chart_workspace is None:
            self._chart_workspace = ChartWorkspaceDialog(
                self,
                shared_chart_container=self._bottom_charts_container,
                profile_chart=self._profile_chart,
                statistics_chart=self._statistics_chart,
                combination_region_chart=self._combination_region_chart,
                chart_mode_stack=self._chart_mode_stack,
            )
            self._chart_workspace.profile_chart.chart_selection_changed.connect(
                self._chart_workspace.set_chart_selected_paths
            )
            self._chart_workspace.path_filter_apply_requested.connect(
                self._on_chart_workspace_filter_apply_requested
            )
            self._chart_workspace.path_filter_revert_requested.connect(
                self._on_chart_workspace_filter_revert_requested
            )
            self._chart_workspace.closed.connect(
                lambda: self._set_main_charts_detached(False)
            )
        else:
            self._chart_workspace.attach_shared_charts()
        self._chart_workspace.set_chart_mode("profile")
        self._chart_workspace.profile_chart.set_undo_available(
            bool(self._chart_operation_history)
        )
        self._chart_workspace.set_chart_selected_paths(
            self._profile_chart.selected_path_ids()
        )
        self._set_main_charts_detached(True)
        self._chart_workspace.show_and_focus()

    # ─── Data Loading ───────────────────────────────────
    def _init_data(self):
        """Load initial data after window is shown."""
        self._refresh_dataset_views()
        self._statusbar.showMessage("Ready", 3000)

    def _refresh_dataset_views(self):
        self._invalidate_render_signatures()
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        n_paths = self.db.path_count()
        fmin, fmax = self.db.frame_range()

        self._status_paths.setText(f"Paths: {n_paths:,}")
        n_res = int(self.db.get_meta("residue_count", "0"))
        self._status_residues.setText(f"Residues: {n_res:,}")

        # Frame range combos
        self._frame_min_combo.clear()
        self._frame_max_combo.clear()
        self._frame_min_combo.addItem(str(fmin))
        self._frame_max_combo.addItem(str(fmax))
        self._refresh_combo_popup_width(self._frame_min_combo)
        self._refresh_combo_popup_width(self._frame_max_combo)

        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_db(self.db)
            self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
            self._residue_panel.load_options()

        if self._legacy_path_panel_enabled:
            self._path_panel.set_db(self.db)
            self._path_panel.init_model()

        self.state.ensure_dataset_defaults([
            dataset["key"] for dataset in self.db.list_datasets()
        ])

        self._dataset_panel.refresh()
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        if self._legacy_residue_compare_enabled:
            self._residue_compare_panel.set_dataset_choices(datasets)
        if self._legacy_residue_combination_enabled:
            self._residue_combination_panel.set_db(self._primary_analysis_db())
            self._residue_combination_panel.set_dataset_choices(datasets)
            self._residue_combination_panel.set_residue_position_maps(
                self._dataset_residue_position_maps()
            )
            self._residue_combination_panel.set_residue_atom_position_maps(
                self._dataset_residue_atom_position_maps()
            )
        self._refresh_dataset_compare_cluster_options()
        if self._legacy_residue_compare_enabled:
            self._sync_residue_compare_selection_state()
        self._prime_primary_residue_positions()
        if self._legacy_residue_combination_enabled:
            profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
            self._refresh_residue_combination_current_bottlenecks(
                profiles_by_dataset_key=profiles_by_dataset_key,
            )
            self._refresh_residue_combination_contained_match_status()
            self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
            self._sync_residue_combination_context()

        # Load 3D background paths in background thread
        self._reload_3d()

    def _current_3d_load_signature(self) -> tuple:
        """Identify both the coordinate view and the mutable dataset inventory."""
        dataset_inventory = tuple(sorted(
            str(dataset.get("key") or "")
            for dataset in self.db.list_datasets()
            if str(dataset.get("key") or "")
        ))
        return (
            id(self.db),
            dataset_inventory,
            int(self.db.path_count()),
            self.state.frame_min,
            self.state.frame_max,
            self._path_coordinate_mode,
        )

    def _reload_3d(self):
        """Load and render 3D background paths."""
        signature = self._current_3d_load_signature()
        if signature == self._last_3d_load_signature:
            self._path_coordinate_reload_pending = False
            return
        if self._path_coordinate_load_busy:
            # A range slider can emit many values while the worker is active.
            # Remember that newer state exists and load only that final state.
            self._path_coordinate_reload_pending = True
            return
        self._path_coordinate_load_busy = True
        self._path_coordinate_reload_pending = False
        self._active_3d_load_signature = signature
        if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
            self._viewer.set_path_coordinate_mode_busy(True)
        mode_label = "original paths" if self._path_coordinate_mode == "original" else "current paths"
        self._begin_loading(f"Loading 3D data ({mode_label})...")

        # Use QThread for non-blocking load
        self._3d_load_request_id += 1
        self._loader_thread = QThread()
        self._loader = DataLoader(
            self.db,
            self.state.frame_min,
            self.state.frame_max,
            self._path_coordinate_mode,
            self._3d_load_request_id,
        )
        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_3d_loaded)
        self._loader.failed.connect(self._on_3d_load_failed)
        self._loader.finished.connect(self._loader_thread.quit)
        self._loader.failed.connect(self._loader_thread.quit)
        self._loader.finished.connect(self._loader.deleteLater)
        self._loader.failed.connect(self._loader.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()

    def _on_3d_loaded(self, data, coordinate_mode: str, request_id: int):
        """Callback when 3D data is loaded."""
        if request_id != self._3d_load_request_id:
            return
        try:
            if coordinate_mode != self._path_coordinate_mode:
                return

            opacity = self._bg_opacity_slider.value() / 100.0
            self._viewer.render_background_paths(data, opacity=opacity)
            self._last_3d_load_signature = self._active_3d_load_signature
            self._invalidate_render_signatures()
            self._viewer.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)
            self._clear_path_delta_exclusion(reset_slider=False, refresh=False)
            self._refresh_3d_overlays()
            if self._path_change_overlay_active:
                self._apply_path_change_overlay()
            elif self._exit_delta_overlay_active:
                self._apply_exit_delta_overlay()
            if self._legacy_residue_combination_enabled:
                self._refresh_residue_combination_markers()
            mode_label = "original paths" if self._path_coordinate_mode == "original" else "current paths"
            self._end_loading(f"Loaded {len(data):,} 3D paths ({mode_label})", 3000)
        finally:
            self._path_coordinate_load_busy = False
            if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
                self._viewer.set_path_coordinate_mode_busy(False)
            if self._path_coordinate_reload_pending:
                QTimer.singleShot(0, self._reload_3d)

    def _on_3d_load_failed(self, error: str, coordinate_mode: str, request_id: int):
        if request_id != self._3d_load_request_id:
            return
        try:
            mode_label = "original paths" if coordinate_mode == "original" else "current paths"
            self._statusbar.showMessage(f"Failed to load 3D data ({mode_label}): {error}", 5000)
            self._end_loading()
        finally:
            self._path_coordinate_load_busy = False
            if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
                self._viewer.set_path_coordinate_mode_busy(False)
            if self._path_coordinate_reload_pending:
                QTimer.singleShot(0, self._reload_3d)

    # ─── Event handlers ─────────────────────────────────
    def _on_residue_changed(self):
        """Residue selection changed → update effective paths + table (charts debounced)."""
        t_total = time.perf_counter()

        if not self.state.selected_residues:
            self.state.set_residue_filtered_paths([])
            self._path_panel.refresh()
            self._viewer.clear_effective()
            self._clear_chart_views()
            self._residue_panel.load_options()
            self._sync_residue_combination_context()
            self._refresh_residue_combination_markers()
            t_elapsed = (time.perf_counter() - t_total) * 1000
            print(f"[PERF][ResidueClick] total={t_elapsed:.1f}ms (clear)")
            return

        t0 = time.perf_counter()
        matched_ids = self.db.get_matched_path_ids(self.state.selected_residues)
        t_query = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] query_paths={t_query:.1f}ms")

        t0 = time.perf_counter()
        self.state.set_residue_filtered_paths(matched_ids)
        t_recompute = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] recompute_effective={t_recompute:.1f}ms")

        t0 = time.perf_counter()
        self._path_panel.refresh()
        t_panel = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] update_path_panel={t_panel:.1f}ms")

        opts = self.db.get_residue_options(self.state.selected_residues)
        self._residue_panel.load_options(opts)
        self._sync_residue_combination_context()
        self._refresh_residue_combination_markers()

        self._schedule_chart_update()

        t_elapsed = (time.perf_counter() - t_total) * 1000
        print(f"[PERF][ResidueClick] total={t_elapsed:.1f}ms")
        self._statusbar.showMessage(f"Matched {len(matched_ids)} paths", 3000)

    # ─── Chart debounce ──────────────────────────────────
    def _schedule_chart_update(self):
        """Restart chart debounce timer (150ms)."""
        if not self._legacy_charts_enabled:
            if self._chart_timer.isActive():
                self._chart_timer.stop()
            return
        self._chart_timer.start()

    def _do_chart_update(self):
        """Update chart status label; clear charts when no selection."""
        if not self._legacy_charts_enabled:
            return
        effective = self.state.effective_path_ids
        if not effective:
            self._clear_chart_views()
            self._chart_status_label.setText("")
            return
        self._chart_status_label.setText(
            f"{len(effective)} paths selected — click Sync Charts to update"
        )

    def _cached_path_profiles(self, path_ids) -> tuple[list, bool]:
        """Return profiles for a stable path set using a small per-database LRU."""
        normalized_ids = tuple(sorted({int(path_id) for path_id in path_ids if int(path_id) >= 0}))
        cache_key = (id(self.db), normalized_ids)
        cached = self._chart_profile_cache.get(cache_key)
        if cached is not None:
            self._chart_profile_cache.move_to_end(cache_key)
            return cached, True

        profiles = list(self.db.get_path_profiles(list(normalized_ids)))
        max_cached_paths = int(getattr(self, "_chart_profile_cache_max_paths", 6_000))
        if len(normalized_ids) <= max_cached_paths:
            self._chart_profile_cache[cache_key] = profiles
            self._chart_profile_cache.move_to_end(cache_key)
            while (
                len(self._chart_profile_cache) > self._chart_profile_cache_limit
                or sum(len(key[1]) for key in self._chart_profile_cache) > max_cached_paths
            ):
                self._chart_profile_cache.popitem(last=False)
        return profiles, False

    def _sync_charts(self):
        """Manually render the complete active Tunnel Properties population."""
        if not self._legacy_charts_enabled:
            return
        if self._tunnel_properties_visible():
            path_ids = self._active_tunnel_property_path_ids()
            if not path_ids:
                self._clear_chart_views()
                self._chart_status_label.setText("No paths are available to synchronize")
                return
            self._chart_status_label.setText(
                f"Loading and rendering all {len(path_ids):,} paths..."
            )
            QApplication.processEvents()
            self._sync_tunnel_properties_for_paths(path_ids)
            return
        effective = self._visible_effective_path_ids()
        if not effective:
            self._clear_chart_views()
            self._chart_status_label.setText("")
            return

        ids = sorted(int(path_id) for path_id in effective)
        self._chart_status_label.setText(f"Computing {len(ids)} paths...")
        QApplication.processEvents()

        t0 = time.perf_counter()
        profiles, cache_hit = self._cached_path_profiles(ids)
        t_fetch = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._set_chart_profiles(profiles)
        t_chart_total = (time.perf_counter() - t0) * 1000

        self._chart_status_label.setText(
            f"{len(profiles)} paths — "
            f"{'cached' if cache_hit else f'fetch {t_fetch:.0f}ms'}, "
            f"charts {t_chart_total:.0f}ms"
        )
        print(
            f"[PERF][SyncCharts] paths={len(profiles)}, "
            f"cache_hit={cache_hit}, fetch={t_fetch:.1f}ms, charts={t_chart_total:.1f}ms"
        )

    def _tunnel_properties_visible(self) -> bool:
        panel = getattr(self, "_evidence_panel", None)
        return bool(
            getattr(self, "_tunnel_properties_enabled", False)
            and panel is not None
            and hasattr(panel, "tunnel_properties_visible")
            and panel.tunnel_properties_visible()
        )

    def _active_tunnel_property_path_ids(self) -> tuple[int, ...]:
        active = dict(getattr(self, "_dataset_compare_active", None) or {})
        path_scope = dict(active.get("path_scope") or {})
        path_ids = set(path_scope.get("reference_global_path_ids", []) or ())
        path_ids.update(path_scope.get("target_global_path_ids", []) or ())
        if not path_ids:
            path_ids = set(self._visible_effective_path_ids())
        return tuple(sorted(int(path_id) for path_id in path_ids if int(path_id) >= 0))

    def _sync_tunnel_properties_for_paths(self, path_ids) -> None:
        """Load all three detailed tunnel-property views only while visible."""
        if not self._tunnel_properties_visible():
            return
        normalized = tuple(sorted({int(path_id) for path_id in (path_ids or ()) if int(path_id) >= 0}))
        if not normalized:
            self._clear_chart_views()
            return
        signature = (id(self.db), normalized)
        if (
            signature == self._tunnel_properties_render_signature
            and self._chart_profiles
        ):
            self._chart_status_label.setText(
                f"{len(self._chart_profiles):,} paths · already rendered"
            )
            return
        started = time.perf_counter()
        profiles, cache_hit = self._cached_path_profiles(normalized)
        fetched = (time.perf_counter() - started) * 1000.0
        plotted_at = time.perf_counter()
        self._set_chart_profiles(profiles)
        plotted = (time.perf_counter() - plotted_at) * 1000.0
        self._tunnel_properties_render_signature = signature
        self._chart_status_label.setText(
            f"{len(profiles):,} paths · "
            f"{'cache' if cache_hit else f'load {fetched:.0f} ms'} · "
            f"render {plotted:.0f} ms"
        )
        print(
            f"[PERF][TunnelProperties] paths={len(profiles)}, cache_hit={cache_hit}, "
            f"load={fetched:.1f}ms, render={plotted:.1f}ms"
        )

    def _sync_combination_regions_for_paths(self, path_ids) -> None:
        """Backward-compatible entry point for the unified properties page."""
        self._sync_tunnel_properties_for_paths(path_ids)

    def _on_tunnel_properties_requested(self) -> None:
        """Entering the page never computes; Sync Charts is the sole trigger."""
        path_ids = self._active_tunnel_property_path_ids()
        signature = (id(self.db), path_ids)
        if signature == self._tunnel_properties_render_signature and self._chart_profiles:
            self._chart_status_label.setText(
                f"{len(self._chart_profiles):,} paths · already rendered"
            )
        elif path_ids:
            self._chart_status_label.setText(
                f"{len(path_ids):,} paths ready · click Sync Charts to render all paths"
            )
        else:
            self._chart_status_label.setText(
                "Select a Dataset Compare mapping, then click Sync Charts"
            )

    # ─── Lasso selection ─────────────────────────────────
    def _toggle_lasso(self, checked: bool):
        self._viewer.set_lasso_mode(checked)
        if checked:
            self._statusbar.showMessage(
                "Force lasso mode ON — plain left-drag=lasso. Shift+left-drag always works.",
                5000,
            )
        else:
            self._statusbar.showMessage("Force lasso mode OFF — Shift+left-drag still works", 3000)

    def _on_lasso_selected(self, path_ids, mode):
        incoming = len(path_ids) if path_ids is not None else 0
        print(f"[LASSO][L5][MainWindow] received lasso_selected mode={mode}, incoming={incoming}")
        if not path_ids:
            path_ids = set()
        self.state.apply_lasso(path_ids, mode)
        self._schedule_chart_update()

    def _on_effective_paths_changed(self):
        """Coalesce bursty effective-set signals into one display-frame update."""
        n = len(self.state.effective_path_ids)
        self._status_selected.setText(f"Effective: {n} paths")
        self._refresh_statistics_selection_label()
        self._effective_refresh_requested_at = time.perf_counter()
        timer = getattr(self, "_effective_render_timer", None)
        if timer is None:
            self._flush_effective_path_refresh()
            return
        timer.start()

    def _flush_effective_path_refresh(self):
        """Render only the newest effective set after event bursts settle."""
        started = time.perf_counter()
        t0 = time.perf_counter()
        self._render_effective_paths_if_needed()
        t_3d = (time.perf_counter() - t0) * 1000
        if self._legacy_residue_combination_enabled:
            self._refresh_residue_combination_contained_match_status()
        # Refresh entry/exit point highlight if visible
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
        self._sync_observer_frame_to_selection()
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays()
        total = (time.perf_counter() - started) * 1000
        queued = 0.0
        if self._effective_refresh_requested_at:
            queued = max(0.0, (started - self._effective_refresh_requested_at) * 1000)
        print(
            f"[PERF][EffectiveRefresh] paths={len(self.state.effective_path_ids)}, "
            f"queue={queued:.1f}ms, update_3d={t_3d:.1f}ms, total={total:.1f}ms"
        )

    def _manual_update_residue_combination_compare(self, mode: str = "dataset"):
        if (
            not getattr(self, "_legacy_residue_combination_enabled", True)
            or not getattr(self, "_residue_combination_panel", None)
        ):
            return
        panel = self._residue_combination_panel
        prefer_explicit_selection = str(mode or "").strip().lower() == "path"
        scope_label = "path" if prefer_explicit_selection else "dataset"
        self._begin_loading(f"Updating residue-combination compare context by {scope_label}...")
        finished = False
        total_started = time.perf_counter()
        panel.begin_view_update_batch()
        try:
            phase_started = time.perf_counter()
            if prefer_explicit_selection:
                profiles_by_dataset_key = self._refresh_residue_combination_current_paths(
                    prefer_explicit_selection=True,
                )
            else:
                # Dataset scope uses the complete precomputed CSV/NPY inputs and
                # must not inherit the current table, lasso, or preview selection.
                profiles_by_dataset_key = {}
                panel.set_current_path_combination_keys(set())
                panel.set_current_path_combination_rows({}, {})
            current_paths_ms = (time.perf_counter() - phase_started) * 1000.0

            phase_started = time.perf_counter()
            if prefer_explicit_selection and panel.current_bottleneck_filter_active():
                self._refresh_residue_combination_current_bottlenecks(
                    prefer_explicit_selection=True,
                    profiles_by_dataset_key=profiles_by_dataset_key,
                )
            else:
                # Bottleneck rows are only needed when that filter is enabled.
                # Clear stale context so the first toggle requests a fresh load.
                panel.set_current_bottleneck_combination_keys(set())
                panel.set_current_bottleneck_combination_rows({}, {})
            bottlenecks_ms = (time.perf_counter() - phase_started) * 1000.0
            self._refresh_residue_combination_contained_match_status()

            phase_started = time.perf_counter()
            self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
            distances_ms = (time.perf_counter() - phase_started) * 1000.0
            self._sync_residue_combination_context()
            # All setters above may request a refresh. Defer the actual table
            # rebuild until the complete path context is ready.
            panel._refresh_view()
            finished = True
        finally:
            render_started = time.perf_counter()
            panel.end_view_update_batch()
            render_ms = (time.perf_counter() - render_started) * 1000.0
            if finished:
                total_ms = (time.perf_counter() - total_started) * 1000.0
                print(
                    f"[PERF][Load{scope_label.title()}] paths={current_paths_ms:.1f}ms "
                    f"bottlenecks={bottlenecks_ms:.1f}ms distances={distances_ms:.1f}ms "
                    f"render={render_ms:.1f}ms total={total_ms:.1f}ms"
                )
                self._end_loading(f"Residue Combination Compare updated by {scope_label}", 3000)
            else:
                self._end_loading()

    def _clear_selection(self):
        """Esc: clear lasso edits, restore to pure residue filter result."""
        self.state.clear_lasso_edits()
        self._schedule_chart_update()

    def _on_category_visibility_changed(self):
        """Category visibility toggled → render visible categories in 3D."""
        t_total = time.perf_counter()

        # Build visible overlay payloads from cache (no DB query)
        t0 = time.perf_counter()
        visible_categories = self._visible_category_payload()
        visible_datasets = self._visible_dataset_payload()
        t_fetch = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._last_3d_overlay_signature = None
        self._refresh_3d_overlays()
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
        elif self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
        t_render = (time.perf_counter() - t0) * 1000
        t_bg = 0.0
        t_ee = 0.0

        total_paths = sum(len(ids) for ids, _ in visible_categories.values())
        t_elapsed = (time.perf_counter() - t_total) * 1000

        print(f"[PERF][CategoryClick] fetch_category_ids={t_fetch:.1f}ms")
        print(f"[PERF][CategoryClick] update_3d={t_render:.1f}ms")
        print(f"[PERF][CategoryClick] update_bg_exclusion={t_bg:.1f}ms")
        print(f"[PERF][CategoryClick] update_entry_exit={t_ee:.1f}ms")
        print(f"[PERF][CategoryClick] total={t_elapsed:.1f}ms")

        self._statusbar.showMessage(
            f"Showing {len(visible_categories)} categories ({total_paths} paths)", 3000
        )

    def _on_dataset_display_changed(self):
        self._refresh_3d_overlays()
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
        elif self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
        self._refresh_observer_path_overlays()
        self._dataset_panel.refresh()
        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
            self._residue_panel.load_options()
        if self._legacy_charts_enabled and self._chart_profiles:
            path_color_map = self._chart_path_color_map(self._chart_profiles)
            self._profile_chart.set_path_color_map(path_color_map)
            self._statistics_chart.set_path_color_map(path_color_map)
            if self._chart_workspace is not None:
                self._chart_workspace.set_path_color_map(path_color_map)

    def _on_dataset_color_changed(self, key: str, color: str):
        self.state.set_dataset_color(key, color)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        self._statusbar.showMessage(f"Dataset {label} tunnel color: {color}", 3000)

    def _on_dataset_selection_changed(self, dataset_keys):
        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_dataset_filter(dataset_keys)
            self._residue_panel.load_options()
        active = getattr(self, "_dataset_compare_active", None)
        current = self._current_dataset_compare_reference()
        if active and (current is None or (active.get("reference_key"), int(active.get("reference_cluster", -1))) != current):
            self._clear_dataset_compare_observer_selection()
            self._dataset_compare_active = None
            self._observer_change_events = []
            self._refresh_observer_change_event_selector()
            self._viewer.clear_matched()
            self._viewer.clear_dataset_compare_motifs(render=False)
            self._viewer.clear_dataset_compare_residue_hotspots(render=False)
            self._viewer.clear_dataset_compare_bottlenecks(render=False)
            self._viewer.set_protein_highlight_residues([], refresh=True, render=True)
            self._viewer.clear_residues()
        self._refresh_dataset_compare_cluster_options()

    def _on_dataset_cluster_mode_changed(self, key: str, enabled: bool):
        self.state.set_dataset_cluster_display(key if enabled else None)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        mode_text = "cluster colors" if enabled else "dataset color"
        self._statusbar.showMessage(f"Dataset {label} render mode: {mode_text}", 3000)

    def _on_dataset_cluster_labels_changed(self, key: str, enabled: bool):
        self.state.set_dataset_cluster_labels(key if enabled else None)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        mode_text = "cluster labels" if enabled else "labels hidden"
        self._statusbar.showMessage(f"Dataset {label}: {mode_text}", 3000)

    def _on_dataset_cluster_paths_selected(self, path_ids):
        selected = {int(pid) for pid in (path_ids or [])}
        self.state.apply_lasso(selected, "replace")
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare_observer_selection()
            self._dataset_compare_active = None
            self._observer_change_events = []
            self._refresh_observer_change_event_selector()
            self._viewer.clear_matched()
            self._viewer.clear_dataset_compare_motifs(render=False)
            self._viewer.clear_dataset_compare_residue_hotspots(render=False)
            self._viewer.clear_dataset_compare_bottlenecks(render=False)
            self._viewer.set_protein_highlight_residues([], refresh=True, render=True)
            self._viewer.clear_residues()
        self._refresh_dataset_compare_cluster_options()
        self._schedule_chart_update()
        if selected:
            self._statusbar.showMessage(f"Selected {len(selected)} paths by cluster", 3000)
        else:
            self._statusbar.showMessage("Cleared cluster path selection", 3000)

    # ─── Aligned multi-dataset comparison ─────────────────
    @staticmethod
    def _read_compare_json(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _dataset_compare_root(self, dataset_key: str) -> str:
        binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        if binding is None:
            return ""
        current = os.path.abspath(str(binding.folder or ""))
        for _ in range(6):
            if os.path.isfile(os.path.join(current, "cluster_matches.json")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return ""

    def _dataset_compare_center_data(self, dataset_key: str) -> dict:
        binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        if binding is None:
            return {}
        path = os.path.join(str(binding.folder), "cluster_center_paths.json")
        payload = self._read_compare_json(path)
        if payload:
            return payload
        # The UI may load the original database from TTM_Out while the
        # alignment pipeline writes center metadata into a sibling output
        # directory (for example TTM_Out_aligned_framefix). Resolve centers
        # by dataset label so users do not have to reload aligned databases.
        label_candidates = {
            self._dataset_compare_label(dataset_key).lower(),
            os.path.basename(str(binding.folder).rstrip(os.sep)).lower(),
            str(getattr(binding, "name", "") or "").lower(),
        }
        center_index = getattr(self, "_dataset_compare_center_index", None)
        if center_index is None:
            center_index = {}
            search_root = os.path.abspath(str(binding.folder))
            for _ in range(5):
                parent = os.path.dirname(search_root)
                if parent == search_root:
                    break
                search_root = parent
                if os.path.basename(search_root).lower() == "pathvisual_out":
                    break
            # Prefer sibling alignment outputs (for example
            # TTM_Out_aligned_framefix) so the first lookup does not walk the
            # entire raw trajectory tree.
            try:
                metadata_roots = []
                for entry in os.scandir(search_root):
                    if entry.is_dir() and "align" in entry.name.lower():
                        metadata_roots.append(entry.path)
                if not metadata_roots:
                    metadata_roots = [search_root]
                for metadata_root in metadata_roots:
                    for root, _dirs, files in os.walk(metadata_root):
                        if "cluster_center_paths.json" not in files:
                            continue
                        candidate = os.path.join(root, "cluster_center_paths.json")
                        item = self._read_compare_json(candidate)
                        dataset_label = str(item.get("dataset") or os.path.basename(root)).lower()
                        center_index.setdefault(dataset_label, candidate)
                        center_index.setdefault(os.path.basename(root).lower(), candidate)
            except OSError:
                center_index = {}
            self._dataset_compare_center_index = center_index
        for candidate_label in label_candidates:
            candidate_path = center_index.get(candidate_label)
            if candidate_path:
                payload = self._read_compare_json(candidate_path)
                if payload:
                    return payload
        # Fallback for an aligned DB without center metadata: expose cluster IDs
        # from the database, while keeping comparison disabled until centers exist.
        return {"clusters": {str(cid): {} for cid in getattr(self.db, "get_dataset_cluster_path_groups", lambda *_a: {})(dataset_key)} }

    def _dataset_compare_label(self, dataset_key: str) -> str:
        dataset = self.db.get_dataset(dataset_key) if hasattr(self.db, "get_dataset") else None
        if dataset:
            name = str(dataset.get("name") or "")
            folder_name = os.path.basename(str(dataset.get("folder") or "").rstrip(os.sep))
            if name.lower() in {"preprocessed_paths.db", "preprocessed_paths.pkl", "tunnel_data.db", "tunnel.db"} and folder_name:
                return folder_name
            return name or folder_name or str(dataset_key)
        return str(dataset_key)

    def _refresh_dataset_compare_cluster_options(self):
        panel = getattr(self, "_dataset_compare_panel", None)
        if panel is None:
            return
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        current = self._current_dataset_compare_reference()
        selected_source = panel.source_dataset_key() if hasattr(panel, "source_dataset_key") else ""
        if not selected_source and current is not None:
            selected_source = str(current[0])
        panel.set_source_datasets(datasets, selected_key=selected_source)
        # Dataset Compare owns the cross-dataset selection.  Persist its
        # source + target set immediately so the first Observer opening is not
        # limited to the legacy two-dataset default.
        self._sync_compare_targets_to_observer()
        reference_key = panel.source_dataset_key() if hasattr(panel, "source_dataset_key") else ""
        if not reference_key:
            panel.set_reference_context()
            panel.set_comparison_rows([])
            return
        groups = self.db.get_dataset_cluster_path_groups(reference_key)
        ref_cluster = int(current[1]) if current is not None and str(current[0]) == reference_key else None
        path_count = len(groups.get(int(ref_cluster), set())) if ref_cluster is not None else None
        panel.set_reference_context(
            dataset_label=self._dataset_compare_label(reference_key),
            cluster_id=ref_cluster,
            path_count=path_count,
            cluster_count=len(groups),
        )

    def _current_dataset_compare_reference(self) -> tuple[str, int] | None:
        """Resolve the checked/current cluster in the Datasets tree."""
        dataset_panel = getattr(self, "_dataset_panel", None)
        if dataset_panel is None:
            return None
        checked = dataset_panel.checked_cluster_keys()
        current = dataset_panel.current_cluster_key()
        if current is not None:
            return current
        if len(checked) == 1:
            return next(iter(checked))
        effective = sorted(int(path_id) for path_id in getattr(self.state, "effective_path_ids", set()) or set())
        if effective:
            try:
                dataset_key = self.db.get_dataset_key_for_path_id(effective[0])
                profiles = self.db.get_path_profiles([effective[0]])
                cluster_id = int(profiles[0].get("cluster_id", 0)) if profiles else 0
                if dataset_key and cluster_id:
                    return str(dataset_key), cluster_id
            except Exception:
                pass
        return None

    @staticmethod
    def _resample_compare_points(points, count: int):
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
            return None
        values = values[:, :3]
        # Compare equal physical positions along the tunnel, not equal point
        # indices. Different CAVER paths may have different point densities.
        segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        source = (
            cumulative / total_length
            if total_length > 1e-12
            else np.linspace(0.0, 1.0, values.shape[0])
        )
        target = np.linspace(0.0, 1.0, count)
        return np.column_stack([np.interp(target, source, values[:, axis]) for axis in range(3)])

    def _compare_center_distances(self, reference_points, target_points) -> dict | None:
        reference = self._resample_compare_points(reference_points, max(2, len(reference_points or [])))
        if reference is None:
            return None
        target = self._resample_compare_points(target_points, reference.shape[0])
        if target is None:
            return None
        delta = reference - target
        distances = np.linalg.norm(delta, axis=1)
        return {
            "center_rmsd": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
            "mean_point_distance": float(np.mean(distances)),
            "endpoint_distance": float(np.linalg.norm(delta[-1])),
        }

    def _dataset_compare_cluster_residue_profiles(self, dataset_key: str) -> dict[int, dict[int, float]]:
        """Return per-cluster path-frequency fingerprints for residue identity."""
        cache = getattr(self, "_dataset_compare_residue_profile_cache", None)
        if cache is None:
            cache = {}
            self._dataset_compare_residue_profile_cache = cache
        key = str(dataset_key)
        if key in cache:
            return cache[key]
        binding = getattr(self.db, "_datasets_by_key", {}).get(key)
        connection = getattr(getattr(binding, "db", None), "conn", None)
        if connection is None:
            cache[key] = {}
            return {}
        try:
            path_totals = {
                int(cluster_id): max(1, int(count or 0))
                for cluster_id, count in connection.execute(
                    "SELECT cluster_id, COUNT(*) FROM paths GROUP BY cluster_id"
                )
            }
            residue_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            has_residue_paths = bool(connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='residue_paths'"
            ).fetchone())
            if has_residue_paths:
                for cluster_id, _path_id, residue_id in connection.execute(
                    "SELECT p.cluster_id, rp.path_id, rp.residue_id "
                    "FROM residue_paths rp JOIN paths p ON p.id = rp.path_id"
                ):
                    residue_id = int(residue_id or 0)
                    if residue_id > 0:
                        residue_counts[int(cluster_id)][residue_id] += 1
            else:
                current_pair = None
                path_residues: set[int] = set()

                def flush_path() -> None:
                    if current_pair is None:
                        return
                    cluster_id, _path_id = current_pair
                    for residue_id in path_residues:
                        residue_counts[int(cluster_id)][int(residue_id)] += 1

                for row in connection.execute(
                    "SELECT p.cluster_id, pp.path_id, pp.res_1, pp.res_2, pp.res_3, pp.res_4 "
                    "FROM path_points pp JOIN paths p ON p.id = pp.path_id "
                    "ORDER BY p.cluster_id, pp.path_id, pp.seq"
                ):
                    pair = (int(row[0]), int(row[1]))
                    if current_pair is not None and pair != current_pair:
                        flush_path()
                        path_residues.clear()
                    current_pair = pair
                    for value in row[2:6]:
                        residue_id = int(value or 0)
                        if residue_id > 0:
                            path_residues.add(residue_id)
                flush_path()
            profiles = {
                int(cluster_id): {
                    int(residue_id): float(count) / max(1, path_totals.get(int(cluster_id), 1))
                    for residue_id, count in counts.items()
                }
                for cluster_id, counts in residue_counts.items()
            }
        except Exception:
            profiles = {}
        cache[key] = profiles
        return profiles

    @staticmethod
    def _dataset_compare_weighted_jaccard(left: dict, right: dict) -> float | None:
        keys = set(left or {}) | set(right or {})
        if not keys:
            return None
        shared = sum(min(float((left or {}).get(key, 0.0)), float((right or {}).get(key, 0.0))) for key in keys)
        union = sum(max(float((left or {}).get(key, 0.0)), float((right or {}).get(key, 0.0))) for key in keys)
        return float(shared / union) if union > 1e-12 else None

    def _dataset_compare_pair_metrics(self, reference_key: str, target_key: str) -> dict[tuple[int, int], dict]:
        """Compute reusable geometry+composition relationship metrics."""
        cache = getattr(self, "_dataset_compare_relation_cache", None)
        if cache is None:
            cache = {}
            self._dataset_compare_relation_cache = cache
        cache_key = (str(reference_key), str(target_key))
        if cache_key in cache:
            return cache[cache_key]
        reference_clusters = self._dataset_compare_center_data(reference_key).get("clusters") or {}
        target_clusters = self._dataset_compare_center_data(target_key).get("clusters") or {}
        reference_ids = sorted(int(key) for key, value in reference_clusters.items() if isinstance(value, dict) and value.get("points"))
        target_ids = sorted(int(key) for key, value in target_clusters.items() if isinstance(value, dict) and value.get("points"))
        if not reference_ids or not target_ids:
            cache[cache_key] = {}
            return {}
        reference_profiles = self._dataset_compare_cluster_residue_profiles(reference_key)
        target_profiles = self._dataset_compare_cluster_residue_profiles(target_key)
        reference_groups = self.db.get_dataset_cluster_path_groups(reference_key)
        target_groups = self.db.get_dataset_cluster_path_groups(target_key)
        metrics: dict[tuple[int, int], dict] = {}
        for reference_id in reference_ids:
            ref_points = reference_clusters[str(reference_id)].get("points")
            for target_id in target_ids:
                target_points = target_clusters[str(target_id)].get("points")
                metric = self._compare_center_distances(ref_points, target_points)
                if metric is not None:
                    residue_similarity = self._dataset_compare_weighted_jaccard(
                        reference_profiles.get(reference_id, {}),
                        target_profiles.get(target_id, {}),
                    )
                    reference_count = len(reference_groups.get(reference_id, set()))
                    target_count = len(target_groups.get(target_id, set()))
                    path_count_similarity = (
                        float(min(reference_count, target_count)) / max(reference_count, target_count)
                        if reference_count and target_count
                        else None
                    )
                    metrics[(reference_id, target_id)] = {
                        **metric,
                        "residue_similarity": residue_similarity,
                        "path_count_similarity": path_count_similarity,
                        "reference_path_count": reference_count,
                        "target_path_count": target_count,
                        # Candidate ranking is computed entirely from the datasets
                        # currently loaded in Hex.  Imported match files must not
                        # act as a hidden prior in an analyst-facing comparison.
                        "pipeline_match": False,
                    }
        for metric in metrics.values():
            metric["relation_cost"] = float(metric["center_rmsd"])
        cache[cache_key] = metrics
        return metrics

    def _dataset_compare_global_assignment(self, reference_key: str, target_key: str) -> dict[int, dict]:
        """Build one-to-many mappings from the RMSD-valid target family."""
        reference_clusters = self._dataset_compare_center_data(reference_key).get("clusters") or {}
        target_clusters = self._dataset_compare_center_data(target_key).get("clusters") or {}
        reference_ids = sorted(int(key) for key, value in reference_clusters.items() if isinstance(value, dict) and value.get("points"))
        target_ids = sorted(int(key) for key, value in target_clusters.items() if isinstance(value, dict) and value.get("points"))
        metrics = self._dataset_compare_pair_metrics(reference_key, target_key)
        if not reference_ids or not target_ids or not metrics:
            return {}
        pair_key = (str(reference_key), str(target_key))
        locks = dict(getattr(self, "_dataset_compare_mapping_locks", {}).get(pair_key, {}) or {})
        threshold = max(
            0.0,
            float(getattr(self, "_dataset_compare_rmsd_distance_threshold", DEFAULT_RMSD_FAMILY_MARGIN)),
        )
        maximum = max(
            0.0,
            float(getattr(self, "_dataset_compare_rmsd_max_threshold", DEFAULT_RMSD_MAX_THRESHOLD)),
        )
        valid_locks: dict[int, tuple[int, ...]] = {}
        result: dict[int, dict] = {}
        for reference_id in reference_ids:
            candidates = [
                (int(target_id), metric)
                for (metric_reference_id, target_id), metric in metrics.items()
                if int(metric_reference_id) == int(reference_id)
                and metric.get("center_rmsd") is not None
                and np.isfinite(float(metric.get("center_rmsd")))
            ]
            candidates.sort(key=lambda item: (
                float(item[1].get("center_rmsd", float("inf"))),
                int(item[0]),
            ))
            if not candidates:
                result[reference_id] = {
                    "target_cluster_id": None,
                    "target_cluster_ids": (),
                    "matching_method": "No aligned center path",
                    "locked": False,
                    "rmsd_family_margin": threshold,
                    "rmsd_max_threshold": maximum,
                    "rmsd_family_limit": maximum,
                    "rmsd_family_size": 0,
                }
                continue
            candidates = [
                item for item in candidates
                if float(item[1]["center_rmsd"]) <= maximum + 1e-12
            ]
            if not candidates:
                result[reference_id] = {
                    "target_cluster_id": None,
                    "target_cluster_ids": (),
                    "matching_method": f"No match (best RMSD > {maximum:.2f} Å)",
                    "locked": False,
                    "rmsd_family_margin": threshold,
                    "rmsd_max_threshold": maximum,
                    "rmsd_family_size": 0,
                    "rmsd_family_cluster_ids": (),
                }
                continue
            best_rmsd = float(candidates[0][1]["center_rmsd"])
            family_limit = min(best_rmsd + threshold, maximum)
            family_ids = tuple(
                int(target_id) for target_id, metric in candidates
                if float(metric["center_rmsd"]) <= family_limit + 1e-12
            )
            raw_lock = locks.get(reference_id, ())
            if isinstance(raw_lock, (list, tuple, set)):
                requested_ids = tuple(int(value) for value in raw_lock)
            elif raw_lock is None:
                requested_ids = ()
            else:
                requested_ids = (int(raw_lock),)
            selected_ids = tuple(cluster_id for cluster_id in family_ids if cluster_id in requested_ids)
            locked = bool(selected_ids)
            if not locked:
                selected_ids = family_ids
            else:
                valid_locks[int(reference_id)] = selected_ids
            target_id = int(selected_ids[0]) if selected_ids else None
            if target_id is None:
                continue
            metric = metrics[(reference_id, target_id)]
            result[reference_id] = {
                "target_cluster_id": target_id,
                "target_cluster_ids": selected_ids,
                "matching_method": "locked family" if locked else "RMSD family",
                "locked": bool(locked),
                "rmsd_family_margin": threshold,
                "rmsd_max_threshold": maximum,
                "rmsd_family_limit": float(family_limit),
                "rmsd_family_size": len(family_ids),
                "rmsd_family_cluster_ids": family_ids,
                "in_rmsd_family": target_id in family_ids,
                **metric,
            }
        self._dataset_compare_mapping_locks[pair_key] = valid_locks
        return result

    def _dataset_compare_mapping_rows(
        self,
        reference_key: str,
        target_keys=None,
    ) -> list[dict]:
        """List each source-cluster mapping against the selected target dataset."""
        if target_keys is None:
            panel = getattr(self, "_dataset_compare_panel", None)
            if panel is not None and hasattr(panel, "target_dataset_keys"):
                allowed_targets: set[str] | None = set(panel.target_dataset_keys())
            else:
                allowed_targets = None
        else:
            allowed_targets = {str(key) for key in (target_keys or []) if str(key)}
        rows: list[dict] = []
        for dataset in self.db.list_datasets():
            target_key = str(dataset.get("key") or "")
            if not target_key or target_key == reference_key:
                continue
            if allowed_targets is not None and target_key not in allowed_targets:
                continue
            metrics = self._dataset_compare_pair_metrics(reference_key, target_key)
            assignment = self._dataset_compare_global_assignment(reference_key, target_key)
            for reference_id in sorted(assignment):
                assigned = dict(assignment[reference_id])
                target_options = []
                for (metric_reference_id, target_id), metric in metrics.items():
                    if int(metric_reference_id) != int(reference_id):
                        continue
                    family_limit = float(assigned.get("rmsd_family_limit", float("inf")))
                    maximum = float(assigned.get(
                        "rmsd_max_threshold",
                        getattr(self, "_dataset_compare_rmsd_max_threshold", DEFAULT_RMSD_MAX_THRESHOLD),
                    ))
                    if float(metric.get("center_rmsd", float("inf"))) > maximum + 1e-12:
                        continue
                    if float(metric.get("center_rmsd", float("inf"))) > family_limit + 1e-12:
                        continue
                    target_options.append({
                        "target_cluster_id": int(target_id),
                        "in_rmsd_family": True,
                        **dict(metric),
                    })
                target_options.sort(key=lambda row: (
                    float(row.get("center_rmsd", float("inf"))),
                    int(row.get("target_cluster_id", 0)),
                ))
                for rank, option in enumerate(target_options, start=1):
                    option["local_rank"] = rank
                assigned_target = assigned.get("target_cluster_id")
                target_id = int(assigned_target) if assigned_target is not None else None
                selected_option = next(
                    (row for row in target_options if target_id is not None and int(row["target_cluster_id"]) == target_id),
                    {},
                )
                rows.append({
                    "reference_key": str(reference_key),
                    "reference_cluster_id": int(reference_id),
                    "target_dataset": target_key,
                    "target_label": self._dataset_compare_label(target_key),
                    "target_cluster_id": target_id,
                    "target_cluster_ids": tuple(assigned.get("target_cluster_ids", ()) or ()),
                    "target_options": target_options,
                    **assigned,
                    **selected_option,
                })
        rows.sort(key=lambda row: (int(row.get("reference_cluster_id", 0)), str(row.get("target_label", ""))))
        return rows

    def _dataset_compare_mapping_sensitivity(self, active: dict) -> dict:
        reference_key = str(active.get("reference_key") or "")
        target_key = str(active.get("target_key") or "")
        reference_cluster = int(active.get("reference_cluster", 0) or 0)
        target_cluster = int(active.get("target_cluster", 0) or 0)
        metrics = self._dataset_compare_pair_metrics(reference_key, target_key)
        candidates = [
            {"target_cluster_id": int(candidate_cluster), **dict(metric)}
            for (candidate_reference, candidate_cluster), metric in metrics.items()
            if int(candidate_reference) == reference_cluster
        ]
        if not candidates or target_cluster <= 0:
            return {}
        return mapping_sensitivity(
            candidates,
            target_cluster,
            thresholds=(0.0,),
            scenarios=((f"RMSD +{self._dataset_compare_rmsd_distance_threshold:.2f} Å family", 0.0, 0.0),),
        )

    def _dataset_compare_temporal_evidence(self, active: dict, hotspot: dict) -> dict:
        """Recompute selected-region residue deltas in sequential frame blocks."""
        reference_key = str(active.get("reference_key") or "")
        target_key = str(active.get("target_key") or "")
        reference_cluster = int(active.get("reference_cluster", 0) or 0)
        target_cluster = int(active.get("target_cluster", 0) or 0)
        target_clusters = tuple(
            int(value) for value in (active.get("target_clusters", ()) or ()) if int(value) > 0
        ) or ((target_cluster,) if target_cluster > 0 else ())
        path_scope = dict(active.get("path_scope") or {})
        reference_ids = tuple(int(value) for value in path_scope.get("reference_path_ids", []) or [])
        target_by_cluster = dict(path_scope.get("target_path_ids_by_cluster", {}) or {})
        target_ids = tuple(dict.fromkeys(
            int(value)
            for cluster_id in target_clusters
            for value in (target_by_cluster.get(cluster_id, []) or [])
        ))
        if not target_ids:
            target_ids = tuple(int(value) for value in path_scope.get("target_path_ids", []) or [])
        cache_key = (
            reference_key, reference_cluster, target_key, target_clusters,
            reference_ids, target_ids,
            round(float(hotspot.get("start_fraction", 0.0)), 4),
            round(float(hotspot.get("end_fraction", 1.0)), 4),
        )
        cached = self._dataset_compare_temporal_evidence_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        reference_binding = getattr(self.db, "_datasets_by_key", {}).get(reference_key)
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(target_key)
        reference_db = getattr(reference_binding, "db", None)
        target_db = getattr(target_binding, "db", None)
        reference_connection = getattr(reference_db, "conn", None)
        target_connection = getattr(target_db, "conn", None)
        if reference_connection is None or target_connection is None or not reference_ids or not target_ids:
            return {"rows": [], "note": "Frame-resolved path populations are unavailable for this relation."}
        block_cache_key = cache_key[:-2]
        block_payload = self._dataset_compare_temporal_block_cache.get(block_cache_key)
        if block_payload is None:
            try:
                reference_blocks = split_path_blocks_by_frame(
                    reference_db.get_path_profiles(list(reference_ids)), 5,
                )
                target_blocks = split_path_blocks_by_frame(
                    target_db.get_path_profiles(list(target_ids)), 5,
                )
            except Exception:
                return {"rows": [], "note": "Frame IDs could not be read for the selected path populations."}
            block_count = min(len(reference_blocks), len(target_blocks))
            if block_count < 2:
                return {"rows": [], "note": "At least two sequential frame blocks are required for temporal inspection."}
            block_profiles = []
            for block_index in range(block_count):
                block_profiles.append(build_residue_change_profile(
                    reference_connection,
                    target_connection,
                    reference_cluster,
                    None,
                    reference_path_ids=reference_blocks[block_index]["path_ids"],
                    target_path_ids=target_blocks[block_index]["path_ids"],
                    bin_count=64,
                ))
            block_payload = {
                "reference_blocks": reference_blocks,
                "target_blocks": target_blocks,
                "block_profiles": block_profiles,
                "block_count": block_count,
            }
            self._dataset_compare_temporal_block_cache[block_cache_key] = block_payload
        reference_blocks = list(block_payload["reference_blocks"])
        target_blocks = list(block_payload["target_blocks"])
        block_count = int(block_payload["block_count"])
        deltas_by_block = [
            hotspot_residue_deltas(block_profile, hotspot)
            for block_profile in block_payload["block_profiles"]
        ]
        priority_ids = tuple(dict.fromkeys(
            int(value)
            for key in ("lost_residue_ids", "gained_residue_ids", "reference_residue_ids", "target_residue_ids")
            for value in (hotspot.get(key, ()) or ())
            if int(value) > 0
        ))
        if not priority_ids:
            magnitudes: dict[int, float] = defaultdict(float)
            for block in deltas_by_block:
                for residue_id, value in block.items():
                    magnitudes[int(residue_id)] += abs(float(value))
            priority_ids = tuple(
                residue_id for residue_id, _value in
                sorted(magnitudes.items(), key=lambda item: item[1], reverse=True)[:8]
            )
        reference_names = getattr(reference_db, "_residue_name_map", {}) or {}
        target_names = getattr(target_db, "_residue_name_map", {}) or {}
        rows = []
        for residue_id in priority_ids[:8]:
            name = str(target_names.get(residue_id) or reference_names.get(residue_id) or "UNK").upper()
            values = [float(block.get(residue_id, 0.0)) for block in deltas_by_block]
            rows.append({
                "residue_id": int(residue_id),
                "label": f"{name} MD{residue_id}",
                "values": values,
            })
        result = {
            "rows": rows,
            "block_count": block_count,
            "blocks": [
                {
                    "index": index + 1,
                    "phase": ("Early", "Early–Mid", "Middle", "Mid–Late", "Late")[
                        min(index, 4)
                    ],
                    "reference": {
                        key: reference_blocks[index][key]
                        for key in ("frame_start", "frame_end", "path_count")
                    },
                    "target": {
                        key: target_blocks[index][key]
                        for key in ("frame_start", "frame_end", "path_count")
                    },
                }
                for index in range(block_count)
            ],
            "note": (
                f"{block_count} sequential frame/path blocks within each selected dataset; "
                "blocks test directional persistence and are not independent replicas."
            ),
        }
        self._dataset_compare_temporal_evidence_cache[cache_key] = dict(result)
        return result

    def _dataset_compare_composition_evidence(
        self,
        active: dict,
        profile: dict,
        hotspot: dict,
    ) -> dict:
        """Build panel-c rows from the same selected arc-length interval."""
        if not hotspot:
            return {"rows": [], "note": "Select R1/R2/... in the full-path map."}
        reference_key = str(active.get("reference_key") or "")
        target_key = str(active.get("target_key") or "")
        reference_binding = getattr(self.db, "_datasets_by_key", {}).get(reference_key)
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(target_key)
        reference_names = getattr(getattr(reference_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        deltas = hotspot_residue_deltas(profile, hotspot)
        rows = []
        for residue_id, delta in sorted(deltas.items(), key=lambda item: (-abs(float(item[1])), int(item[0])))[:8]:
            if abs(float(delta)) < 0.015:
                continue
            residue_name = str(
                (target_names if float(delta) >= 0.0 else reference_names).get(int(residue_id))
                or target_names.get(int(residue_id))
                or reference_names.get(int(residue_id))
                or "UNK"
            ).upper()
            rows.append({
                "residue_id": int(residue_id),
                "sequence_id": int(residue_id) + 1,
                "label": f"{residue_name} S{int(residue_id) + 1}",
                "delta": float(delta),
                "direction": "target_enriched" if float(delta) >= 0.0 else "baseline_depleted",
            })
        region = str(hotspot.get("region_id") or hotspot.get("hotspot_id") or "selected region")
        start = float(hotspot.get("start_fraction", 0.0)) * 100.0
        end = float(hotspot.get("end_fraction", 1.0)) * 100.0
        return {
            "rows": rows,
            "region_id": region,
            "start_fraction": float(hotspot.get("start_fraction", 0.0)),
            "end_fraction": float(hotspot.get("end_fraction", 1.0)),
            "note": f"{region} · physical arc {start:.1f}–{end:.1f}% · values use the constrained path populations shown above.",
        }

    @staticmethod
    def _evidence_frame_sort_key(path: str) -> tuple:
        numbers = re.findall(r"\d+", os.path.basename(str(path or "")))
        return (int(numbers[-1]) if numbers else 10**12, os.path.basename(str(path or "")).lower(), str(path).lower())

    def _dataset_evidence_pdb_frames(self, dataset_key: str) -> tuple[str, list[str]]:
        """Resolve the same MD frame source used by Residue Observer."""
        dataset_key = str(dataset_key or "")
        cached = self._dataset_compare_frame_source_cache.get(dataset_key)
        if cached is not None:
            return str(cached[0]), list(cached[1])
        _label, folder = self._observer_resolve_sequence_source(dataset_key)
        if not folder or not os.path.isdir(folder):
            return "", []
        paths: list[str] = []
        try:
            for current_root, _dirs, files in os.walk(folder):
                for name in files:
                    if not name.lower().endswith(".pdb"):
                        continue
                    if name.lower() in {"representative_frame.pdb", "representative.pdb", "structure.pdb"}:
                        continue
                    paths.append(os.path.abspath(os.path.join(current_root, name)))
        except OSError:
            return folder, []
        paths.sort(key=self._evidence_frame_sort_key)
        self._dataset_compare_frame_source_cache[dataset_key] = (folder, list(paths))
        return folder, paths

    def _dataset_compare_distance_evidence(
        self,
        active: dict,
        profile: dict,
        hotspot: dict,
        *,
        background: bool = False,
    ) -> dict:
        """Build panel-e trajectory C-alpha distance summaries on demand."""
        substitutions = [dict(row) for row in profile.get("substitutions", []) or []]
        if not hotspot or not substitutions:
            return {
                "rows": [],
                "note": "An explicit substitution and a selected replacement region are required for the framewise distance audit.",
            }
        mutation = dict(hotspot.get("nearest_mutation") or substitutions[0])
        anchor_sequence_id = int(mutation.get("sequence_id", 0) or 0)
        if anchor_sequence_id <= 0:
            return {"rows": [], "note": "The substitution sequence identifier is unavailable."}
        sequence_offset = int(profile.get("sequence_offset", 1) or 0)
        selected_md_ids = tuple(dict.fromkeys(
            int(value)
            for key in ("lost_residue_ids", "gained_residue_ids", "reference_residue_ids", "target_residue_ids")
            for value in (hotspot.get(key, ()) or ())
            if int(value) > 0 and int(value) + sequence_offset != anchor_sequence_id
        ))[:12]
        selected_sequence_ids = tuple(int(value) + sequence_offset for value in selected_md_ids)
        if not selected_sequence_ids:
            return {"rows": [], "note": "No selected-region residues are available for the distance audit."}
        preload_md_ids = tuple(dict.fromkeys(
            int(value)
            for region in (profile.get("hotspots", []) or [hotspot])
            for key in ("lost_residue_ids", "gained_residue_ids", "reference_residue_ids", "target_residue_ids")
            for value in (dict(region).get(key, ()) or ())
            if int(value) > 0 and int(value) + sequence_offset != anchor_sequence_id
        ))
        preload_sequence_ids = tuple(
            int(value) + sequence_offset
            for value in preload_md_ids
        ) or selected_sequence_ids
        reference_key = str(active.get("reference_key") or "")
        target_key = str(active.get("target_key") or "")
        reference_folder, reference_paths = self._dataset_evidence_pdb_frames(reference_key)
        target_folder, target_paths = self._dataset_evidence_pdb_frames(target_key)
        cache_key = (
            reference_key, target_key, anchor_sequence_id, selected_sequence_ids,
            reference_folder, len(reference_paths), target_folder, len(target_paths),
        )
        summary_key = (
            reference_key, target_key, anchor_sequence_id, preload_sequence_ids,
            reference_folder, len(reference_paths), target_folder, len(target_paths),
        )
        cached = self._dataset_compare_distance_evidence_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        if not reference_paths or not target_paths:
            return {
                "rows": [],
                "anchor_label": str(mutation.get("label") or f"S{anchor_sequence_id}"),
                "note": "MD PDB frame folders could not be resolved for both selected datasets. Check MD_path.txt or the dataset MD root.",
                "source_folders": {"reference": reference_folder, "target": target_folder},
            }
        self._dataset_compare_distance_evidence_context[cache_key] = {
            "reference_key": reference_key,
            "target_key": target_key,
            "anchor_sequence_id": anchor_sequence_id,
            "anchor_label": str(mutation.get("label") or f"S{anchor_sequence_id}"),
            "selected_md_ids": selected_md_ids,
            "selected_sequence_ids": selected_sequence_ids,
            "reference_folder": reference_folder,
            "target_folder": target_folder,
            "threshold": float(profile.get("remote_distance_threshold", 10.0) or 10.0),
            "summary_key": summary_key,
            "preload_sequence_ids": preload_sequence_ids,
        }
        shared_summary = self._dataset_compare_distance_summary_cache.get(summary_key)
        if shared_summary is not None:
            return self._compose_dataset_compare_distance_evidence(
                cache_key,
                dict(shared_summary.get("reference") or {}),
                dict(shared_summary.get("target") or {}),
            )
        if background:
            thread = self._dataset_compare_distance_evidence_thread
            if thread is None or not thread.isRunning():
                self._start_dataset_compare_distance_evidence(
                    cache_key,
                    reference_paths,
                    target_paths,
                    anchor_sequence_id,
                    preload_sequence_ids,
                )
            return {
                "rows": [],
                "anchor_label": str(mutation.get("label") or f"S{anchor_sequence_id}"),
                "note": (
                    "Calculating framewise C-alpha distance medians and IQRs in the background. "
                    "The view will update automatically; subsequent visits use the cached result."
                ),
                "calculating": True,
            }
        reference_summary = summarize_ca_distance_frames(
            reference_paths, anchor_sequence_id, preload_sequence_ids, max_frames=500,
        )
        target_summary = summarize_ca_distance_frames(
            target_paths, anchor_sequence_id, preload_sequence_ids, max_frames=500,
        )
        self._dataset_compare_distance_summary_cache[summary_key] = {
            "reference": dict(reference_summary),
            "target": dict(target_summary),
        }
        return self._compose_dataset_compare_distance_evidence(
            cache_key, reference_summary, target_summary,
        )

    def _compose_dataset_compare_distance_evidence(
        self,
        cache_key: tuple,
        reference_summary: dict,
        target_summary: dict,
    ) -> dict:
        context = dict(self._dataset_compare_distance_evidence_context.get(tuple(cache_key)) or {})
        reference_key = str(context.get("reference_key") or "")
        target_key = str(context.get("target_key") or "")
        selected_md_ids = tuple(int(value) for value in context.get("selected_md_ids", ()) or ())
        selected_sequence_ids = tuple(int(value) for value in context.get("selected_sequence_ids", ()) or ())
        reference_by_id = {int(row["residue_id"]): dict(row) for row in reference_summary.get("rows", []) or []}
        target_by_id = {int(row["residue_id"]): dict(row) for row in target_summary.get("rows", []) or []}
        reference_binding = getattr(self.db, "_datasets_by_key", {}).get(reference_key)
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(target_key)
        reference_names = getattr(getattr(reference_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        rows = []
        for md_id, sequence_id in zip(selected_md_ids, selected_sequence_ids):
            reference_stats = reference_by_id.get(sequence_id)
            target_stats = target_by_id.get(sequence_id)
            if not reference_stats and not target_stats:
                continue
            name = str(target_names.get(md_id) or reference_names.get(md_id) or "UNK").upper()
            rows.append({
                "residue_id": int(md_id),
                "sequence_id": int(sequence_id),
                "label": f"{name} S{sequence_id}",
                "reference": reference_stats or {},
                "target": target_stats or {},
            })
        ref_frames = int(reference_summary.get("readable_frame_count", 0) or 0)
        target_frames = int(target_summary.get("readable_frame_count", 0) or 0)
        result = {
            "rows": rows,
            "anchor_sequence_id": int(context.get("anchor_sequence_id", 0) or 0),
            "anchor_label": str(context.get("anchor_label") or "Mutation"),
            "threshold": float(context.get("threshold", 10.0) or 10.0),
            "reference_frame_count": ref_frames,
            "target_frame_count": target_frames,
            "source_folders": {
                "reference": str(context.get("reference_folder") or ""),
                "target": str(context.get("target_folder") or ""),
            },
            "statistic": "framewise C-alpha Euclidean distance; median and interquartile range",
            "note": (
                f"Baseline n={ref_frames}, target n={target_frames} readable MD frames · "
                "rigid display alignment is distance-invariant · descriptive within-trajectory audit."
            ),
        }
        self._dataset_compare_distance_evidence_cache[cache_key] = dict(result)
        return result

    def _start_dataset_compare_distance_evidence(
        self,
        cache_key: tuple,
        reference_paths: list[str],
        target_paths: list[str],
        anchor_sequence_id: int,
        selected_sequence_ids: tuple[int, ...],
    ) -> None:
        thread = QThread(self)
        worker = DistanceEvidenceTaskWorker(
            cache_key,
            reference_paths,
            target_paths,
            anchor_sequence_id,
            selected_sequence_ids,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_dataset_compare_distance_evidence_ready)
        worker.failed.connect(self._on_dataset_compare_distance_evidence_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_dataset_compare_distance_evidence_thread_finished)
        self._dataset_compare_distance_evidence_thread = thread
        self._dataset_compare_distance_evidence_worker = worker
        thread.start()

    def _on_dataset_compare_distance_evidence_ready(
        self,
        cache_key: object,
        reference_summary: object,
        target_summary: object,
    ) -> None:
        context = dict(
            self._dataset_compare_distance_evidence_context.get(tuple(cache_key)) or {}
        )
        summary_key = tuple(context.get("summary_key") or ())
        if summary_key:
            self._dataset_compare_distance_summary_cache[summary_key] = {
                "reference": dict(reference_summary or {}),
                "target": dict(target_summary or {}),
            }
        self._compose_dataset_compare_distance_evidence(
            tuple(cache_key), dict(reference_summary or {}), dict(target_summary or {}),
        )

    def _on_dataset_compare_distance_evidence_failed(self, cache_key: object, message: str) -> None:
        self._dataset_compare_distance_evidence_cache[tuple(cache_key)] = {
            "rows": [],
            "note": f"Framewise C-alpha distance calculation failed: {message}",
        }

    def _on_dataset_compare_distance_evidence_thread_finished(self) -> None:
        self._dataset_compare_distance_evidence_thread = None
        self._dataset_compare_distance_evidence_worker = None
        QTimer.singleShot(0, lambda: self._refresh_evidence_panel(include_temporal=True))

    def _refresh_evidence_panel(
        self,
        hotspot: dict | None = None,
        *,
        include_temporal: bool | None = None,
    ) -> None:
        panel = getattr(self, "_evidence_panel", None)
        active = dict(getattr(self, "_dataset_compare_active", {}) or {})
        if panel is None:
            return
        if not active:
            panel.set_evidence({})
            return
        profile = dict(active.get("residue_change_profile") or {})
        hotspots = [dict(row) for row in profile.get("hotspots", []) or []]
        selected = dict(hotspot or {})
        if not selected:
            region_id = str(getattr(self.state, "analysis_selection", {}).get("region_id") or "")
            selected = next((row for row in hotspots if str(row.get("region_id") or row.get("hotspot_id") or "") == region_id), hotspots[0] if hotspots else {})
        selected_region = str(selected.get("region_id") or selected.get("hotspot_id") or "full path")
        if include_temporal is None:
            # EvidencePanel is a sibling of the main-view stack inside the
            # splitter, so it can never be that stack's current widget.
            include_temporal = str(
                getattr(self, "_bottom_analysis_mode", "charts") or "charts"
            ) == "evidence"
        reference_key = str(active.get("reference_key") or "")
        mapping_rows = self._dataset_compare_mapping_rows(reference_key) if reference_key else []
        profile_for_view = dict(profile)
        matching_remote = [
            dict(row)
            for row in profile.get("allosteric_evidence", []) or []
            if str(row.get("region_id") or "") == selected_region
        ]
        if matching_remote:
            profile_for_view["allosteric_evidence"] = matching_remote
        profile_for_view["selected_region"] = selected_region
        composition = self._dataset_compare_composition_evidence(active, profile, selected)
        bottleneck = {
            "reference": dict(profile.get("reference_bottleneck") or {}),
            "target": dict(profile.get("target_bottleneck") or {}),
            "reference_label": self._dataset_compare_label(reference_key),
            "target_label": self._dataset_compare_label(str(active.get("target_key") or "")),
            "position_basis": str(profile.get("position_basis") or "physical_arc_length"),
        }
        distance = (
            self._dataset_compare_distance_evidence(active, profile, selected, background=True)
            if selected and include_temporal
            else {
                "rows": [],
                "note": (
                    "Click an R region in the full-path map to calculate framewise mutation-to-region C-alpha distances."
                    if selected
                    else "Select R1/R2/... before calculating framewise distances."
                ),
            }
        )
        path_scope = dict(active.get("path_scope") or {})
        target_cluster = int(active.get("target_cluster", 0) or 0)
        target_clusters = tuple(
            int(value) for value in (active.get("target_clusters", ()) or ()) if int(value) > 0
        ) or ((target_cluster,) if target_cluster > 0 else ())
        target_cluster_label = ", ".join(f"C{cluster_id}" for cluster_id in target_clusters)
        target_ids = list(path_scope.get("target_path_ids", []) or [])
        payload = {
            **active,
            "reference_label": self._dataset_compare_label(reference_key),
            "target_label": self._dataset_compare_label(str(active.get("target_key") or "")),
            "selected_region": selected_region,
            "target_cluster_label": target_cluster_label,
            "mapping_sensitivity": self._dataset_compare_mapping_sensitivity(active),
            "ensemble_points": ensemble_consistency_points(
                mapping_rows, int(active.get("reference_cluster", 0) or 0),
            ),
            "composition": composition,
            "bottleneck": bottleneck,
            "distance": distance,
            "temporal": (
                self._dataset_compare_temporal_evidence(active, selected)
                if selected and include_temporal
                else {
                    "rows": [],
                    "note": (
                        "Click an R region in the full-path map to calculate sequential frame blocks."
                        if selected
                        else "Select R1/R2/... in the full-path map to calculate sequential block evidence."
                    ),
                }
            ),
            "residue_change_profile": profile_for_view,
            "provenance": {
                "reference_dataset_key": reference_key,
                "target_dataset_key": str(active.get("target_key") or ""),
                "reference_cluster": int(active.get("reference_cluster", 0) or 0),
                "target_cluster": target_cluster,
                "target_clusters": target_clusters,
                "target_cluster_label": target_cluster_label,
                "selected_region": selected_region,
                "reference_path_count": len(path_scope.get("reference_path_ids", []) or []),
                "target_path_count": len(target_ids),
                "position_basis": str(profile.get("position_basis") or "physical_arc_length"),
                "keep_ratio": path_scope.get("keep_ratio"),
                "residue_delta_definition": "target minus baseline path-population frequency",
                "bottleneck_summary": "median position with position IQR; median radius",
                "distance_summary": "framewise C-alpha distance median with IQR",
                "temporal_summary": "five sequential within-trajectory path/frame blocks",
            },
        }
        panel.set_evidence(payload)

    def _dataset_compare_candidates(self, reference_key: str, ref_cluster: int) -> list[dict]:
        """Compatibility view containing one mapping row per target dataset."""
        return [
            row for row in self._dataset_compare_mapping_rows(reference_key)
            if int(row.get("reference_cluster_id", -1)) == int(ref_cluster)
        ]

    def _dataset_compare_matched_path_scope(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        *,
        keep_ratio: float = 1.0,
    ) -> dict:
        """Resolve exact reference/target path populations across target clusters."""
        normalized_keep = round(max(0.05, min(1.0, float(keep_ratio))), 4)
        cache_key = (str(reference_key), int(ref_cluster), str(target_key), normalized_keep)
        cached = self._dataset_compare_path_scope_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        reference_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        reference_connection = getattr(getattr(reference_binding, "db", None), "conn", None)
        target_connection = getattr(getattr(target_binding, "db", None), "conn", None)
        if reference_connection is None or target_connection is None:
            return {"error": "Dataset database binding is unavailable."}
        result = match_reference_cluster_paths(
            reference_connection,
            target_connection,
            int(ref_cluster),
            keep_ratio=normalized_keep,
        )
        reference_base = int(getattr(reference_binding, "path_id_base", 0) or 0)
        target_base = int(getattr(target_binding, "path_id_base", 0) or 0)
        result["reference_global_path_ids"] = [
            reference_base + int(path_id) for path_id in result.get("reference_path_ids", [])
        ]
        result["target_global_path_ids"] = [
            target_base + int(path_id) for path_id in result.get("target_path_ids", [])
        ]
        result["target_global_path_ids_by_cluster"] = {
            int(cluster_id): [target_base + int(path_id) for path_id in path_ids]
            for cluster_id, path_ids in dict(result.get("target_path_ids_by_cluster", {}) or {}).items()
        }
        self._dataset_compare_path_scope_cache[cache_key] = dict(result)
        return result

    def _dataset_compare_unfiltered_cluster_scope(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        target_clusters,
    ) -> dict:
        """Return every path in the source cluster and selected target family."""
        reference_groups = self.db.get_dataset_cluster_path_groups(str(reference_key))
        target_groups = self.db.get_dataset_cluster_path_groups(str(target_key))
        reference_global = sorted(int(path_id) for path_id in reference_groups.get(int(ref_cluster), set()))
        if isinstance(target_clusters, (int, np.integer)):
            target_cluster_ids = (int(target_clusters),)
        else:
            target_cluster_ids = tuple(dict.fromkeys(int(value) for value in (target_clusters or ()) if int(value) > 0))
        target_global_by_cluster = {
            cluster_id: sorted(int(path_id) for path_id in target_groups.get(cluster_id, set()))
            for cluster_id in target_cluster_ids
        }
        target_global = sorted({
            path_id for path_ids in target_global_by_cluster.values() for path_id in path_ids
        })
        if not reference_global:
            return {"error": f"Source Cluster {int(ref_cluster)} contains no paths."}
        if not target_global:
            labels = ", ".join(f"C{cluster_id}" for cluster_id in target_cluster_ids) or "selection"
            return {"error": f"Target clusters {labels} contain no paths."}
        reference_local = self.db.get_local_path_ids_for_dataset(reference_key, reference_global)
        target_local = self.db.get_local_path_ids_for_dataset(target_key, target_global)
        target_local_by_cluster = {
            cluster_id: self.db.get_local_path_ids_for_dataset(target_key, path_ids)
            for cluster_id, path_ids in target_global_by_cluster.items()
            if path_ids
        }
        return {
            "filter_rule": "none",
            "keep_ratio": 1.0,
            "reference_path_ids": reference_local,
            "reference_global_path_ids": reference_global,
            "target_path_ids": target_local,
            "target_global_path_ids": target_global,
            "target_cluster_counts": {
                cluster_id: len(path_ids) for cluster_id, path_ids in target_local_by_cluster.items()
            },
            "target_path_ids_by_cluster": target_local_by_cluster,
            "target_global_path_ids_by_cluster": target_global_by_cluster,
            "prefiltered_count": len(target_local),
        }

    def _dataset_compare_geometry_regions(self, reference_points, target_points, *, max_regions: int = 6) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
        """Locate contiguous normalized path intervals with strong remodeling."""
        reference = self._resample_compare_points(reference_points, max(2, len(reference_points or [])))
        if reference is None:
            return [], None, None
        target = self._resample_compare_points(target_points, reference.shape[0])
        if target is None:
            return [], None, None
        displacements = np.linalg.norm(reference - target, axis=1)
        smooth_window = max(3, min(9, len(displacements) // 8 if len(displacements) >= 24 else 3))
        kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
        smoothed = np.convolve(displacements, kernel, mode="same")
        # Use a robust threshold so a global rigid residual does not become a
        # false local remodeling event.
        q25, q75 = np.percentile(smoothed, [25.0, 75.0])
        threshold = max(0.75, float(q75 + 0.10 * (q75 - q25)))
        mask = smoothed >= threshold
        if not np.any(mask):
            return [], reference, target
        # Fill tiny gaps so one remodeling event is not fragmented by noise.
        for index in range(1, len(mask) - 1):
            if not mask[index] and mask[index - 1] and mask[index + 1]:
                mask[index] = True
        intervals: list[tuple[int, int]] = []
        start = None
        for index, active in enumerate(mask):
            if active and start is None:
                start = index
            elif not active and start is not None:
                if index - start >= 2:
                    intervals.append((start, index - 1))
                start = None
        if start is not None and len(mask) - start >= 2:
            intervals.append((start, len(mask) - 1))
        length_ref = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
        length_target = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
        regions = []
        for start, end in intervals:
            regions.append({
                "start_index": start,
                "end_index": end,
                "start_fraction": float(start / max(1, len(mask) - 1)),
                "end_fraction": float(end / max(1, len(mask) - 1)),
                "mean_displacement": float(np.mean(displacements[start:end + 1])),
                "max_displacement": float(np.max(displacements[start:end + 1])),
                "reference_length": length_ref,
                "target_length": length_target,
                "threshold": threshold,
            })
        regions.sort(key=lambda row: (-float(row["max_displacement"]), -float(row["mean_displacement"])))
        return regions[:max_regions], reference, target

    def _dataset_compare_records(self, reference_key: str, target_key: str) -> tuple[list[dict], list[dict]]:
        root = self._dataset_compare_root(reference_key) or self._dataset_compare_root(target_key)
        if not root:
            return [], []
        reference_label = self._dataset_compare_label(reference_key)
        target_label = self._dataset_compare_label(target_key)
        matches_payload = self._read_compare_json(os.path.join(root, "cluster_matches.json"))
        changes_payload = self._read_compare_json(os.path.join(root, "residue_composition_changes.json"))
        def label_matches(row, field, expected):
            value = str(row.get(field, ""))
            return value in {expected, os.path.basename(expected), self._dataset_compare_label(expected)}
        matches = [
            row for row in matches_payload.get("matches", []) or []
            if label_matches(row, "reference_dataset", reference_label) and label_matches(row, "target_dataset", target_label)
        ]
        changes = [
            row for row in changes_payload.get("changes", []) or []
            if label_matches(row, "reference_dataset", reference_label) and label_matches(row, "target_dataset", target_label)
        ]
        return matches, changes

    def _dataset_compare_direct_changes(self, reference_key: str, ref_cluster: int, target_key: str, target_cluster: int) -> list[dict]:
        """Backward-compatible alias for path residue-combination changes."""
        return self._dataset_compare_combination_changes(
            reference_key, ref_cluster, target_key, target_cluster,
        )

    def _dataset_compare_combination_changes(self, reference_key: str, ref_cluster: int, target_key: str, target_cluster: int) -> list[dict]:
        """Compare 2/3/4-residue combinations at the path level.

        A combination is counted once per path, even if it occurs at many
        tunnel points. This makes the frequency a path-composition measure
        and prevents long paths from dominating the mutation comparison.
        """
        cache_key = (str(reference_key), int(ref_cluster), str(target_key), int(target_cluster))
        cache = getattr(self, "_dataset_compare_combination_cache", None)
        if cache is None:
            cache = {}
            self._dataset_compare_combination_cache = cache
        if cache_key in cache:
            return list(cache[cache_key])

        def counts(key: str, cluster_id: int) -> tuple[dict[tuple[int, ...], set[int]], int]:
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(key))
            if binding is None:
                return {}, 0
            try:
                path_rows = binding.db.conn.execute(
                    "SELECT id FROM paths WHERE cluster_id = ?", (int(cluster_id),)
                ).fetchall()
                path_ids = [int(row[0]) for row in path_rows]
                point_rows = binding.db.conn.execute(
                    "SELECT path_id, res_1, res_2, res_3, res_4 "
                    "FROM path_points WHERE path_id IN ("
                    + ",".join("?" * len(path_ids)) + ") ORDER BY path_id, seq",
                    path_ids,
                ).fetchall() if path_ids else []
                path_combinations: dict[tuple[int, ...], set[int]] = defaultdict(set)
                for row in point_rows:
                    residue_ids = sorted({
                        int(row[index] or 0)
                        for index in range(1, 5)
                        if int(row[index] or 0) > 0
                    })
                    for size in range(2, min(4, len(residue_ids)) + 1):
                        for combo in combinations(residue_ids, size):
                            path_combinations[tuple(combo)].add(int(row[0]))
                path_row = binding.db.conn.execute(
                    "SELECT COUNT(*) AS cnt FROM paths WHERE cluster_id = ?", (int(cluster_id),)
                ).fetchone()
            except Exception:
                return {}, 0
            return path_combinations, int(path_row[0] if path_row else 0)

        reference_counts, reference_total = counts(reference_key, ref_cluster)
        target_counts, target_total = counts(target_key, target_cluster)
        if not reference_total and not target_total:
            return []
        ref_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        ref_names = getattr(getattr(ref_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        rows: list[dict] = []
        for residue_id in sorted(set(reference_counts) | set(target_counts)):
            combo = tuple(int(value) for value in residue_id)
            ref_count = len(reference_counts.get(combo, set()))
            target_count = len(target_counts.get(combo, set()))
            ref_frequency = ref_count / reference_total if reference_total else 0.0
            target_frequency = target_count / target_total if target_total else 0.0
            if not ref_count:
                status = "added_in_target"
            elif not target_count:
                status = "removed_in_target"
            else:
                status = "shared"
            ref_label = "-".join(str(ref_names.get(value, "UNK")).upper() for value in combo)
            target_label = "-".join(str(target_names.get(value, "UNK")).upper() for value in combo)
            identity_change = ref_label != target_label and ref_label != "" and target_label != ""
            frequency_delta = target_frequency - ref_frequency
            # Keep the table focused on combinations affected by the
            # mutation, rather than listing every unchanged tunnel pair.
            if status == "shared" and abs(frequency_delta) < 0.05 and not identity_change:
                continue
            rows.append({
                "reference_dataset": self._dataset_compare_label(reference_key),
                "target_dataset": self._dataset_compare_label(target_key),
                "reference_cluster_id": int(ref_cluster),
                "target_cluster_id": int(target_cluster),
                "residue_ids": combo,
                "combination_size": len(combo),
                "combination_label": ";".join(str(value) for value in combo),
                "residue_label": f"{';'.join(str(value) for value in combo)} ({ref_label} -> {target_label})",
                "status": status,
                "display_status": "replacement" if identity_change else status,
                "identity_change": identity_change,
                "reference_count": ref_count,
                "target_count": target_count,
                "reference_frequency": ref_frequency,
                "target_frequency": target_frequency,
                "frequency_delta": frequency_delta,
                "reference_path_count": ref_count,
                "target_path_count": target_count,
            })
        rows.sort(key=lambda row: (-abs(float(row.get("frequency_delta", 0.0))), int(row.get("combination_size", 0)), tuple(row.get("residue_ids", ()))))
        cache[cache_key] = list(rows)
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
        return rows

    def _dataset_compare_residue_rows(self, reference_key: str, target_key: str, changes: list[dict]) -> tuple[list[dict], set[int]]:
        changed_ids: set[int] = set()
        for row in changes:
            residue_values = row.get("residue_ids", ())
            if not residue_values and row.get("residue_id"):
                residue_values = (row.get("residue_id"),)
            if not (bool(row.get("identity_change")) or str(row.get("status", "")) != "shared" or abs(float(row.get("frequency_delta", 0.0) or 0.0)) >= 0.05):
                continue
            changed_ids.update(int(value) for value in residue_values if int(value or 0) > 0)
        rows: list[dict] = []
        for key in (reference_key, target_key):
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(key))
            if binding is None:
                continue
            dataset_color = self.state.dataset_colors.get(str(key), "#4169E1")
            for item in binding.db.get_residue_positions():
                local_id = int(item.get("residue_id", 0) or 0)
                if local_id not in changed_ids:
                    continue
                rows.append({
                    "residue_id": binding.residue_id_base + local_id,
                    "label": f"{binding.prefix}:{local_id}",
                    "x": float(item["x"]), "y": float(item["y"]), "z": float(item["z"]),
                    "path_count": int(item.get("path_count", 0) or 0),
                    "dataset_key": str(key), "dataset_color": dataset_color,
                })
        return rows, changed_ids

    def _dataset_compare_region_combination_changes(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        target_cluster: int,
        regions: list[dict],
    ) -> list[dict]:
        """Find residue motifs inside each geometric remodeling interval."""
        if not regions:
            return []

        def counts(key: str, cluster_id: int, start_fraction: float, end_fraction: float):
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(key))
            if binding is None:
                return {}, 0
            try:
                path_rows = binding.db.conn.execute(
                    "SELECT id FROM paths WHERE cluster_id = ?", (int(cluster_id),)
                ).fetchall()
                path_ids = [int(row[0]) for row in path_rows]
                point_rows = binding.db.conn.execute(
                    "SELECT path_id, res_1, res_2, res_3, res_4 "
                    "FROM path_points WHERE path_id IN ("
                    + ",".join("?" * len(path_ids)) + ") ORDER BY path_id, seq",
                    path_ids,
                ).fetchall() if path_ids else []
                grouped: dict[int, list] = defaultdict(list)
                for row in point_rows:
                    grouped[int(row[0])].append(row)
                combination_paths: dict[tuple[int, ...], set[int]] = defaultdict(set)
                for path_id, path_points in grouped.items():
                    n_points = len(path_points)
                    for point_index, row in enumerate(path_points):
                        fraction = point_index / max(1, n_points - 1)
                        if fraction < start_fraction or fraction > end_fraction:
                            continue
                        residue_ids = sorted({
                            int(row[index] or 0)
                            for index in range(1, 5)
                            if int(row[index] or 0) > 0
                        })
                        for size in range(2, min(4, len(residue_ids)) + 1):
                            for combo in combinations(residue_ids, size):
                                combination_paths[tuple(combo)].add(path_id)
                return combination_paths, len(path_ids)
            except Exception:
                return {}, 0

        ref_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        ref_names = getattr(getattr(ref_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        result: list[dict] = []
        for region in regions:
            start_fraction = float(region.get("start_fraction", 0.0))
            end_fraction = float(region.get("end_fraction", 1.0))
            reference_counts, reference_total = counts(reference_key, ref_cluster, start_fraction, end_fraction)
            target_counts, target_total = counts(target_key, target_cluster, start_fraction, end_fraction)
            if not reference_total and not target_total:
                continue
            for combo in sorted(set(reference_counts) | set(target_counts)):
                ref_path_count = len(reference_counts.get(combo, set()))
                target_path_count = len(target_counts.get(combo, set()))
                ref_frequency = ref_path_count / reference_total if reference_total else 0.0
                target_frequency = target_path_count / target_total if target_total else 0.0
                ref_motif = "-".join(str(ref_names.get(value, "UNK")).upper() for value in combo)
                target_motif = "-".join(str(target_names.get(value, "UNK")).upper() for value in combo)
                identity_change = ref_motif != target_motif
                delta = target_frequency - ref_frequency
                if not identity_change and abs(delta) < 0.05:
                    continue
                changed_residues = [
                    str(value) for value in combo
                    if str(ref_names.get(value, "UNK")).upper() != str(target_names.get(value, "UNK")).upper()
                ]
                result.append({
                    "start_fraction": start_fraction,
                    "end_fraction": end_fraction,
                    "combination_size": len(combo),
                    "residue_ids": combo,
                    "reference_motif": ref_motif if ref_path_count else "(absent)",
                    "target_motif": target_motif if target_path_count else "(absent)",
                    "residue_label": f"{';'.join(changed_residues or [str(value) for value in combo])} ({ref_motif} -> {target_motif})",
                    "reference_path_count": ref_path_count,
                    "target_path_count": target_path_count,
                    "reference_frequency": ref_frequency,
                    "target_frequency": target_frequency,
                    "frequency_delta": delta,
                    "identity_change": identity_change,
                    "status": "replacement" if identity_change else ("added_in_target" if not ref_path_count else "removed_in_target" if not target_path_count else "shared"),
                })
        result.sort(key=lambda row: (
            not bool(row.get("identity_change")),
            -abs(float(row.get("frequency_delta", 0.0))),
            int(row.get("combination_size", 0)),
        ))
        return result[:100]

    def _dataset_compare_observer_change_events(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        target_cluster: int,
    ) -> list[dict]:
        """Build selectable baseline→mutant events for single residues and pairs."""

        def collect(dataset_key: str, cluster_id: int) -> dict[tuple[int, ...], dict]:
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
            database = getattr(binding, "db", None)
            connection = getattr(database, "conn", None)
            if connection is None:
                return {}
            try:
                total = int(connection.execute(
                    "SELECT COUNT(*) FROM paths WHERE cluster_id = ?",
                    (int(cluster_id),),
                ).fetchone()[0] or 0)
                max_seq = {
                    int(row[0]): max(1, int(row[1] or 0))
                    for row in connection.execute(
                        "SELECT pp.path_id, MAX(pp.seq) FROM path_points pp "
                        "JOIN paths p ON p.id = pp.path_id WHERE p.cluster_id = ? "
                        "GROUP BY pp.path_id",
                        (int(cluster_id),),
                    )
                }
                rows = connection.execute(
                    "SELECT pp.path_id, pp.seq, pp.res_1, pp.res_2, pp.res_3, pp.res_4 "
                    "FROM path_points pp JOIN paths p ON p.id = pp.path_id "
                    "WHERE p.cluster_id = ? ORDER BY pp.path_id, pp.seq",
                    (int(cluster_id),),
                ).fetchall()
            except Exception:
                return {}
            marked_paths: dict[tuple[int, ...], set[int]] = defaultdict(set)
            fractions: dict[tuple[int, ...], list[float]] = defaultdict(list)
            for row in rows:
                path_id = int(row[0])
                fraction = float(int(row[1] or 0) / max_seq.get(path_id, 1))
                residue_ids = sorted({
                    int(row[index] or 0)
                    for index in range(2, 6)
                    if int(row[index] or 0) > 0
                })
                keys = [(rid,) for rid in residue_ids]
                keys.extend(tuple(pair) for pair in combinations(residue_ids, 2))
                for key in keys:
                    marked_paths[key].add(path_id)
                    fractions[key].append(fraction)

            names = getattr(database, "_residue_name_map", {}) or {}
            positions: dict[int, np.ndarray] = {}
            try:
                for row in database.get_residue_positions():
                    rid = int(row.get("local_residue_id", row.get("residue_id", 0)) or 0)
                    if rid > 0:
                        positions[rid] = np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64)
            except Exception:
                positions = {}
            result = {}
            for ids, paths in marked_paths.items():
                coords = [positions[rid] for rid in ids if rid in positions]
                result[ids] = {
                    "ids": tuple(ids),
                    "motif": "-".join(str(names.get(rid, "UNK")).upper() for rid in ids),
                    "path_count": len(paths),
                    "total_paths": total,
                    "frequency": float(len(paths) / total) if total else 0.0,
                    "path_fraction": float(np.median(fractions.get(ids, [0.5]))),
                    "centroid": np.mean(coords, axis=0) if coords else None,
                    "pair_distance": (
                        float(np.linalg.norm(positions[ids[0]] - positions[ids[1]]))
                        if len(ids) == 2 and ids[0] in positions and ids[1] in positions
                        else None
                    ),
                }
            return result

        reference = collect(reference_key, ref_cluster)
        target = collect(target_key, target_cluster)
        if not reference and not target:
            return []

        def id_text(ids: tuple[int, ...]) -> str:
            return "–".join(str(rid) for rid in ids)

        def make_event(ref_item: dict | None, target_item: dict | None, event_type: str) -> dict:
            ref_item = dict(ref_item or {})
            target_item = dict(target_item or {})
            reference_ids = tuple(ref_item.get("ids") or ())
            target_ids = tuple(target_item.get("ids") or ())
            size = len(reference_ids or target_ids)
            selection_type = "single" if size == 1 else "pair"
            type_label = "Single" if size == 1 else "Pair"
            ref_motif = str(ref_item.get("motif") or "")
            target_motif = str(target_item.get("motif") or "")
            ref_display = f"{id_text(reference_ids)} {ref_motif}".strip() if reference_ids else "absent"
            target_display = f"{id_text(target_ids)} {target_motif}".strip() if target_ids else "absent"
            ref_count = int(ref_item.get("path_count", 0) or 0)
            target_count = int(target_item.get("path_count", 0) or 0)
            if event_type == "identity_substitution":
                effect = (
                    f"Amino-acid identity changed at the same path-contact position: "
                    f"baseline {ref_display}, mutant {target_display}."
                )
            elif event_type == "rewired":
                effect = (
                    f"Path contact was rewired: baseline {ref_display} is replaced by "
                    f"mutant {target_display}."
                )
            elif event_type == "contact_loss":
                effect = f"The mutant loses the baseline path contact {ref_display}."
            elif event_type == "contact_gain":
                effect = f"The mutant gains a new path contact {target_display}."
            else:
                direction = "increases" if target_count > ref_count else "decreases"
                effect = (
                    f"The occurrence of {ref_display or target_display} {direction} "
                    f"from {ref_count} baseline-cluster paths to {target_count} mutant-cluster paths."
                )
            ref_centroid = ref_item.get("centroid")
            target_centroid = target_item.get("centroid")
            spatial_shift = None
            if ref_centroid is not None and target_centroid is not None:
                spatial_shift = float(np.linalg.norm(ref_centroid - target_centroid))
            ref_pair_distance = ref_item.get("pair_distance")
            target_pair_distance = target_item.get("pair_distance")
            if size == 1 and spatial_shift is not None:
                if reference_ids == target_ids:
                    effect += f" Its aligned residue center shifts by {spatial_shift:.2f} Å."
                else:
                    effect += f" The replacement contact lies {spatial_shift:.2f} Å from the baseline contact center."
            elif size == 2 and ref_pair_distance is not None and target_pair_distance is not None:
                effect += (
                    f" Pair separation changes from {float(ref_pair_distance):.2f} Å "
                    f"to {float(target_pair_distance):.2f} Å "
                    f"(Δ {float(target_pair_distance) - float(ref_pair_distance):+.2f} Å)."
                )
            event_id = (
                f"{selection_type}:{','.join(map(str, reference_ids)) or 'none'}>"
                f"{','.join(map(str, target_ids)) or 'none'}:{event_type}"
            )
            return {
                "event_id": event_id,
                "selection_type": selection_type,
                "event_type": event_type,
                "transition_type": event_type,
                "combination_size": size,
                "reference_dataset_key": str(reference_key),
                "target_dataset_key": str(target_key),
                "reference_residue_ids": reference_ids,
                "target_residue_ids": target_ids,
                "reference_motif": ref_motif,
                "target_motif": target_motif,
                "reference_frequency": float(ref_item.get("frequency", 0.0) or 0.0),
                "target_frequency": float(target_item.get("frequency", 0.0) or 0.0),
                "reference_path_count": ref_count,
                "target_path_count": target_count,
                "reference_total_paths": int(ref_item.get("total_paths", 0) or 0),
                "target_total_paths": int(target_item.get("total_paths", 0) or 0),
                "spatial_shift": spatial_shift,
                "reference_pair_distance": ref_pair_distance,
                "target_pair_distance": target_pair_distance,
                "path_fraction": float(
                    np.mean([
                        value for value in (
                            ref_item.get("path_fraction"), target_item.get("path_fraction")
                        ) if value is not None
                    ])
                ),
                "present_reference": bool(reference_ids),
                "present_target": bool(target_ids),
                "display_label": f"{type_label} | {ref_display}  →  {target_display}",
                "impact_text": effect,
                "importance": max(
                    float(ref_item.get("frequency", 0.0) or 0.0),
                    float(target_item.get("frequency", 0.0) or 0.0),
                    abs(float(target_item.get("frequency", 0.0) or 0.0) - float(ref_item.get("frequency", 0.0) or 0.0)),
                ),
            }

        events: list[dict] = []
        shared_keys = set(reference) & set(target)
        for key in shared_keys:
            ref_item = reference[key]
            target_item = target[key]
            identity_changed = ref_item["motif"] != target_item["motif"]
            frequency_delta = float(target_item["frequency"] - ref_item["frequency"])
            if identity_changed:
                events.append(make_event(ref_item, target_item, "identity_substitution"))
            elif abs(frequency_delta) >= 0.10:
                events.append(make_event(ref_item, target_item, "population_shift"))

        removed = [reference[key] for key in set(reference) - set(target)]
        added = [target[key] for key in set(target) - set(reference)]
        candidates = []
        for ref_item in removed:
            for target_item in added:
                if len(ref_item["ids"]) != len(target_item["ids"]):
                    continue
                overlap = len(set(ref_item["ids"]) & set(target_item["ids"])) / max(
                    1, len(set(ref_item["ids"]) | set(target_item["ids"]))
                )
                fraction_distance = abs(float(ref_item["path_fraction"] - target_item["path_fraction"]))
                if ref_item["centroid"] is not None and target_item["centroid"] is not None:
                    spatial_distance = float(np.linalg.norm(ref_item["centroid"] - target_item["centroid"]))
                else:
                    spatial_distance = 999.0
                eligible = (
                    overlap > 0.0
                    or fraction_distance <= (0.16 if len(ref_item["ids"]) == 1 else 0.22)
                    or spatial_distance <= (6.0 if len(ref_item["ids"]) == 1 else 8.0)
                )
                if not eligible:
                    continue
                score = (
                    overlap * 5.0
                    + max(0.0, 1.0 - fraction_distance) * 2.0
                    + 1.0 / (1.0 + spatial_distance / 4.0)
                    + max(0.0, 1.0 - abs(float(ref_item["frequency"] - target_item["frequency"])))
                )
                candidates.append((score, ref_item, target_item))
        candidates.sort(key=lambda item: item[0], reverse=True)
        used_reference: set[tuple[int, ...]] = set()
        used_target: set[tuple[int, ...]] = set()
        for _score, ref_item, target_item in candidates:
            if ref_item["ids"] in used_reference or target_item["ids"] in used_target:
                continue
            used_reference.add(ref_item["ids"])
            used_target.add(target_item["ids"])
            events.append(make_event(ref_item, target_item, "rewired"))
        for ref_item in removed:
            if ref_item["ids"] not in used_reference:
                events.append(make_event(ref_item, None, "contact_loss"))
        for target_item in added:
            if target_item["ids"] not in used_target:
                events.append(make_event(None, target_item, "contact_gain"))

        priority = {
            "identity_substitution": 0,
            "rewired": 1,
            "contact_loss": 2,
            "contact_gain": 3,
            "population_shift": 4,
        }
        events.sort(key=lambda row: (
            priority.get(str(row.get("event_type") or ""), 9),
            -float(row.get("importance", 0.0) or 0.0),
            float(row.get("path_fraction", 0.5) or 0.5),
        ))
        singles = [row for row in events if row.get("selection_type") == "single"][:24]
        pairs = [row for row in events if row.get("selection_type") == "pair"][:24]
        return [*singles, *pairs]

    def _dataset_compare_motif_transitions(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        target_cluster: int,
        regions: list[dict],
    ) -> list[dict]:
        """Build explicit reference-motif -> target-motif transitions.

        Rows keyed by residue IDs are insufficient when a motif disappears and
        another motif appears nearby.  This routine therefore matches absent /
        added motifs within each remodeling interval using residue overlap,
        centroid distance and path-frequency change.
        """
        if not regions:
            return []

        def collect(key: str, cluster_id: int, start: float, end: float) -> tuple[dict[tuple[int, ...], dict], int]:
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(key))
            if binding is None:
                return {}, 0
            try:
                path_rows = binding.db.conn.execute(
                    "SELECT id FROM paths WHERE cluster_id = ?", (int(cluster_id),)
                ).fetchall()
                path_ids = [int(row[0]) for row in path_rows]
                if not path_ids:
                    return {}, 0
                marks: dict[tuple[int, ...], set[int]] = defaultdict(set)
                point_rows = binding.db.conn.execute(
                    "SELECT path_id, res_1, res_2, res_3, res_4 FROM path_points WHERE path_id IN ("
                    + ",".join("?" * len(path_ids)) + ") ORDER BY path_id, seq", path_ids,
                ).fetchall()
                grouped: dict[int, list] = defaultdict(list)
                for row in point_rows:
                    grouped[int(row[0])].append(row)
                for path_id, path_points in grouped.items():
                    n_points = len(path_points)
                    for index, row in enumerate(path_points):
                        fraction = index / max(1, n_points - 1)
                        if fraction < start or fraction > end:
                            continue
                        residue_ids = sorted({int(row[col] or 0) for col in range(1, 5) if int(row[col] or 0) > 0})
                        for size in range(2, min(4, len(residue_ids)) + 1):
                            for combo in combinations(residue_ids, size):
                                marks[tuple(combo)].add(path_id)
                return ({combo: {"path_count": len(paths)} for combo, paths in marks.items()}, len(path_ids))
            except Exception:
                return {}, 0

        def position_map(key: str) -> dict[int, np.ndarray]:
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(key))
            if binding is None:
                return {}
            result = {}
            try:
                for row in binding.db.get_residue_positions():
                    local_id = int(row.get("local_residue_id", row.get("residue_id", 0)) or 0)
                    result[local_id] = np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64)
            except Exception:
                return {}
            return result

        ref_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        ref_names = getattr(getattr(ref_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}
        ref_positions = position_map(reference_key)
        target_positions = position_map(target_key)
        transitions: list[dict] = []

        for region_index, region in enumerate(regions):
            start = float(region.get("start_fraction", 0.0))
            end = float(region.get("end_fraction", 1.0))
            reference, reference_total = collect(reference_key, ref_cluster, start, end)
            target, target_total = collect(target_key, target_cluster, start, end)
            if not reference_total and not target_total:
                continue

            def describe(combo: tuple[int, ...], counts: dict, total: int, names: dict, positions: dict) -> dict:
                frequency = float(counts.get(combo, {}).get("path_count", 0)) / float(total or 1)
                centroid_values = [positions[rid] for rid in combo if rid in positions]
                centroid = np.mean(centroid_values, axis=0) if centroid_values else None
                motif = "-".join(str(names.get(rid, "UNK")).upper() for rid in combo)
                return {"ids": tuple(combo), "frequency": frequency, "centroid": centroid, "motif": motif}

            all_combos = set(reference) | set(target)
            direct: list[dict] = []
            removed: list[dict] = []
            added: list[dict] = []
            for combo in all_combos:
                ref_item = describe(combo, reference, reference_total, ref_names, ref_positions)
                target_item = describe(combo, target, target_total, target_names, target_positions)
                ref_exists = combo in reference
                target_exists = combo in target
                if ref_exists and target_exists:
                    identity_change = ref_item["motif"] != target_item["motif"]
                    if identity_change or abs(target_item["frequency"] - ref_item["frequency"]) >= 0.05:
                        direct.append((ref_item, target_item, "replacement" if identity_change else "population_shift"))
                elif ref_exists:
                    removed.append(ref_item)
                else:
                    added.append(target_item)

            for ref_item, target_item, transition_type in direct:
                if ref_item["centroid"] is None or target_item["centroid"] is None:
                    continue
                transitions.append({
                    "region_index": region_index,
                    "start_fraction": start, "end_fraction": end,
                    "reference_residue_ids": ref_item["ids"], "target_residue_ids": target_item["ids"],
                    "reference_centroid": ref_item["centroid"], "target_centroid": target_item["centroid"],
                    "reference_motif": ref_item["motif"], "target_motif": target_item["motif"],
                    "reference_frequency": ref_item["frequency"], "target_frequency": target_item["frequency"],
                    "frequency_delta": target_item["frequency"] - ref_item["frequency"],
                    "transition_type": transition_type,
                    "display_label": f"{';'.join(map(str, ref_item['ids']))} {ref_item['motif']} -> { ';'.join(map(str, target_item['ids']))} {target_item['motif']}",
                })

            # Greedy one-to-one pairing for removed and added motifs.  A high
            # overlap score preserves the interpretation of a local rewiring.
            candidates = []
            for ref_item in removed:
                for target_item in added:
                    if len(ref_item["ids"]) != len(target_item["ids"]):
                        continue
                    overlap = len(set(ref_item["ids"]) & set(target_item["ids"])) / max(1, len(set(ref_item["ids"]) | set(target_item["ids"])))
                    if ref_item["centroid"] is not None and target_item["centroid"] is not None:
                        distance = float(np.linalg.norm(ref_item["centroid"] - target_item["centroid"]))
                    else:
                        distance = 999.0
                    score = overlap * 3.0 + 1.0 / (1.0 + distance) + abs(ref_item["frequency"] - target_item["frequency"])
                    candidates.append((score, ref_item, target_item))
            candidates.sort(key=lambda item: item[0], reverse=True)
            used_ref: set[tuple[int, ...]] = set()
            used_target: set[tuple[int, ...]] = set()
            for _score, ref_item, target_item in candidates:
                if ref_item["ids"] in used_ref or target_item["ids"] in used_target:
                    continue
                if ref_item["centroid"] is None or target_item["centroid"] is None:
                    continue
                used_ref.add(ref_item["ids"])
                used_target.add(target_item["ids"])
                transitions.append({
                    "region_index": region_index,
                    "start_fraction": start, "end_fraction": end,
                    "reference_residue_ids": ref_item["ids"], "target_residue_ids": target_item["ids"],
                    "reference_centroid": ref_item["centroid"], "target_centroid": target_item["centroid"],
                    "reference_motif": ref_item["motif"], "target_motif": target_item["motif"],
                    "reference_frequency": ref_item["frequency"], "target_frequency": target_item["frequency"],
                    "frequency_delta": target_item["frequency"] - ref_item["frequency"],
                    "transition_type": "rewired",
                    "display_label": f"{';'.join(map(str, ref_item['ids']))} {ref_item['motif']} -> { ';'.join(map(str, target_item['ids']))} {target_item['motif']}",
                })

        transitions.sort(key=lambda row: (row.get("transition_type") not in {"replacement", "rewired"}, -abs(float(row.get("frequency_delta", 0.0)))))
        return transitions[:30]

    def _dataset_compare_residue_change_profile(
        self,
        reference_key: str,
        ref_cluster: int,
        target_key: str,
        target_cluster: int | None,
        *,
        bin_count: int = 64,
        reference_path_ids=None,
        target_path_ids=None,
        use_arc_length: bool = True,
    ) -> dict:
        """Measure residue-composition replacement across the entire path."""
        bin_count = max(16, int(bin_count))
        if use_arc_length:
            reference_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
            target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
            reference_connection = getattr(getattr(reference_binding, "db", None), "conn", None)
            target_connection = getattr(getattr(target_binding, "db", None), "conn", None)
            if reference_connection is None or target_connection is None:
                return {"bins": [], "hotspots": [], "position_rows": [], "threshold": 1.0}
            profile = build_residue_change_profile(
                reference_connection,
                target_connection,
                int(ref_cluster),
                int(target_cluster) if target_cluster is not None else None,
                reference_path_ids=reference_path_ids,
                target_path_ids=target_path_ids,
                bin_count=bin_count,
            )
            enriched_hotspots = []
            for hotspot in profile.get("hotspots", []) or []:
                enriched = dict(hotspot)
                region_id = str(enriched.get("region_id") or enriched.get("hotspot_id") or "R")
                start_percent = float(enriched.get("start_fraction", 0.0)) * 100.0
                end_percent = float(enriched.get("end_fraction", 0.0)) * 100.0
                lost_ids = tuple(int(rid) for rid in enriched.get("lost_residue_ids", ()) or ())
                gained_ids = tuple(int(rid) for rid in enriched.get("gained_residue_ids", ()) or ())
                lost_sequence_ids = tuple(rid + 1 for rid in lost_ids)
                gained_sequence_ids = tuple(rid + 1 for rid in gained_ids)
                lost_text = "/".join(f"S{rid}" for rid in lost_sequence_ids) or "none"
                gained_text = "/".join(f"S{rid}" for rid in gained_sequence_ids) or "none"
                enriched.update({
                    "lost_sequence_ids": lost_sequence_ids,
                    "gained_sequence_ids": gained_sequence_ids,
                    "region_label": f"{region_id} | arc {start_percent:.0f}-{end_percent:.0f}%",
                    "display_label": f"{region_id}: {lost_text} -> {gained_text}",
                    "scene_label": (
                        f"{region_id} {start_percent:.0f}-{end_percent:.0f}%\n"
                        f"{lost_text} -> {gained_text}"
                    ),
                })
                enriched_hotspots.append(enriched)
            profile["hotspots"] = enriched_hotspots
            for key, short_label in (
                ("reference_bottleneck", "BN-Ref"),
                ("target_bottleneck", "BN-Target"),
            ):
                bottleneck = dict(profile.get(key, {}) or {})
                if not bottleneck:
                    continue
                fraction = max(0.0, min(1.0, float(bottleneck.get("fraction", 0.0))))
                radius = float(bottleneck.get("radius", 0.0) or 0.0)
                bottleneck.update({
                    "marker_id": short_label,
                    "scene_label": f"{short_label}\n{fraction * 100:.0f}% | r={radius:.2f} Å",
                })
                profile[key] = bottleneck
            return profile

        def collect(dataset_key: str, cluster_id: int):
            binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
            if binding is None:
                return (
                    [defaultdict(int) for _ in range(bin_count)],
                    [defaultdict(int) for _ in range(bin_count)],
                    0,
                )
            try:
                path_total = int(
                    binding.db.conn.execute(
                        "SELECT COUNT(*) FROM paths WHERE cluster_id = ?",
                        (int(cluster_id),),
                    ).fetchone()[0]
                    or 0
                )
                max_seq = {
                    int(row[0]): max(0, int(row[1] or 0))
                    for row in binding.db.conn.execute(
                        "SELECT pp.path_id, MAX(pp.seq) "
                        "FROM path_points pp JOIN paths p ON p.id = pp.path_id "
                        "WHERE p.cluster_id = ? GROUP BY pp.path_id",
                        (int(cluster_id),),
                    )
                }
                residue_counts = [defaultdict(int) for _ in range(bin_count)]
                motif_counts = [defaultdict(int) for _ in range(bin_count)]
                current_path = None
                residue_marks: set[tuple[int, int]] = set()
                motif_marks: set[tuple[int, tuple[int, ...]]] = set()

                def flush_marks():
                    for bin_index, residue_id in residue_marks:
                        residue_counts[bin_index][residue_id] += 1
                    for bin_index, motif in motif_marks:
                        motif_counts[bin_index][motif] += 1

                rows = binding.db.conn.execute(
                    "SELECT pp.path_id, pp.seq, pp.res_1, pp.res_2, pp.res_3, pp.res_4 "
                    "FROM path_points pp JOIN paths p ON p.id = pp.path_id "
                    "WHERE p.cluster_id = ? ORDER BY pp.path_id, pp.seq",
                    (int(cluster_id),),
                )
                for row in rows:
                    path_id = int(row[0])
                    if current_path is not None and path_id != current_path:
                        flush_marks()
                        residue_marks.clear()
                        motif_marks.clear()
                    current_path = path_id
                    denominator = max(1, int(max_seq.get(path_id, 0)))
                    fraction = max(0.0, min(1.0, float(row[1] or 0) / denominator))
                    bin_index = min(bin_count - 1, int(fraction * bin_count))
                    residue_ids = sorted({
                        int(row[column] or 0)
                        for column in range(2, 6)
                        if int(row[column] or 0) > 0
                    })
                    for residue_id in residue_ids:
                        residue_marks.add((bin_index, residue_id))
                    for size in range(2, min(4, len(residue_ids)) + 1):
                        for motif in combinations(residue_ids, size):
                            motif_marks.add((bin_index, tuple(motif)))
                if current_path is not None:
                    flush_marks()
                return residue_counts, motif_counts, path_total
            except Exception:
                return (
                    [defaultdict(int) for _ in range(bin_count)],
                    [defaultdict(int) for _ in range(bin_count)],
                    0,
                )

        reference_counts, reference_motif_counts, reference_total = collect(reference_key, ref_cluster)
        target_counts, target_motif_counts, target_total = collect(target_key, target_cluster)
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
            reference_motif_frequency = {
                tuple(motif): float(count) / max(1, reference_total)
                for motif, count in reference_motif_counts[bin_index].items()
            }
            target_motif_frequency = {
                tuple(motif): float(count) / max(1, target_total)
                for motif, count in target_motif_counts[bin_index].items()
            }
            motif_ids = set(reference_motif_frequency) | set(target_motif_frequency)
            shared_mass = sum(
                min(reference_motif_frequency.get(motif, 0.0), target_motif_frequency.get(motif, 0.0))
                for motif in motif_ids
            )
            union_mass = sum(
                max(reference_motif_frequency.get(motif, 0.0), target_motif_frequency.get(motif, 0.0))
                for motif in motif_ids
            )
            if union_mass <= 1e-12:
                residue_ids = set(reference_frequency) | set(target_frequency)
                shared_mass = sum(
                    min(reference_frequency.get(residue_id, 0.0), target_frequency.get(residue_id, 0.0))
                    for residue_id in residue_ids
                )
                union_mass = sum(
                    max(reference_frequency.get(residue_id, 0.0), target_frequency.get(residue_id, 0.0))
                    for residue_id in residue_ids
                )
            score = 1.0 - shared_mass / union_mass if union_mass > 1e-12 else 0.0
            raw_scores.append(float(score))
            rows.append({
                "bin_index": bin_index,
                "start_fraction": bin_index / bin_count,
                "end_fraction": (bin_index + 1) / bin_count,
                "score_raw": float(score),
                "reference_frequency": reference_frequency,
                "target_frequency": target_frequency,
                "reference_motif_frequency": reference_motif_frequency,
                "target_motif_frequency": target_motif_frequency,
            })

        values = np.asarray(raw_scores, dtype=np.float64)
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= kernel.sum()
        padded = np.pad(values, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        for row, score in zip(rows, smoothed):
            row["score"] = float(score)

        # Select several non-overlapping high-score windows rather than one
        # percentile-contiguous plateau. This keeps the entire path explorable
        # when motif replacement is broadly elevated by a mutation.
        window_size = max(4, int(round(bin_count * 0.12)))
        stride = max(1, window_size // 3)
        window_candidates = []
        for start_index in range(0, max(1, bin_count - window_size + 1), stride):
            end_index = min(bin_count - 1, start_index + window_size - 1)
            window_values = smoothed[start_index:end_index + 1]
            window_candidates.append((
                float(np.mean(window_values)),
                float(np.max(window_values)),
                start_index,
                end_index,
            ))
        final_start = max(0, bin_count - window_size)
        if not any(item[2] == final_start for item in window_candidates):
            window_values = smoothed[final_start:bin_count]
            window_candidates.append((
                float(np.mean(window_values)),
                float(np.max(window_values)),
                final_start,
                bin_count - 1,
            ))
        window_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        regions: list[tuple[int, int]] = []
        selected_scores: list[float] = []
        for mean_score, _max_score, start_index, end_index in window_candidates:
            if mean_score < 0.25:
                continue
            overlaps = any(
                not (end_index < existing_start or start_index > existing_end)
                for existing_start, existing_end in regions
            )
            if overlaps:
                continue
            regions.append((start_index, end_index))
            selected_scores.append(mean_score)
            if len(regions) >= 6:
                break
        if not regions and float(np.max(smoothed, initial=0.0)) >= 0.20:
            peak_index = int(np.argmax(smoothed))
            start_index = max(0, peak_index - window_size // 2)
            end_index = min(bin_count - 1, start_index + window_size - 1)
            regions.append((start_index, end_index))
            selected_scores.append(float(np.mean(smoothed[start_index:end_index + 1])))
        threshold = min(selected_scores) if selected_scores else 1.0

        candidates: list[dict] = []
        for start_index, end_index in regions:
            region_rows = rows[start_index:end_index + 1]
            reference_average: dict[int, float] = defaultdict(float)
            target_average: dict[int, float] = defaultdict(float)
            reference_motif_average: dict[tuple[int, ...], float] = defaultdict(float)
            target_motif_average: dict[tuple[int, ...], float] = defaultdict(float)
            for row in region_rows:
                for residue_id, frequency in row["reference_frequency"].items():
                    reference_average[int(residue_id)] += float(frequency) / len(region_rows)
                for residue_id, frequency in row["target_frequency"].items():
                    target_average[int(residue_id)] += float(frequency) / len(region_rows)
                for motif, frequency in row["reference_motif_frequency"].items():
                    reference_motif_average[tuple(motif)] += float(frequency) / len(region_rows)
                for motif, frequency in row["target_motif_frequency"].items():
                    target_motif_average[tuple(motif)] += float(frequency) / len(region_rows)
            residue_ids = set(reference_average) | set(target_average)
            deltas = {
                residue_id: target_average.get(residue_id, 0.0) - reference_average.get(residue_id, 0.0)
                for residue_id in residue_ids
            }
            lost_ranked = sorted(
                (item for item in deltas.items() if item[1] < -0.02),
                key=lambda item: item[1],
            )
            gained_ranked = sorted(
                (item for item in deltas.items() if item[1] > 0.02),
                key=lambda item: item[1],
                reverse=True,
            )
            motif_ids = set(reference_motif_average) | set(target_motif_average)
            motif_deltas = {
                motif: target_motif_average.get(motif, 0.0) - reference_motif_average.get(motif, 0.0)
                for motif in motif_ids
            }
            lost_motif_ranked = sorted(
                (item for item in motif_deltas.items() if item[1] < -0.02),
                key=lambda item: item[1],
            )
            gained_motif_ranked = sorted(
                (item for item in motif_deltas.items() if item[1] > 0.02),
                key=lambda item: item[1],
                reverse=True,
            )
            lost_motifs = tuple(tuple(item[0]) for item in lost_motif_ranked[:3])
            gained_motifs = tuple(tuple(item[0]) for item in gained_motif_ranked[:3])

            def changed_residue_ids(motifs, fallback_ranked):
                values: list[int] = []
                for motif in motifs:
                    for residue_id in motif:
                        if int(residue_id) not in values:
                            values.append(int(residue_id))
                for residue_id, _delta in fallback_ranked:
                    if int(residue_id) not in values:
                        values.append(int(residue_id))
                return tuple(values[:4])

            lost_ids = changed_residue_ids(lost_motifs, lost_ranked)
            gained_ids = changed_residue_ids(gained_motifs, gained_ranked)

            reference_ranked = [
                int(item[0])
                for item in sorted(reference_average.items(), key=lambda item: item[1], reverse=True)
            ]
            target_ranked = [
                int(item[0])
                for item in sorted(target_average.items(), key=lambda item: item[1], reverse=True)
            ]
            reference_ids = tuple(dict.fromkeys((*lost_ids, *reference_ranked)))[:4]
            target_ids = tuple(dict.fromkeys((*gained_ids, *target_ranked)))[:4]
            region_scores = smoothed[start_index:end_index + 1]
            local_peak = int(np.argmax(region_scores))
            peak_index = start_index + local_peak
            candidates.append({
                "start_fraction": start_index / bin_count,
                "end_fraction": (end_index + 1) / bin_count,
                "peak_fraction": (peak_index + 0.5) / bin_count,
                "mean_score": float(np.mean(region_scores)),
                "max_score": float(np.max(region_scores)),
                "reference_residue_ids": reference_ids,
                "target_residue_ids": target_ids,
                "lost_residue_ids": lost_ids,
                "gained_residue_ids": gained_ids,
                "lost_motifs": lost_motifs,
                "gained_motifs": gained_motifs,
                "reference_motif": "residue composition",
                "target_motif": "residue composition",
                "transition_type": "path_residue_hotspot",
            })

        candidates.sort(key=lambda row: -float(row.get("max_score", 0.0)))
        hotspots = []
        for rank, row in enumerate(candidates[:6], start=1):
            hotspot = dict(row)
            hotspot["hotspot_id"] = f"H{rank}"
            lost_text = " / ".join(
                "-".join(map(str, motif)) for motif in hotspot.get("lost_motifs", ()) or ()
            ) or ",".join(map(str, hotspot.get("lost_residue_ids", ()))) or "none"
            gained_text = " / ".join(
                "-".join(map(str, motif)) for motif in hotspot.get("gained_motifs", ()) or ()
            ) or ",".join(map(str, hotspot.get("gained_residue_ids", ()))) or "none"
            hotspot["display_label"] = f"H{rank}: -{lost_text} / +{gained_text}"
            hotspots.append(hotspot)

        visible_bins = [
            {
                "bin_index": int(row["bin_index"]),
                "start_fraction": float(row["start_fraction"]),
                "end_fraction": float(row["end_fraction"]),
                "score": float(row.get("score", 0.0)),
            }
            for row in rows
        ]
        return {
            "bins": visible_bins,
            "hotspots": hotspots,
            # Internal full-composition rows drive the Residue Observer's
            # same-normalized-length comparison. The compact track only needs
            # ``visible_bins`` above.
            "position_rows": rows,
            "threshold": float(threshold),
            "reference_path_count": int(reference_total),
            "target_path_count": int(target_total),
        }

    def _dataset_compare_path_position_events(
        self,
        reference_key: str,
        target_key: str,
        residue_change_profile: dict,
    ) -> list[dict]:
        """Describe residue-environment changes at matched arc-length positions."""
        profile = dict(residue_change_profile or {})
        position_rows = [dict(row) for row in profile.get("position_rows", []) or []]
        hotspots = [dict(row) for row in profile.get("hotspots", []) or []]
        if not position_rows or not hotspots:
            return []
        ref_binding = getattr(self.db, "_datasets_by_key", {}).get(str(reference_key))
        target_binding = getattr(self.db, "_datasets_by_key", {}).get(str(target_key))
        ref_names = getattr(getattr(ref_binding, "db", None), "_residue_name_map", {}) or {}
        target_names = getattr(getattr(target_binding, "db", None), "_residue_name_map", {}) or {}

        def dominant_ids(frequencies: dict, *, limit: int = 4) -> tuple[int, ...]:
            ranked = sorted(
                (
                    (int(residue_id), float(value))
                    for residue_id, value in dict(frequencies or {}).items()
                    if int(residue_id) > 0 and float(value) >= 0.02
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            return tuple(residue_id for residue_id, _value in ranked[:limit])

        def residue_text(ids, names) -> str:
            if not ids:
                return "none"
            return " + ".join(
                f"S{int(rid) + 1} {str(names.get(int(rid), 'UNK')).upper()} (MD{int(rid)})"
                for rid in ids
            )

        events = []
        for hotspot in hotspots:
            peak_fraction = max(0.0, min(1.0, float(hotspot.get("peak_fraction", 0.5))))
            peak_row = min(
                position_rows,
                key=lambda row: abs(
                    (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0))) * 0.5
                    - peak_fraction
                ),
            )
            reference_ids = dominant_ids(peak_row.get("reference_frequency", {}))
            target_ids = dominant_ids(peak_row.get("target_frequency", {}))
            if not reference_ids and not target_ids:
                continue
            reference_display = residue_text(reference_ids, ref_names)
            target_display = residue_text(target_ids, target_names)
            reference_set = set(reference_ids)
            target_set = set(target_ids)
            lost_ids = tuple(rid for rid in reference_ids if rid not in target_set)
            gained_ids = tuple(rid for rid in target_ids if rid not in reference_set)
            retained_ids = tuple(rid for rid in reference_ids if rid in target_set)
            reference_frequency = {
                int(rid): float(value)
                for rid, value in dict(peak_row.get("reference_frequency", {}) or {}).items()
            }
            target_frequency = {
                int(rid): float(value)
                for rid, value in dict(peak_row.get("target_frequency", {}) or {}).items()
            }
            all_ids = set(reference_frequency) | set(target_frequency)
            changed_ids = tuple(sorted(
                rid for rid in all_ids
                if abs(target_frequency.get(rid, 0.0) - reference_frequency.get(rid, 0.0)) >= 0.08
            ))
            decreased_ids = tuple(sorted(
                rid for rid in all_ids
                if target_frequency.get(rid, 0.0) - reference_frequency.get(rid, 0.0) <= -0.08
            ))
            increased_ids = tuple(sorted(
                rid for rid in all_ids
                if target_frequency.get(rid, 0.0) - reference_frequency.get(rid, 0.0) >= 0.08
            ))
            lost_text = residue_text(lost_ids, ref_names)
            gained_text = residue_text(gained_ids, target_names)
            retained_text = residue_text(retained_ids, ref_names)
            position_percent = peak_fraction * 100.0
            bin_start = float(peak_row.get("start_fraction", peak_fraction))
            bin_end = float(peak_row.get("end_fraction", peak_fraction))
            hotspot_id = str(hotspot.get("hotspot_id") or f"P{len(events) + 1}")
            effect = (
                f"At the same physical arc-length position ({position_percent:.1f}%), the baseline residue "
                f"environment is [{reference_display}], while the mutant environment is [{target_display}]. "
                f"Retained: {retained_text}; lost: {lost_text}; gained: {gained_text}."
            )
            if decreased_ids:
                effect += " Decreased contacts: " + ", ".join(
                    f"S{rid + 1} {str(ref_names.get(rid, 'UNK')).upper()} (MD{rid}) "
                    f"({reference_frequency.get(rid, 0.0) * 100:.0f}%→{target_frequency.get(rid, 0.0) * 100:.0f}%)"
                    for rid in decreased_ids
                ) + "."
            if increased_ids:
                effect += " Increased contacts: " + ", ".join(
                    f"S{rid + 1} {str(target_names.get(rid, 'UNK')).upper()} (MD{rid}) "
                    f"({reference_frequency.get(rid, 0.0) * 100:.0f}%→{target_frequency.get(rid, 0.0) * 100:.0f}%)"
                    for rid in increased_ids
                ) + "."
            events.append({
                "event_id": f"path_position:{hotspot_id}:{peak_fraction:.5f}",
                "selection_type": "path_position",
                "event_type": "path_position_change",
                "transition_type": "path_position_change",
                "hotspot_id": hotspot_id,
                "reference_dataset_key": str(reference_key),
                "target_dataset_key": str(target_key),
                "reference_residue_ids": reference_ids,
                "target_residue_ids": target_ids,
                "reference_sequence_ids": tuple(rid + 1 for rid in reference_ids),
                "target_sequence_ids": tuple(rid + 1 for rid in target_ids),
                "reference_motif": " + ".join(str(ref_names.get(rid, "UNK")).upper() for rid in reference_ids),
                "target_motif": " + ".join(str(target_names.get(rid, "UNK")).upper() for rid in target_ids),
                "lost_residue_ids": lost_ids,
                "gained_residue_ids": gained_ids,
                "retained_residue_ids": retained_ids,
                "changed_residue_ids": changed_ids,
                "decreased_residue_ids": decreased_ids,
                "increased_residue_ids": increased_ids,
                "start_fraction": bin_start,
                "end_fraction": bin_end,
                "peak_fraction": peak_fraction,
                "path_fraction": peak_fraction,
                "hotspot_start_fraction": float(hotspot.get("start_fraction", bin_start)),
                "hotspot_end_fraction": float(hotspot.get("end_fraction", bin_end)),
                "present_reference": bool(reference_ids),
                "present_target": bool(target_ids),
                "display_label": (
                    f"{hotspot_id} | Path {position_percent:.1f}% | "
                    f"{reference_display}  →  {target_display}"
                ),
                "impact_text": effect,
                "importance": float(hotspot.get("max_score", 0.0) or 0.0),
            })
        events.sort(key=lambda row: -float(row.get("importance", 0.0) or 0.0))
        return events

    def _render_dataset_compare_overlay(self):
        payload = getattr(self, "_dataset_compare_active", None)
        if not payload:
            return
        # Exact compositions are inspected in the paired Residue Observer
        # views; the main canvas only marks full-path replacement hotspots.
        self._viewer.clear_dataset_compare_motifs(render=False)
        self._viewer.clear_dataset_compare_residue_hotspots(render=False)
        self._viewer.clear_dataset_compare_bottlenecks(render=False)
        self._viewer.clear_residues()
        self._viewer.set_protein_highlight_residues([], refresh=True, render=False)
        centers = payload.get("centers", {})
        path_coords = {}
        color_map = {}
        if centers.get("reference") is not None:
            path_coords[-1001] = np.asarray(centers["reference"], dtype=np.float64)
            color_map[-1001] = "#2563EB"
        target_family = dict(centers.get("target_family", {}) or {})
        if target_family:
            for family_index, (_cluster_id, points) in enumerate(target_family.items()):
                path_id = -1002 - family_index
                path_coords[path_id] = np.asarray(points, dtype=np.float64)
                color_map[path_id] = "#F97316" if family_index == 0 else "#FB923C"
        elif centers.get("target") is not None:
            path_coords[-1002] = np.asarray(centers["target"], dtype=np.float64)
            color_map[-1002] = "#F97316"
        reference_array = np.asarray(centers.get("reference"), dtype=np.float64) if centers.get("reference") is not None else None
        target_array = np.asarray(centers.get("target"), dtype=np.float64) if centers.get("target") is not None else None
        residue_hotspots = list((payload.get("residue_change_profile") or {}).get("hotspots", []) or [])
        legend_hotspots = []
        highlighted_hotspot_ids: set[int] = set()
        target_hotspot_colors = ("#DC2626", "#E11D48", "#BE123C", "#9F1239", "#F59E0B", "#D97706")
        reference_hotspot_colors = ("#991B1B", "#9F1239", "#881337", "#701A75", "#B45309", "#92400E")
        for region_index, hotspot in enumerate(residue_hotspots):
            target_hotspot_color = target_hotspot_colors[region_index % len(target_hotspot_colors)]
            reference_hotspot_color = reference_hotspot_colors[region_index % len(reference_hotspot_colors)]
            start_fraction = max(0.0, min(1.0, float(hotspot.get("start_fraction", 0.0))))
            end_fraction = max(start_fraction, min(1.0, float(hotspot.get("end_fraction", 1.0))))
            peak_fraction = max(start_fraction, min(end_fraction, float(hotspot.get("peak_fraction", start_fraction))))
            label_point = None
            if target_array is not None and len(target_array) >= 2:
                start = max(0, min(len(target_array) - 2, int(round(start_fraction * (len(target_array) - 1)))))
                end = max(start + 1, min(len(target_array) - 1, int(round(end_fraction * (len(target_array) - 1)))))
                target_segment_id = -1300 - region_index
                path_coords[target_segment_id] = target_array[start:end + 1]
                color_map[target_segment_id] = target_hotspot_color
                if region_index < 4:
                    highlighted_hotspot_ids.add(target_segment_id)
                peak_index = max(0, min(len(target_array) - 1, int(round(peak_fraction * (len(target_array) - 1)))))
                label_point = target_array[peak_index]
            if reference_array is not None and len(reference_array) >= 2:
                start = max(0, min(len(reference_array) - 2, int(round(start_fraction * (len(reference_array) - 1)))))
                end = max(start + 1, min(len(reference_array) - 1, int(round(end_fraction * (len(reference_array) - 1)))))
                reference_segment_id = -1400 - region_index
                path_coords[reference_segment_id] = reference_array[start:end + 1]
                color_map[reference_segment_id] = reference_hotspot_color
                if region_index < 4:
                    highlighted_hotspot_ids.add(reference_segment_id)
                if label_point is None:
                    peak_index = max(0, min(len(reference_array) - 1, int(round(peak_fraction * (len(reference_array) - 1)))))
                    label_point = reference_array[peak_index]
            # Keep the main 3D view readable: every hotspot remains encoded as
            # a coloured path segment, while only the strongest region gets a
            # residue-composition tag.  The full R1/R2/... set stays available
            # in the full-path replacement map and Residue Observer.
            if label_point is not None and region_index == 0:
                legend_hotspots.append({**hotspot, "tag_color": target_hotspot_color, "point": label_point})
        if path_coords:
            self._viewer.render_matched_paths(
                path_coords,
                color_map=color_map,
                opacity=0.95,
                line_width=4.0,
                highlighted=highlighted_hotspot_ids,
            )
        self._viewer.render_dataset_compare_residue_hotspots(legend_hotspots)
        bottleneck_markers = []
        profile = dict(payload.get("residue_change_profile") or {})
        for key, path_array, color in (
            ("reference_bottleneck", reference_array, "#0284C7"),
            ("target_bottleneck", target_array, "#F59E0B"),
        ):
            marker = dict(profile.get(key, {}) or {})
            if not marker or path_array is None or len(path_array) < 2:
                continue
            sampled = self._resample_compare_points(path_array, 257)
            if sampled is None:
                continue
            fraction = max(0.0, min(1.0, float(marker.get("fraction", 0.0))))
            marker["point"] = sampled[int(round(fraction * (len(sampled) - 1)))]
            marker["color"] = color
            bottleneck_markers.append(marker)
        self._viewer.render_dataset_compare_bottlenecks(bottleneck_markers)

    def _on_dataset_compare_source_changed(self, reference_key: str) -> None:
        reference_key = str(reference_key or "")
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare()
        self._dataset_compare_panel.set_comparison_rows([])
        if not reference_key:
            self._dataset_compare_panel.set_reference_context()
            return
        groups = self.db.get_dataset_cluster_path_groups(reference_key)
        current = self._current_dataset_compare_reference()
        current_cluster = int(current[1]) if current is not None and str(current[0]) == reference_key else None
        path_count = len(groups.get(current_cluster, set())) if current_cluster is not None else None
        self._dataset_compare_panel.set_reference_context(
            dataset_label=self._dataset_compare_label(reference_key),
            cluster_id=current_cluster,
            path_count=path_count,
            cluster_count=len(groups),
        )
        self._sync_compare_targets_to_observer()
        self._statusbar.showMessage(
            f"Dataset Compare source changed to {self._dataset_compare_label(reference_key)}; rebuild mappings to continue",
            3500,
        )

    def _sync_compare_targets_to_observer(self) -> None:
        panel = getattr(self, "_dataset_compare_panel", None)
        if panel is None:
            return
        source_key = panel.source_dataset_key() if hasattr(panel, "source_dataset_key") else ""
        target_keys = panel.target_dataset_keys() if hasattr(panel, "target_dataset_keys") else []
        selected = []
        for key in (source_key, *target_keys):
            key = str(key or "")
            if key and key not in selected:
                selected.append(key)
        self._observer_compare_dataset_keys = selected
        if not self._protein_observer_page_active():
            return
        self._refresh_observer_dataset_multiselect(preferred_keys=selected)
        if not self._load_uninitialized_observer_datasets():
            self._finalize_observer_dataset_loads()

    def _on_dataset_compare_targets_changed(self, dataset_keys) -> None:
        panel = self._dataset_compare_panel
        reference_key = panel.source_dataset_key()
        selected_targets = [
            str(key) for key in (dataset_keys or [])
            if str(key) and str(key) != reference_key
        ]
        active = dict(getattr(self, "_dataset_compare_active", None) or {})
        if active and str(active.get("target_key") or "") not in set(selected_targets):
            self._clear_dataset_compare()
            active = {}
        self._sync_compare_targets_to_observer()
        if not reference_key:
            panel.set_comparison_rows([])
            return
        rows = self._dataset_compare_mapping_rows(reference_key, selected_targets)
        selected = None
        if active and str(active.get("reference_key") or "") == reference_key:
            selected = {
                "reference_cluster_id": int(active.get("reference_cluster", -1)),
                "target_dataset": str(active.get("target_key") or ""),
            }
        panel.set_comparison_rows(rows, selected=selected)
        self._statusbar.showMessage(
            "Dataset Compare target updated; Residue Observer will display the "
            "source and selected target",
            4000,
        )

    def _on_dataset_compare_residue_threshold_changed(self, threshold: float) -> None:
        # Kept for compatibility with older automation; residue similarity is
        # descriptive and no longer filters mapping candidates.
        self._dataset_compare_residue_similarity_threshold = 0.0
        reference_key = self._dataset_compare_panel.source_dataset_key()
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare()
            self._refresh_dataset_compare_cluster_options()
        if not reference_key:
            return
        rows = self._dataset_compare_mapping_rows(reference_key)
        self._dataset_compare_panel.set_comparison_rows(rows)
        self._statusbar.showMessage(
            f"Residue filtering is disabled; mappings use pure RMSD with a "
            f"{self._dataset_compare_rmsd_max_threshold:.2f} Å absolute maximum and "
            f"{self._dataset_compare_rmsd_distance_threshold:.2f} Å candidate-family margin",
            5000,
        )

    def _on_dataset_compare_rmsd_threshold_changed(self, threshold: float) -> None:
        self._dataset_compare_rmsd_distance_threshold = max(0.0, float(threshold))
        reference_key = self._dataset_compare_panel.source_dataset_key()
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare()
            self._refresh_dataset_compare_cluster_options()
        if not reference_key:
            return
        rows = self._dataset_compare_mapping_rows(reference_key)
        self._dataset_compare_panel.set_comparison_rows(rows)
        self._statusbar.showMessage(
            f"RMSD family margin updated to best + "
            f"{self._dataset_compare_rmsd_distance_threshold:.2f} Å (absolute maximum "
            f"{self._dataset_compare_rmsd_max_threshold:.2f} Å); mappings rebuilt",
            5000,
        )

    def _on_dataset_compare_rmsd_max_threshold_changed(self, threshold: float) -> None:
        self._dataset_compare_rmsd_max_threshold = max(0.0, float(threshold))
        reference_key = self._dataset_compare_panel.source_dataset_key()
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare()
            self._refresh_dataset_compare_cluster_options()
        if not reference_key:
            return
        rows = self._dataset_compare_mapping_rows(reference_key)
        self._dataset_compare_panel.set_comparison_rows(rows)
        self._statusbar.showMessage(
            f"Maximum path RMSD updated to {self._dataset_compare_rmsd_max_threshold:.2f} Å; "
            "sources whose best candidate exceeds it are now left unmatched",
            5000,
        )

    def _on_dataset_compare_requested(self):
        panel = self._dataset_compare_panel
        self._dataset_compare_residue_similarity_threshold = 0.0
        reference_key = panel.source_dataset_key() if hasattr(panel, "source_dataset_key") else ""
        if not reference_key:
            self._dataset_compare_panel.set_reference_context()
            self._dataset_compare_panel.set_comparison_rows([])
            self._statusbar.showMessage("Choose a source dataset in Dataset Compare", 4000)
            return
        self._sync_compare_targets_to_observer()
        rows = self._dataset_compare_mapping_rows(reference_key)
        selected = None
        active = dict(getattr(self, "_dataset_compare_active", None) or {})
        if str(active.get("reference_key") or "") == reference_key:
            selected = {
                "reference_cluster_id": int(active.get("reference_cluster", -1)),
                "target_dataset": str(active.get("target_key") or ""),
            }
        else:
            current = self._current_dataset_compare_reference()
            if current is not None and str(current[0]) == reference_key:
                selected = {
                    "reference_cluster_id": int(current[1]),
                    "target_dataset": str(rows[0].get("target_dataset") or "") if rows else "",
                }
        self._dataset_compare_panel.set_comparison_rows(rows, selected=selected)
        if not rows:
            self._dataset_compare_panel.set_result(
                summary="No aligned center-path metadata was found for the other loaded datasets.",
                changes=[],
            )
            self._statusbar.showMessage("No comparable target clusters were found; run the alignment pipeline first", 5000)
            return
        source_cluster_count = len({int(row.get("reference_cluster_id", 0)) for row in rows})
        self._statusbar.showMessage(
            f"Built {len(rows)} source-to-target mapping rows from {source_cluster_count} "
            f"source clusters for {self._dataset_compare_label(reference_key)} "
            f"(pure RMSD, "
            f"maximum RMSD {self._dataset_compare_rmsd_max_threshold:.2f} Å, "
            f"family margin {self._dataset_compare_rmsd_distance_threshold:.2f} Å; no residue/path filters)",
            4000,
        )

    def _on_dataset_compare_selection(self, candidate: dict):
        reference_key = str(candidate.get("reference_key") or "")
        ref_cluster = int(candidate.get("reference_cluster_id", 0) or 0)
        target_key = str(candidate.get("target_dataset") or "")
        target_clusters = tuple(dict.fromkeys(
            int(value) for value in (candidate.get("target_cluster_ids", ()) or ()) if int(value) > 0
        ))
        if not target_clusters and int(candidate.get("target_cluster_id", 0) or 0) > 0:
            target_clusters = (int(candidate["target_cluster_id"]),)
        if not reference_key or not target_key or ref_cluster <= 0 or not target_clusters:
            return
        target_cluster = int(target_clusters[0])
        target_labels = ", ".join(f"C{cluster_id}" for cluster_id in target_clusters)
        self._clear_dataset_compare_observer_selection()
        self._statusbar.showMessage(
            f"Loading all paths from Cluster {ref_cluster} and target family {target_labels}...",
            0,
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path_scope = self._dataset_compare_unfiltered_cluster_scope(
                reference_key,
                ref_cluster,
                target_key,
                target_clusters,
            )
        finally:
            QApplication.restoreOverrideCursor()
        if path_scope.get("error") or not path_scope.get("target_path_ids"):
            error = str(path_scope.get("error") or "The selected cluster pair contains no comparable paths.")
            self._dataset_compare_panel.set_result(summary=error, changes=[])
            self._statusbar.showMessage(error, 5000)
            return

        target_cluster_counts = {
            int(cluster_id): int(count)
            for cluster_id, count in dict(path_scope.get("target_cluster_counts", {}) or {}).items()
            if int(cluster_id) > 0 and int(count) > 0
        }
        target_path_ids_by_cluster = {
            int(cluster_id): [int(path_id) for path_id in path_ids]
            for cluster_id, path_ids in dict(path_scope.get("target_path_ids_by_cluster", {}) or {}).items()
        }
        critical_cluster = int(target_cluster)
        critical_target_path_ids = list(path_scope.get("target_path_ids", []) or [])
        critical_target_global_ids = list(path_scope.get("target_global_path_ids", []) or [])
        target_cluster = critical_cluster
        if hasattr(self._dataset_compare_panel, "set_selected_critical_cluster"):
            self._dataset_compare_panel.set_selected_critical_cluster(
                target_clusters,
                len(critical_target_path_ids),
            )
        dataset_panel = getattr(self, "_dataset_panel", None)
        if dataset_panel is not None and hasattr(dataset_panel, "set_checked_clusters"):
            dataset_panel.set_checked_clusters(
                {(reference_key, ref_cluster)} | {(target_key, cluster_id) for cluster_id in target_clusters},
                current=(reference_key, ref_cluster),
                emit=True,
            )
        exact_path_ids = set(path_scope.get("reference_global_path_ids", []) or ()) | set(
            critical_target_global_ids
        )
        if (
            exact_path_ids
            and self._tunnel_properties_visible()
        ):
            self.state.apply_lasso(exact_path_ids, "replace")
            self._path_panel.set_highlighted_paths(sorted(exact_path_ids))

        reference_data = self._dataset_compare_center_data(reference_key)
        target_data = self._dataset_compare_center_data(target_key)
        ref_entry = (reference_data.get("clusters") or {}).get(str(ref_cluster), {})
        target_entry = (target_data.get("clusters") or {}).get(str(critical_cluster), {})
        ref_points = ref_entry.get("points")
        target_points = target_entry.get("points")
        target_family_points = {
            cluster_id: (target_data.get("clusters") or {}).get(str(cluster_id), {}).get("points")
            for cluster_id in target_clusters
        }
        target_family_points = {
            cluster_id: points for cluster_id, points in target_family_points.items() if points is not None
        }
        if ref_points is None or target_points is None:
            self._dataset_compare_panel.set_result(
                summary="Aligned center-path metadata was not found for the matched cluster population.",
                changes=[],
            )
            self._statusbar.showMessage("Matched cluster center metadata is unavailable; run the alignment pipeline first", 5000)
            return

        # Compare the complete path populations of the explicitly selected
        # source and target clusters. No residue/path constraint changes the
        # user's target choice after the table click.
        residue_change_profile = self._dataset_compare_residue_change_profile(
            reference_key,
            ref_cluster,
            target_key,
            None,
            reference_path_ids=path_scope.get("reference_path_ids", []),
            target_path_ids=critical_target_path_ids,
        )
        residue_change_profile = self._dataset_compare_allosteric_profile(
            reference_key,
            target_key,
            residue_change_profile,
            ref_points,
            target_points,
        )
        scope_summary = (
            f"Selected target family {target_labels}: "
            f"all {len(critical_target_path_ids):,} paths across {len(target_clusters)} cluster(s)"
        )
        residue_change_profile["scope_summary"] = scope_summary
        residue_change_profile["target_cluster_counts"] = target_cluster_counts
        residue_change_profile["keep_ratio"] = 1.0
        residue_change_profile["filter_rule"] = "none"
        change_events = self._dataset_compare_path_position_events(
            reference_key, target_key, residue_change_profile,
        )
        self._dataset_compare_active = {
            "reference_key": reference_key, "target_key": target_key,
            "reference_cluster": ref_cluster,
            "target_cluster": critical_cluster,
            "target_clusters": target_clusters,
            "target_cluster_counts": target_cluster_counts,
            "path_scope": path_scope,
            "centers": {
                "reference": ref_points,
                "target": target_points,
                "target_family": target_family_points,
            },
            "residue_change_profile": residue_change_profile,
            "change_events": change_events,
        }
        available_hotspots = list(residue_change_profile.get("hotspots", []) or [])
        first_hotspot = next(
            (
                dict(hotspot)
                for hotspot in available_hotspots
                if str(
                    hotspot.get("region_id") or hotspot.get("hotspot_id") or ""
                ).strip().upper() == "R1"
            ),
            dict(available_hotspots[0]) if available_hotspots else {},
        )
        self.state.set_analysis_selection({
            "reference_key": reference_key,
            "reference_cluster": ref_cluster,
            "target_key": target_key,
            "target_cluster": critical_cluster,
            "target_clusters": target_clusters,
            "region_id": str(first_hotspot.get("region_id") or first_hotspot.get("hotspot_id") or ""),
            "arc_start": float(first_hotspot.get("start_fraction", 0.0) or 0.0),
            "arc_end": float(first_hotspot.get("end_fraction", 1.0) or 1.0),
            "arc_peak": float(first_hotspot.get("peak_fraction", 0.5) or 0.5),
        })
        self._observer_change_events = list(change_events)
        self._refresh_observer_change_event_selector()
        # The main comparison view shows path-wide residue replacement; exact
        # reference/target compositions are inspected in Residue Observer.
        self._viewer.clear_residues()
        self._viewer.set_protein_highlight_residues([], refresh=True, render=False)
        self._render_dataset_compare_overlay()
        summary = (
            f"Cluster {ref_cluster} → {self._dataset_compare_label(target_key)} "
            f"Clusters {target_labels} | unfiltered family comparison | "
            f"paths {len(path_scope.get('reference_path_ids', [])):,} / "
            f"{len(critical_target_path_ids):,}"
        )
        hotspot_count = len(residue_change_profile.get("hotspots", []) or [])
        summary += f" | arc-length residue regions {hotspot_count} | position events {len(change_events)}"
        self._dataset_compare_panel.set_result(
            summary=summary,
            changes=[],
            residue_change_profile=residue_change_profile,
            reference_dataset_key=reference_key,
            target_dataset_key=target_key,
        )
        self._set_bottom_analysis_mode("evidence")
        # R1 is the initial analysis region. Prepare its temporal evidence and
        # start the framewise distance audit immediately, so opening Evidence
        # never requires a second click merely to begin calculation.
        self._refresh_evidence_panel(first_hotspot, include_temporal=True)
        self._statusbar.showMessage(
            f"Compared all paths in {reference_key} Cluster {ref_cluster} with all paths in "
            f"{target_key} clusters {target_labels} ({len(critical_target_path_ids):,} target paths)",
            5000,
        )

    def _on_dataset_compare_mapping_lock_requested(self, candidate: dict) -> None:
        candidate = dict(candidate or {})
        reference_key = str(candidate.get("reference_key") or "")
        target_key = str(candidate.get("target_dataset") or "")
        reference_cluster = int(candidate.get("reference_cluster_id", 0) or 0)
        target_clusters = tuple(dict.fromkeys(
            int(value) for value in (candidate.get("target_cluster_ids", ()) or ()) if int(value) > 0
        ))
        if not target_clusters and int(candidate.get("target_cluster_id", 0) or 0) > 0:
            target_clusters = (int(candidate["target_cluster_id"]),)
        if not reference_key or not target_key or reference_cluster <= 0 or not target_clusters:
            return
        pair_key = (reference_key, target_key)
        locks = self._dataset_compare_mapping_locks.setdefault(pair_key, {})
        if bool(candidate.get("locked")):
            locks[reference_cluster] = target_clusters
            action = "locked"
        else:
            locks.pop(reference_cluster, None)
            action = "unlocked"
        self._dataset_compare_panel.set_current_mapping_locked(action == "locked")
        target_labels = ", ".join(f"C{cluster_id}" for cluster_id in target_clusters)
        self._statusbar.showMessage(
            f"Mapping {action}: {self._dataset_compare_label(reference_key)} cluster {reference_cluster} → "
            f"{self._dataset_compare_label(target_key)} clusters {target_labels}; each source cluster may map to multiple targets",
            5000,
        )

    def _on_dataset_compare_export_evidence(self) -> None:
        """Export a compact, inspectable record of the active comparison."""
        active = dict(getattr(self, "_dataset_compare_active", None) or {})
        if not active:
            self._statusbar.showMessage("No active dataset comparison to export", 3000)
            return
        reference_key = str(active.get("reference_key") or "")
        target_key = str(active.get("target_key") or "")
        reference_cluster = int(active.get("reference_cluster", 0) or 0)
        target_cluster = int(active.get("target_cluster", 0) or 0)
        suggested = (
            f"hex_evidence_{self._dataset_compare_label(reference_key)}_C{reference_cluster}_"
            f"to_{self._dataset_compare_label(target_key)}_C{target_cluster}.json"
        )
        suggested = re.sub(r"[^A-Za-z0-9_.-]+", "_", suggested)
        export_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Remote-Regulation Evidence",
            suggested,
            "JSON evidence (*.json)",
        )
        if not export_path:
            return

        def json_ready(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(key): json_ready(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [json_ready(item) for item in value]
            return value

        path_scope = dict(active.get("path_scope") or {})
        profile = dict(active.get("residue_change_profile") or {})
        hotspots = [dict(row) for row in profile.get("hotspots", []) or []]
        selected_region_id = str(getattr(self.state, "analysis_selection", {}).get("region_id") or "")
        selected_hotspot = next(
            (
                row for row in hotspots
                if str(row.get("region_id") or row.get("hotspot_id") or "") == selected_region_id
            ),
            hotspots[0] if hotspots else {},
        )
        selected_region_id = str(
            selected_hotspot.get("region_id") or selected_hotspot.get("hotspot_id") or "full path"
        )
        composition_view = self._dataset_compare_composition_evidence(active, profile, selected_hotspot)
        distance_view = self._dataset_compare_distance_evidence(active, profile, selected_hotspot)
        temporal_view = self._dataset_compare_temporal_evidence(active, selected_hotspot) if selected_hotspot else {}
        evidence = {
            "schema": "hex.linked-case-evidence.v2",
            "exported_at_unix": float(time.time()),
            "claim_boundary": (
                "A mutation-to-active-site distance classifies a distal perturbation candidate; "
                "this export does not establish causal allosteric transmission."
            ),
            "reference": {
                "dataset_key": reference_key,
                "dataset_label": self._dataset_compare_label(reference_key),
                "cluster_id": reference_cluster,
                "path_count": len(path_scope.get("reference_path_ids", []) or []),
            },
            "target": {
                "dataset_key": target_key,
                "dataset_label": self._dataset_compare_label(target_key),
                "cluster_id": target_cluster,
                "path_count": len(
                    dict(path_scope.get("target_path_ids_by_cluster", {}) or {}).get(target_cluster, []) or []
                ),
            },
            "path_scope": {
                key: path_scope.get(key)
                for key in (
                    "keep_ratio", "length_region", "exit_region", "prefiltered_count",
                    "target_cluster_counts", "source_residue_ids",
                )
                if key in path_scope
            },
            "comparison": {
                key: profile.get(key)
                for key in (
                    "position_basis", "threshold", "reference_path_count", "target_path_count",
                    "reference_bottleneck", "target_bottleneck", "bottleneck_fraction_shift",
                    "bottleneck_radius_shift", "substitutions", "remote_distance_threshold",
                    "remote_distance_basis", "active_site_definition", "remote_candidate_count",
                    "allosteric_evidence", "hotspots", "bins",
                )
                if key in profile
            },
            "linked_views": {
                "selection": {
                    "region_id": selected_region_id,
                    "position_basis": str(profile.get("position_basis") or "physical_arc_length"),
                    "reference_path_count": len(path_scope.get("reference_path_ids", []) or []),
                    "target_path_count": len(
                        dict(path_scope.get("target_path_ids_by_cluster", {}) or {}).get(target_cluster, []) or []
                    ),
                    "keep_ratio": path_scope.get("keep_ratio"),
                },
                "panel_c_composition": composition_view,
                "panel_d_bottleneck": {
                    "reference": profile.get("reference_bottleneck"),
                    "target": profile.get("target_bottleneck"),
                    "position_basis": profile.get("position_basis"),
                },
                "panel_e_distance": distance_view,
                "panel_f_time": temporal_view,
            },
        }
        try:
            with open(export_path, "w", encoding="utf-8") as handle:
                json.dump(json_ready(evidence), handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.critical(self, "Evidence Export Failed", str(exc))
            return
        self._statusbar.showMessage(f"Evidence exported to {export_path}", 5000)

    def _on_dataset_compare_region_selected(self, row: dict) -> None:
        """Update linked evidence without replacing the current central view."""
        row = dict(row or {})
        if not row:
            return
        observer_active = self._protein_observer_page_active()
        self._on_dataset_compare_motif_selected(
            row,
            open_observer=observer_active,
            refresh_evidence=False,
        )
        # R1/R2/... changes the analytical selection, not the central
        # workspace. This keeps Residue Observer visible when it is active and
        # likewise preserves the main 3D view when analysis starts there.
        self._set_bottom_analysis_mode("evidence")
        self._refresh_evidence_panel(row, include_temporal=True)

    def _on_dataset_compare_motif_selected(
        self,
        row: dict,
        open_observer: bool = True,
        refresh_evidence: bool = True,
    ):
        row = dict(row or {})
        active = dict(getattr(self, "_dataset_compare_active", None) or {})
        if not row or not active:
            return
        if str(row.get("transition_type") or "") == "path_residue_hotspot":
            hotspot_id = str(row.get("hotspot_id") or "")
            matched_event = next(
                (
                    dict(event)
                    for event in (getattr(self, "_observer_change_events", []) or [])
                    if hotspot_id and str(event.get("hotspot_id") or "") == hotspot_id
                ),
                None,
            )
            if matched_event:
                row = matched_event
        event_id = str(row.get("event_id") or "")
        combo = getattr(self, "_observer_change_event_combo", None)
        if combo is not None and event_id:
            for index in range(combo.count()):
                payload = combo.itemData(index)
                if isinstance(payload, dict) and str(payload.get("event_id") or "") == event_id:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(index)
                    combo.blockSignals(False)
                    break
        if row.get("peak_fraction") is not None:
            peak = max(0.0, min(1.0, float(row.get("peak_fraction", 0.5))))
            start = max(0.0, min(peak, float(
                row.get("hotspot_start_fraction", row.get("start_fraction", peak))
            )))
            end = max(peak, min(1.0, float(
                row.get("hotspot_end_fraction", row.get("end_fraction", peak))
            )))
            self._observer_path_focus_interval = (start, end, peak)
            self._last_observer_overlay_signature = None
            region_id = str(row.get("region_id") or row.get("hotspot_id") or "")
            self.state.set_analysis_selection({
                "region_id": region_id,
                "arc_start": start,
                "arc_end": end,
                "arc_peak": peak,
            })
            profile_hotspot = next((
                dict(item)
                for item in dict(active.get("residue_change_profile") or {}).get("hotspots", []) or []
                if region_id and str(item.get("region_id") or item.get("hotspot_id") or "") == region_id
            ), dict(row))
            if refresh_evidence:
                self._refresh_evidence_panel(
                    profile_hotspot,
                    include_temporal=not bool(open_observer),
                )

        def normalized_ids(values) -> tuple[int, ...]:
            result: list[int] = []
            for value in values or ():
                try:
                    residue_id = int(value)
                except (TypeError, ValueError):
                    continue
                if residue_id > 0 and residue_id not in result:
                    result.append(residue_id)
            return tuple(result)

        reference_ids = normalized_ids(row.get("reference_residue_ids"))
        target_ids = normalized_ids(row.get("target_residue_ids"))
        if not reference_ids and not target_ids:
            return

        reference_key = str(
            row.get("reference_dataset_key") or active.get("reference_key") or ""
        )
        target_key = str(
            row.get("target_dataset_key") or active.get("target_key") or ""
        )
        reference_present = row.get("present_reference")
        target_present = row.get("present_target")
        if reference_present is None:
            reference_present = bool(reference_ids) and float(row.get("reference_frequency", 0.0) or 0.0) > 0.0
        if target_present is None:
            target_present = bool(target_ids) and float(row.get("target_frequency", 0.0) or 0.0) > 0.0

        payload = {
            "source": "dataset_compare_motif",
            "left_dataset_key": reference_key,
            "right_dataset_key": target_key,
            "left_residue_ids": reference_ids,
            "right_residue_ids": target_ids,
            "changed_residue_ids": tuple(row.get("changed_residue_ids", ()) or ()),
            "present_left": bool(reference_present),
            "present_right": bool(target_present),
            "left_title": "Baseline",
            "right_title": "Mutant",
        }
        self._observer_pending_payload = payload
        merged_ids = tuple(dict.fromkeys((*reference_ids, *target_ids)))
        self._observer_residue_input.setText(",".join(str(rid) for rid in merged_ids))
        if open_observer:
            if self._protein_observer_page_active():
                self._update_combination_observer_views(payload)
            else:
                self._toggle_protein_observer_panel(True)

        reference_text = ";".join(f"S{rid + 1}(MD{rid})" for rid in reference_ids) or "absent"
        target_text = ";".join(f"S{rid + 1}(MD{rid})" for rid in target_ids) or "absent"
        if str(row.get("transition_type") or "") == "path_residue_hotspot":
            start_percent = float(row.get("start_fraction", 0.0)) * 100.0
            end_percent = float(row.get("end_fraction", 0.0)) * 100.0
            lost_text = ",".join(
                f"S{int(rid) + 1}(MD{int(rid)})"
                for rid in row.get("lost_residue_ids", ()) or ()
            ) or "none"
            gained_text = ",".join(
                f"S{int(rid) + 1}(MD{int(rid)})"
                for rid in row.get("gained_residue_ids", ()) or ()
            ) or "none"
            transition_text = (
                f"{row.get('region_id', row.get('hotspot_id', 'Region'))} physical arc {start_percent:.0f}-{end_percent:.0f}% | "
                f"Reference/lost {lost_text} -> Target/gained {gained_text}"
            )
        else:
            transition_text = (
                f"{reference_text} {row.get('reference_motif', '')} -> "
                f"{target_text} {row.get('target_motif', '')}"
            ).strip()
        display_alignment_active = bool(
            self._observer_alignment_metadata(reference_key)
            or self._observer_alignment_metadata(target_key)
        )
        alignment_note = (
            " | PDB display uses pipeline alignment"
            if display_alignment_active
            else ""
        )
        impact_text = str(row.get("impact_text") or transition_text)
        explanation = getattr(self, "_observer_change_explanation", None)
        if explanation is not None:
            explanation.setText(impact_text + alignment_note)
        self._protein_observer_debug_label.setText(
            f"Baseline → Mutant: {transition_text} | selected datasets are shown left-to-right{alignment_note}"
        )
        self._statusbar.showMessage(
            f"Residue Observer updated: {transition_text}",
            5000,
        )
        if hasattr(self._dataset_compare_panel, "set_discovery_stage"):
            self._dataset_compare_panel.set_discovery_stage(
                4,
                "Structural verification active. Compare the same arc-length region and residue identities across all selected datasets.",
            )

    def _step_dataset_compare_region(self, delta: int) -> None:
        panel = getattr(self, "_dataset_compare_panel", None)
        if panel is None or not hasattr(panel, "step_hotspot"):
            return
        selection = dict(getattr(self.state, "analysis_selection", {}) or {})
        region_id = str(selection.get("region_id") or "")
        if region_id and hasattr(panel, "select_hotspot"):
            panel.select_hotspot(region_id, emit=False)
        hotspot = panel.step_hotspot(int(delta), emit=False)
        if not hotspot:
            self._statusbar.showMessage("No residue-replacement region is available.", 2500)
            return
        self._on_dataset_compare_motif_selected(
            dict(hotspot),
            open_observer=self._main_view_stack.currentWidget() is self._protein_observer_container,
        )

    def _clear_dataset_compare_observer_selection(self):
        pending = dict(getattr(self, "_observer_pending_payload", {}) or {})
        if pending.get("source") != "dataset_compare_motif":
            return
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if panel is not None:
                panel.set_highlight_residues((), focus_loaded=False)
                if hasattr(panel, "set_highlight_label_specs"):
                    panel.set_highlight_label_specs({}, render=False)
            summary = slot.get("residue_summary") if isinstance(slot, dict) else None
            if summary is not None:
                summary.clear()
                summary.setVisible(False)
            slot["role_title"] = ""
            self._update_observer_slot_title(slot)
        self._observer_pending_payload = {}
        self._observer_path_focus_interval = None
        self._last_observer_overlay_signature = None
        observer_input = getattr(self, "_observer_residue_input", None)
        if observer_input is not None:
            observer_input.clear()
        debug_label = getattr(self, "_protein_observer_debug_label", None)
        if debug_label is not None:
            debug_label.setText("Select a path-position event to compare Baseline and Mutant residue environments.")
        explanation = getattr(self, "_observer_change_explanation", None)
        if explanation is not None:
            explanation.setText("Choose a physical arc-length position to compare baseline and mutant residues.")

    def _clear_dataset_compare(self):
        self._clear_dataset_compare_observer_selection()
        self._dataset_compare_active = None
        self.state.clear_analysis_selection()
        self._observer_change_events = []
        self._refresh_observer_change_event_selector()
        self._viewer.clear_matched()
        self._viewer.clear_dataset_compare_motifs(render=False)
        self._viewer.clear_dataset_compare_residue_hotspots(render=False)
        self._viewer.clear_dataset_compare_bottlenecks(render=False)
        self._viewer.set_protein_highlight_residues([], refresh=True, render=True)
        self._viewer.clear_residues()
        self._dataset_compare_panel.clear_result()
        self._refresh_evidence_panel()
        self._statusbar.showMessage("Dataset comparison highlight cleared", 2500)

    def _on_dataset_visibility_changed(self, key: str, visible: bool):
        self.state.set_dataset_visible(key, visible)
        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
            self._residue_panel.load_options()
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        action = "shown" if visible else "hidden"
        self._statusbar.showMessage(f"Dataset {label} tunnels {action}", 3000)

    def _on_dataset_visibility_batch_changed(self, keys, visible: bool):
        dataset_labels: list[str] = []
        normalized_keys: list[str] = []
        for key in keys or []:
            normalized = str(key or "").strip()
            if not normalized:
                continue
            normalized_keys.append(normalized)
            dataset = self.db.get_dataset(normalized) if hasattr(self.db, "get_dataset") else None
            dataset_labels.append(dataset["prefix"] if dataset else normalized)
        if normalized_keys:
            self.state.set_dataset_visibility_batch(normalized_keys, visible)
        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
            self._residue_panel.load_options()
        if not dataset_labels:
            return
        action = "shown" if visible else "hidden"
        self._statusbar.showMessage(
            f"{len(dataset_labels)} datasets tunnels {action}: {', '.join(dataset_labels[:4])}"
            + (" ..." if len(dataset_labels) > 4 else ""),
            3000,
        )

    def _on_paths_selected(self, path_ids):
        """Multiple paths selected → update 3D only; charts via Sync button."""
        if self._suppress_next_path_preview:
            self._suppress_next_path_preview = False
            return
        if path_ids and len(path_ids) == 1:
            return
        if not path_ids:
            self._last_preview_path_ids = []
            self._last_highlighted_preview_ids = set()
            self._last_preview_color_map = {}
            self._contained_match_preview_active = False
            self._viewer.clear_matched()
            self._refresh_statistics_selection_label()
            return

        # Render in 3D
        preview_ids = [int(pid) for pid in path_ids[:200]]
        self._last_preview_path_ids = preview_ids
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        coords = self._get_display_path_coords(self._filter_hidden_dataset_paths(preview_ids))
        self._viewer.render_matched_paths(coords)

        self._chart_status_label.setText(
            f"{len(path_ids)} paths selected — click Sync Charts to update"
        )
        self._refresh_statistics_selection_label()

    def _on_path_clicked(self, path_id):
        """Single path clicked → select it in the main 3D view."""
        if self._suppress_next_path_preview:
            self._suppress_next_path_preview = False
            return
        path_id = int(path_id)
        self.state.apply_lasso({path_id}, "replace")
        self._schedule_chart_update()
        self._last_preview_path_ids = [path_id]
        self._last_highlighted_preview_ids = {path_id}
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._viewer.clear_matched()
        self._chart_status_label.setText("1 path selected")
        self._refresh_statistics_selection_label()

    def _on_viewer_path_clicked(self, path_id: int):
        """In Focus mode, clicking a visible path removes it from the current selection."""
        if self.state.focus_mode and path_id in self.state.effective_path_ids:
            self.state.apply_lasso({path_id}, "remove")
            self._schedule_chart_update()
            self._statusbar.showMessage(
                f"Removed path {self._format_path_label(path_id)} from selection "
                f"({len(self.state.effective_path_ids)} remaining)",
                3000,
            )
            return
        self._on_path_clicked(path_id)

    def _on_load_selected_profiles(self, path_ids):
        """Load Selected clicked → update profile/statistics charts only."""
        if not path_ids:
            return
        profiles, cache_hit = self._cached_path_profiles(path_ids)
        self._set_chart_profiles(profiles)
        self._statusbar.showMessage(
            f"Loaded profiles for {len(profiles)} selected paths"
            + (" (cached)" if cache_hit else ""),
            3000,
        )

    def _on_path_color_changed(self, value):
        """Map slider 0-100 to light gray (#C8C8C8) → black (#000000) for background paths."""
        gray = int(200 * (1 - value / 100.0))
        color = f"#{gray:02X}{gray:02X}{gray:02X}"
        self._viewer.set_bg_color(color)

    def _on_bg_opacity_changed(self, value):
        opacity = value / 100.0
        self.state.bg_opacity = opacity
        self._viewer.set_bg_opacity(opacity)

    def _statistics_context_path_ids(self) -> list[int]:
        selected = self._path_panel_selected_path_ids()
        if selected:
            return sorted(selected)
        effective = {
            int(path_id)
            for path_id in getattr(self.state, "effective_path_ids", set())
            if int(path_id) > 0
        }
        if effective:
            return sorted(effective)
        visible = self.state.get_visible_category_path_ids()
        return sorted(int(path_id) for path_id in visible if int(path_id) > 0)

    def _refresh_statistics_selection_label(self):
        label = getattr(self, "_statistics_selection_label", None)
        if label is None:
            return
        count = len(self._statistics_context_path_ids())
        label.setText(f"Selected: {count:,} paths")

    @staticmethod
    def _format_stat_float(value: float) -> str:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4f}"

    def _set_statistics_table(self, headers: list[str], rows: list[dict], summary: str):
        self._statistics_last_headers = list(headers)
        self._statistics_last_rows = [dict(row) for row in rows]
        table = getattr(self, "_statistics_table", None)
        if table is None:
            return
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        for row_idx, row in enumerate(rows):
            for col_idx, header in enumerate(headers):
                value = row.get(header, "")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)
        table.resizeColumnsToContents()
        if headers:
            table.horizontalHeader().setStretchLastSection(True)
        self._statistics_summary_label.setText(summary)
        self._stats_export_btn.setEnabled(bool(rows))
        self._refresh_statistics_selection_label()

    def _current_and_original_coords_for_statistics(self, path_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        path_ids = [int(path_id) for path_id in path_ids if int(path_id) > 0]
        if not path_ids:
            return {}, {}
        loaded = dict(getattr(self._viewer, "_bg_data_dict", {}) or {})
        if self._path_coordinate_mode == "current":
            current = {
                int(path_id): coords
                for path_id, coords in loaded.items()
                if int(path_id) in path_ids
            }
            missing_current = set(path_ids) - set(current)
            if missing_current:
                current.update(self.db.get_render_coords(list(missing_current)))
            original = self.db.get_original_render_coords(path_ids)
        else:
            original = {
                int(path_id): coords
                for path_id, coords in loaded.items()
                if int(path_id) in path_ids
            }
            missing_original = set(path_ids) - set(original)
            if missing_original:
                original.update(self.db.get_original_render_coords(list(missing_original)))
            current = self.db.get_render_coords(path_ids)
        return current, original

    def _compute_statistics_entrance_centroid(self):
        path_ids = self._statistics_context_path_ids()
        if not path_ids:
            QMessageBox.information(self, "Statistics", "No selected paths are available.")
            return
        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_points = [
            np.asarray(coords, dtype=np.float64)[0, :3]
            for coords in current.values()
            if np.asarray(coords).ndim == 2 and len(coords) > 0
        ]
        original_points = [
            np.asarray(coords, dtype=np.float64)[0, :3]
            for coords in original.values()
            if np.asarray(coords).ndim == 2 and len(coords) > 0
        ]
        if not current_points and not original_points:
            QMessageBox.information(self, "Statistics", "No entrance coordinates were available for the selected paths.")
            return

        rows: list[dict] = []
        if current_points:
            centroid = np.mean(np.vstack(current_points), axis=0)
            rows.append({
                "Metric": "Mapped/current entrance centroid",
                "Path Count": len(current_points),
                "X": self._format_stat_float(float(centroid[0])),
                "Y": self._format_stat_float(float(centroid[1])),
                "Z": self._format_stat_float(float(centroid[2])),
            })
        if original_points:
            centroid = np.mean(np.vstack(original_points), axis=0)
            rows.append({
                "Metric": "Original entrance centroid",
                "Path Count": len(original_points),
                "X": self._format_stat_float(float(centroid[0])),
                "Y": self._format_stat_float(float(centroid[1])),
                "Z": self._format_stat_float(float(centroid[2])),
            })
        self._set_statistics_table(
            ["Metric", "Path Count", "X", "Y", "Z"],
            rows,
            f"Entrance centroids calculated from {len(path_ids):,} selected paths.",
        )
        self._statusbar.showMessage("Entrance centroid statistics calculated", 3000)

    @staticmethod
    def _mean_centerline_and_distances(coords_by_id: dict[int, np.ndarray], sample_count: int = 64) -> tuple[np.ndarray | None, dict[int, float]]:
        samples: dict[int, np.ndarray] = {}
        for path_id, coords in coords_by_id.items():
            sample = MainWindow._resample_path_coords_interpolated(coords, sample_count=sample_count)
            if sample.ndim == 2 and sample.shape[0] == int(sample_count):
                samples[int(path_id)] = sample.astype(np.float64)
        if not samples:
            return None, {}
        stack = np.stack(list(samples.values()), axis=0)
        centerline = np.mean(stack, axis=0)
        distances = {
            int(path_id): float(np.sum(np.linalg.norm(sample - centerline, axis=1)))
            for path_id, sample in samples.items()
        }
        return centerline, distances

    def _compute_statistics_centerline_distance(self):
        path_ids = self._statistics_context_path_ids()
        if not path_ids:
            QMessageBox.information(self, "Statistics", "No selected paths are available.")
            return
        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_centerline, current_distances = self._mean_centerline_and_distances(current)
        original_centerline, original_distances = self._mean_centerline_and_distances(original)
        if not current_distances and not original_distances:
            QMessageBox.information(self, "Statistics", "No path coordinates were available for centerline statistics.")
            return

        all_ids = sorted(set(current_distances) | set(original_distances))
        rows: list[dict] = []
        for path_id in all_ids:
            current_value = current_distances.get(int(path_id))
            original_value = original_distances.get(int(path_id))
            delta_value = None
            if current_value is not None and original_value is not None:
                delta_value = current_value - original_value
            rows.append({
                "Path": self._format_path_label(int(path_id)),
                "Current Sum Dist": self._format_stat_float(current_value) if current_value is not None else "",
                "Original Sum Dist": self._format_stat_float(original_value) if original_value is not None else "",
                "Delta Current-Original": self._format_stat_float(delta_value) if delta_value is not None else "",
            })

        current_total = float(np.sum(list(current_distances.values()))) if current_distances else float("nan")
        original_total = float(np.sum(list(original_distances.values()))) if original_distances else float("nan")
        summary_parts = []
        if current_centerline is not None:
            summary_parts.append(
                f"mapped/current total summed distance {self._format_stat_float(current_total)}"
            )
        if original_centerline is not None:
            summary_parts.append(
                f"original total summed distance {self._format_stat_float(original_total)}"
            )
        self._set_statistics_table(
            ["Path", "Current Sum Dist", "Original Sum Dist", "Delta Current-Original"],
            rows,
            f"Centerline distances for {len(all_ids):,} paths; " + "; ".join(summary_parts) + ".",
        )
        self._statusbar.showMessage("Centerline distance statistics calculated", 3000)

    def _current_statistics_dataset_key(self) -> str:
        panel = getattr(self, "_dataset_panel", None)
        if panel is not None and hasattr(panel, "current_dataset_key"):
            key = panel.current_dataset_key()
            if key:
                return str(key)
        keys = []
        if panel is not None and hasattr(panel, "selected_dataset_keys"):
            keys = list(panel.selected_dataset_keys() or [])
        if len(keys) == 1:
            return str(keys[0])
        primary_key = getattr(self.db, "primary_dataset_key", None)
        if callable(primary_key):
            value = primary_key()
            return str(value or "")
        return str(primary_key or "")

    def _select_dataset_clusters_by_id_range(self) -> tuple[str, dict[int, set[int]], list[int], int, int] | None:
        dataset_key = self._current_statistics_dataset_key()
        if not dataset_key:
            QMessageBox.information(self, "Statistics", "No current dataset is available.")
            return None
        if not hasattr(self.db, "get_dataset_cluster_path_groups"):
            QMessageBox.information(self, "Statistics", "Cluster statistics are unavailable for the current database.")
            return None

        cluster_groups = self.db.get_dataset_cluster_path_groups(
            dataset_key,
            self.state.frame_min,
            self.state.frame_max,
        )
        available_cluster_ids = sorted(
            int(cluster_id)
            for cluster_id, ids in cluster_groups.items()
            if len(ids or set()) > 0
        )
        if not available_cluster_ids:
            QMessageBox.information(self, "Statistics", "No clusters were found in the current dataset.")
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Cluster ID Range")
        dialog_layout = QVBoxLayout(dialog)
        form_layout = QGridLayout()
        start_spin = QSpinBox(dialog)
        end_spin = QSpinBox(dialog)
        for spin in (start_spin, end_spin):
            spin.setRange(0, 1_000_000_000)
            spin.setSingleStep(1)
        default_start = min(available_cluster_ids)
        default_end = min(max(available_cluster_ids), max(default_start, 10))
        start_spin.setValue(int(default_start))
        end_spin.setValue(int(default_end))
        form_layout.addWidget(QLabel("Cluster ID From"), 0, 0)
        form_layout.addWidget(start_spin, 0, 1)
        form_layout.addWidget(QLabel("Cluster ID To"), 1, 0)
        form_layout.addWidget(end_spin, 1, 1)
        dialog_layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        cluster_id_start = int(start_spin.value())
        cluster_id_end = int(end_spin.value())
        if cluster_id_start > cluster_id_end:
            QMessageBox.information(self, "Statistics", "Cluster ID From must be less than or equal to Cluster ID To.")
            return None
        target_clusters = [
            int(cluster_id)
            for cluster_id, ids in sorted(cluster_groups.items(), key=lambda item: int(item[0]))
            if int(cluster_id) >= int(cluster_id_start)
            and int(cluster_id) <= int(cluster_id_end)
            and len(ids or set()) > 0
        ]
        if not target_clusters:
            QMessageBox.information(
                self,
                "Statistics",
                f"No clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] were found in the current dataset.",
            )
            return None
        return dataset_key, cluster_groups, target_clusters, int(cluster_id_start), int(cluster_id_end)

    @staticmethod
    def _pairwise_path_compactness(
        coords_by_id: dict[int, np.ndarray],
        sample_count: int = 64,
    ) -> dict:
        samples: list[np.ndarray] = []
        for path_id, coords in sorted(coords_by_id.items()):
            sample = MainWindow._resample_path_coords_interpolated(coords, sample_count=sample_count)
            if sample.ndim == 2 and sample.shape[0] == int(sample_count):
                samples.append(sample.astype(np.float64))
        path_count = len(samples)
        if path_count < 2:
            return {"path_count": path_count, "pair_count": 0, "total_pair_count": 0}

        stack = np.stack(samples, axis=0)
        total_pairs = path_count * (path_count - 1) // 2
        distances: list[np.ndarray] = []
        for idx in range(path_count - 1):
            pair_distances = np.sum(
                np.linalg.norm(stack[idx + 1:] - stack[idx], axis=2),
                axis=1,
            )
            if pair_distances.size:
                distances.append(pair_distances.astype(np.float64, copy=False))
        if not distances:
            return {"path_count": path_count, "pair_count": 0, "total_pair_count": int(total_pairs)}
        values = np.concatenate(distances)
        return {
            "path_count": path_count,
            "pair_count": int(values.size),
            "total_pair_count": int(total_pairs),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }

    @staticmethod
    def _jaccard_filtered_pairwise_path_compactness(
        current_coords_by_id: dict[int, np.ndarray],
        original_coords_by_id: dict[int, np.ndarray],
        residue_sets_by_id: dict[int, set[int]],
        jaccard_threshold: float,
        atom_sets_by_id: dict[int, set[object]] | None = None,
        atom_jaccard_threshold: float | None = None,
        sample_count: int = 64,
        precomputed_candidates: dict | None = None,
    ) -> dict:
        candidate_info = precomputed_candidates or MainWindow._jaccard_filtered_pair_candidates(
            residue_sets_by_id,
            float(jaccard_threshold),
            atom_sets_by_id,
            atom_jaccard_threshold,
        )
        residue_sets: dict[int, set[int]] = {
            int(path_id): set(values)
            for path_id, values in dict(candidate_info.get("_residue_sets", {})).items()
            if values
        }
        path_count = int(candidate_info.get("path_count", len(residue_sets)) or 0)
        total_pairs = int(candidate_info.get("total_pair_count", path_count * (path_count - 1) // 2) or 0)
        threshold = float(jaccard_threshold)
        current_values: list[float] = []
        original_values: list[float] = []
        delta_values: list[float] = []
        abs_delta_values: list[float] = []
        jaccard_values: list[float] = []
        atom_jaccard_values: list[float] = []
        atom_sets = atom_sets_by_id or {}
        atom_threshold = float(atom_jaccard_threshold) if atom_jaccard_threshold is not None else None
        use_atom_filter = atom_threshold is not None
        candidate_pairs = set(candidate_info.get("_candidate_pairs", set()) or set())
        valid_path_ids = {int(path_id) for pair in candidate_pairs for path_id in pair}
        current_samples: dict[int, np.ndarray] = {}
        original_samples: dict[int, np.ndarray] = {}
        for path_id in sorted(valid_path_ids):
            if int(path_id) not in current_coords_by_id or int(path_id) not in original_coords_by_id:
                continue
            current_sample = MainWindow._resample_path_coords_interpolated(
                current_coords_by_id[int(path_id)],
                sample_count=sample_count,
            )
            original_sample = MainWindow._resample_path_coords_interpolated(
                original_coords_by_id[int(path_id)],
                sample_count=sample_count,
            )
            if (
                current_sample.ndim == 2
                and original_sample.ndim == 2
                and current_sample.shape[0] == int(sample_count)
                and original_sample.shape[0] == int(sample_count)
            ):
                current_samples[int(path_id)] = current_sample.astype(np.float64)
                original_samples[int(path_id)] = original_sample.astype(np.float64)
        increased_pair_count = 0
        decreased_pair_count = 0
        near_zero_pair_count = 0
        near_zero_tolerance = 1e-6

        for left_id, right_id in sorted(candidate_pairs):
            if int(left_id) not in current_samples or int(right_id) not in current_samples:
                continue
            left_set = residue_sets[int(left_id)]
            right_set = residue_sets[int(right_id)]
            union = left_set | right_set
            if not union:
                continue
            jaccard = float(len(left_set & right_set)) / float(len(union))
            if jaccard <= threshold:
                continue

            atom_jaccard: float | None = None
            left_atom_set = atom_sets.get(int(left_id), set())
            right_atom_set = atom_sets.get(int(right_id), set())
            atom_union = left_atom_set | right_atom_set
            if atom_union:
                atom_jaccard = float(len(left_atom_set & right_atom_set)) / float(len(atom_union))
            elif use_atom_filter:
                continue
            if use_atom_filter and (atom_jaccard is None or atom_jaccard <= float(atom_threshold)):
                continue

            current_diff = current_samples[int(right_id)] - current_samples[int(left_id)]
            original_diff = original_samples[int(right_id)] - original_samples[int(left_id)]
            current_distance = float(np.sum(np.linalg.norm(current_diff, axis=1)))
            original_distance = float(np.sum(np.linalg.norm(original_diff, axis=1)))
            distance_delta = current_distance - original_distance
            current_values.append(current_distance)
            original_values.append(original_distance)
            delta_values.append(distance_delta)
            abs_delta_values.append(abs(distance_delta))
            if abs(distance_delta) <= near_zero_tolerance:
                near_zero_pair_count += 1
            elif distance_delta > 0.0:
                increased_pair_count += 1
            else:
                decreased_pair_count += 1
            jaccard_values.append(float(jaccard))
            if atom_jaccard is not None:
                atom_jaccard_values.append(float(atom_jaccard))

        result = {
            "path_count": path_count,
            "pair_count": len(current_values),
            "total_pair_count": int(total_pairs),
            "skipped_pair_count": int(candidate_info.get("skipped_pair_count", 0) or 0),
            "residue_skipped_pair_count": int(candidate_info.get("residue_skipped_pair_count", 0) or 0),
            "atom_skipped_pair_count": int(candidate_info.get("atom_skipped_pair_count", 0) or 0),
            "missing_atom_pair_count": int(candidate_info.get("missing_atom_pair_count", 0) or 0),
            "residue_shared_pair_count": int(candidate_info.get("residue_shared_pair_count", 0) or 0),
            "residue_candidate_pair_count": int(candidate_info.get("residue_candidate_pair_count", 0) or 0),
            "atom_shared_pair_count": int(candidate_info.get("atom_shared_pair_count", 0) or 0),
            "atom_candidate_pair_count": int(candidate_info.get("atom_candidate_pair_count", 0) or 0),
            "atom_pair_count": len(atom_jaccard_values),
            "increased_pair_count": int(increased_pair_count),
            "decreased_pair_count": int(decreased_pair_count),
            "near_zero_pair_count": int(near_zero_pair_count),
        }
        for prefix, values in (
            ("current", current_values),
            ("original", original_values),
            ("distance_delta", delta_values),
            ("abs_distance_delta", abs_delta_values),
            ("jaccard", jaccard_values),
            ("atom_jaccard", atom_jaccard_values),
        ):
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            result[f"{prefix}_mean"] = float(np.mean(arr))
            result[f"{prefix}_median"] = float(np.median(arr))
            result[f"{prefix}_min"] = float(np.min(arr))
            result[f"{prefix}_max"] = float(np.max(arr))
            result[f"{prefix}_std"] = float(np.std(arr))
        return result

    @staticmethod
    def _jaccard_filtered_pair_candidates(
        residue_sets_by_id: dict[int, set[int]],
        jaccard_threshold: float,
        atom_sets_by_id: dict[int, set[object]] | None = None,
        atom_jaccard_threshold: float | None = None,
    ) -> dict:
        residue_sets: dict[int, set[int]] = {}
        for path_id, raw_set in sorted((residue_sets_by_id or {}).items()):
            residue_set = {int(residue_id) for residue_id in raw_set if int(residue_id) > 0}
            if residue_set:
                residue_sets[int(path_id)] = residue_set

        path_ids = sorted(residue_sets)
        path_count = len(path_ids)
        total_pairs = path_count * (path_count - 1) // 2
        if path_count < 2:
            return {
                "_residue_sets": residue_sets,
                "_candidate_pairs": set(),
                "path_count": path_count,
                "total_pair_count": int(total_pairs),
                "skipped_pair_count": 0,
                "residue_skipped_pair_count": 0,
                "atom_skipped_pair_count": 0,
                "missing_atom_pair_count": 0,
                "residue_shared_pair_count": 0,
                "residue_candidate_pair_count": 0,
                "atom_shared_pair_count": 0,
                "atom_candidate_pair_count": 0,
            }

        residue_candidate_pairs, residue_prefix_pair_count = MainWindow._jaccard_candidate_pairs_by_inverted_index(
            residue_sets,
            float(jaccard_threshold),
        )
        atom_candidate_pairs: set[tuple[int, int]] | None = None
        atom_prefix_pair_count = 0
        atom_sets = atom_sets_by_id or {}
        if atom_jaccard_threshold is not None:
            atom_candidate_pairs, atom_prefix_pair_count = MainWindow._jaccard_candidate_pairs_by_inverted_index(
                {
                    int(path_id): set(atom_sets.get(int(path_id), set()))
                    for path_id in path_ids
                    if atom_sets.get(int(path_id))
                },
                float(atom_jaccard_threshold),
            )

        candidate_pairs = set(residue_candidate_pairs)
        atom_skipped_pair_count = 0
        missing_atom_pair_count = 0
        if atom_candidate_pairs is not None:
            candidate_pairs &= atom_candidate_pairs
            for left_id, right_id in residue_candidate_pairs - atom_candidate_pairs:
                left_atom_set = atom_sets.get(int(left_id), set())
                right_atom_set = atom_sets.get(int(right_id), set())
                if not (left_atom_set | right_atom_set):
                    missing_atom_pair_count += 1
                else:
                    atom_skipped_pair_count += 1

        return {
            "_residue_sets": residue_sets,
            "_candidate_pairs": candidate_pairs,
            "path_count": path_count,
            "total_pair_count": int(total_pairs),
            "skipped_pair_count": int(total_pairs) - len(candidate_pairs),
            "residue_skipped_pair_count": int(total_pairs) - len(residue_candidate_pairs),
            "atom_skipped_pair_count": int(atom_skipped_pair_count),
            "missing_atom_pair_count": int(missing_atom_pair_count),
            "residue_shared_pair_count": int(residue_prefix_pair_count),
            "residue_candidate_pair_count": len(residue_candidate_pairs),
            "atom_shared_pair_count": int(atom_prefix_pair_count),
            "atom_candidate_pair_count": len(atom_candidate_pairs) if atom_candidate_pairs is not None else 0,
        }

    @staticmethod
    def _pairwise_residue_similarity_stats(
        residue_sets_by_id: dict[int, set[int]],
    ) -> dict:
        residue_sets = [
            set(residue_sets_by_id[path_id])
            for path_id in sorted(residue_sets_by_id)
            if residue_sets_by_id.get(path_id)
        ]
        path_count = len(residue_sets)
        if path_count < 2:
            unique_residues = set().union(*residue_sets) if residue_sets else set()
            return {
                "path_count": path_count,
                "pair_count": 0,
                "total_pair_count": 0,
                "unique_residue_count": len(unique_residues),
                "core_residue_count": len(unique_residues) if path_count == 1 else 0,
            }

        total_pairs = path_count * (path_count - 1) // 2
        pairs = [
            (left, right)
            for left in range(path_count - 1)
            for right in range(left + 1, path_count)
        ]

        values: list[float] = []
        for left, right in pairs:
            union = residue_sets[int(left)] | residue_sets[int(right)]
            if not union:
                continue
            intersection = residue_sets[int(left)] & residue_sets[int(right)]
            values.append(float(len(intersection)) / float(len(union)))
        unique_residues: set[int] = set().union(*residue_sets)
        core_threshold = max(1, int(np.ceil(path_count * 0.8)))
        residue_counts: dict[int, int] = {}
        for residue_set in residue_sets:
            for residue_id in residue_set:
                residue_counts[int(residue_id)] = residue_counts.get(int(residue_id), 0) + 1
        core_residue_count = sum(1 for count in residue_counts.values() if int(count) >= core_threshold)
        if not values:
            return {
                "path_count": path_count,
                "pair_count": 0,
                "total_pair_count": int(total_pairs),
                "unique_residue_count": len(unique_residues),
                "core_residue_count": int(core_residue_count),
            }
        arr = np.asarray(values, dtype=np.float64)
        return {
            "path_count": path_count,
            "pair_count": int(arr.size),
            "total_pair_count": int(total_pairs),
            "unique_residue_count": len(unique_residues),
            "core_residue_count": int(core_residue_count),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    def _dataset_residue_atom_key_map_for_statistics(self, dataset_key: str) -> dict[int, set[tuple[int, int]]]:
        protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
        aligned_atoms = list(protein_cache.get("atoms_aligned", []) or [])
        if not aligned_atoms:
            return {}

        residue_id_base = 0
        datasets_by_key = getattr(self.db, "_datasets_by_key", {})
        dataset = datasets_by_key.get(str(dataset_key)) if isinstance(datasets_by_key, dict) else None
        if dataset is not None:
            residue_id_base = int(getattr(dataset, "residue_id_base", 0) or 0)

        residue_atom_keys: dict[int, set[tuple[int, int]]] = {}
        for atom_idx, atom in enumerate(aligned_atoms):
            try:
                local_residue_id = int(atom.get("res_seq", 0) or 0)
            except Exception:
                continue
            if local_residue_id <= 0:
                continue
            residue_id = int(residue_id_base) + int(local_residue_id)
            residue_atom_keys.setdefault(residue_id, set()).add((residue_id, int(atom_idx)))
        return residue_atom_keys

    def _dataset_info_for_statistics(self, dataset_key: str) -> dict:
        if hasattr(self.db, "get_dataset"):
            dataset_info = self.db.get_dataset(str(dataset_key))
            if dataset_info:
                return dict(dataset_info)
        return {}

    @staticmethod
    def _find_named_file_near_dataset(dataset_info: dict, filename: str) -> str:
        target = str(filename or "").lower()
        candidates: list[tuple[int, str]] = []
        search_roots = [
            str(dataset_info.get("folder") or ""),
            os.path.dirname(str(dataset_info.get("path") or "")),
            os.path.dirname(str(dataset_info.get("residue_statistics_path") or "")),
        ]
        expanded_roots: list[tuple[int, str]] = []
        for root in search_roots:
            if not root:
                continue
            expanded_roots.append((0, root))
            parent = os.path.dirname(root)
            if parent and parent != root:
                expanded_roots.append((2, parent))
                try:
                    for name in os.listdir(parent):
                        child = os.path.join(parent, name)
                        if os.path.isdir(child):
                            expanded_roots.append((1, child))
                except OSError:
                    pass
        seen_roots: set[str] = set()
        for priority, root in expanded_roots:
            if not root or not os.path.isdir(root):
                continue
            normalized_root = os.path.normcase(os.path.abspath(root))
            if normalized_root in seen_roots:
                continue
            seen_roots.add(normalized_root)
            for current_root, _dirs, files in os.walk(root):
                for name in files:
                    if name.lower() == target:
                        candidates.append((int(priority), os.path.join(current_root, name)))
        if not candidates:
            return ""
        unique: dict[str, int] = {}
        for priority, path in candidates:
            abs_path = os.path.abspath(path)
            unique[abs_path] = min(priority, unique.get(abs_path, priority))
        ranked = sorted(unique.items(), key=lambda item: (item[1], len(item[0]), item[0].lower()))
        return ranked[0][0]

    @staticmethod
    def _find_md_path_file_near_dataset(dataset_info: dict) -> str:
        candidate_roots = [
            str(dataset_info.get("folder") or ""),
            os.path.dirname(str(dataset_info.get("path") or "")),
            os.path.dirname(str(dataset_info.get("residue_statistics_path") or "")),
        ]
        checked: set[str] = set()
        for root in candidate_roots:
            normalized = os.path.abspath(str(root or "").strip()) if str(root or "").strip() else ""
            if not normalized or normalized in checked:
                continue
            checked.add(normalized)
            direct = os.path.join(normalized, "MD_path.txt")
            if os.path.isfile(direct):
                return direct
            parent = os.path.dirname(normalized)
            if parent and parent != normalized:
                parent_direct = os.path.join(parent, "MD_path.txt")
                if os.path.isfile(parent_direct):
                    return parent_direct
        return ""

    @staticmethod
    def _tunnel_id_from_source_file(source_file: str) -> int | None:
        match = re.search(r"tunnel[_-](\d+)", str(source_file or ""), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _safe_csv_int(value) -> int | None:
        try:
            return int(float(str(value or "").strip()))
        except (TypeError, ValueError):
            return None

    def _all_points_path_atom_keys(self, all_points_path: str) -> dict[int, tuple[int, int]]:
        abs_path = os.path.abspath(all_points_path)
        stat = os.stat(abs_path)
        cache_key = ("all_points_path_keys", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        if cache_key in cache:
            return dict(cache[cache_key])

        result: dict[int, tuple[int, int]] = {}
        path_index = -1
        with open(abs_path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            required = {"res_seq", "source_file", "frame_id"}
            if not required.issubset(set(reader.fieldnames)):
                return {}
            current_key: tuple[int, int] | None = None
            for row in reader:
                res_seq = self._safe_csv_int(row.get("res_seq"))
                if res_seq == 1 or path_index < 0:
                    if current_key is not None and path_index >= 0:
                        result[int(path_index)] = current_key
                    path_index += 1
                    frame_id = self._safe_csv_int(row.get("frame_id"))
                    tunnel_id = self._tunnel_id_from_source_file(str(row.get("source_file") or ""))
                    current_key = (
                        int(frame_id) if frame_id is not None else -1,
                        int(tunnel_id) if tunnel_id is not None else -1,
                    )
            if current_key is not None and path_index >= 0:
                result[int(path_index)] = current_key

        cache[cache_key] = dict(result)
        self._statistics_atom_csv_cache = cache
        return result

    def _atom_network_key_atom_sets(self, atom_network_path: str) -> dict[tuple[int, int], set[str]]:
        abs_path = os.path.abspath(atom_network_path)
        stat = os.stat(abs_path)
        cache_key = ("atom_network_key_sets", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        if cache_key in cache:
            return {key: set(value) for key, value in cache[cache_key].items()}

        result: dict[tuple[int, int], set[str]] = defaultdict(set)
        atom_columns = ("T_F_A_1_ID", "T_F_A_2_ID", "T_F_A_3_ID", "T_F_A_4_ID")
        with open(abs_path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            required = {"pdb_id", "Tunnel_ID", *atom_columns}
            if not required.issubset(set(reader.fieldnames)):
                return {}
            for row in reader:
                frame_id = self._safe_csv_int(row.get("pdb_id"))
                tunnel_id = self._safe_csv_int(row.get("Tunnel_ID"))
                if frame_id is None or tunnel_id is None or tunnel_id <= 0:
                    continue
                atom_set = result[(int(frame_id), int(tunnel_id))]
                for column in atom_columns:
                    atom_id = str(row.get(column) or "").strip()
                    if atom_id and atom_id != "0" and "_" in atom_id:
                        atom_set.add(atom_id)

        materialized = {key: set(value) for key, value in result.items() if value}
        cache[cache_key] = {key: set(value) for key, value in materialized.items()}
        self._statistics_atom_csv_cache = cache
        return materialized

    @staticmethod
    def _atom_set_from_path_payload(path_payload: dict) -> set[str]:
        atom_set: set[str] = set()
        for key in ("atom_1", "atom_2", "atom_3", "atom_4"):
            for value in path_payload.get(key, []) or []:
                atom_id = str(value or "").strip()
                if atom_id and atom_id != "0":
                    atom_set.add(atom_id)
        return atom_set

    @staticmethod
    def _jaccard_candidate_pairs_by_inverted_index(
        sets_by_id: dict[int, set],
        threshold: float,
    ) -> tuple[set[tuple[int, int]], int]:
        items = {
            int(path_id): set(values)
            for path_id, values in (sets_by_id or {}).items()
            if values
        }
        if len(items) < 2:
            return set(), 0
        threshold = float(threshold)
        if threshold >= 1.0:
            return set(), 0

        value_counts: dict[object, int] = defaultdict(int)
        for values in items.values():
            for value in values:
                value_counts[value] += 1

        ordered_items: dict[int, list[object]] = {}
        for path_id, values in items.items():
            ordered_items[int(path_id)] = sorted(
                values,
                key=lambda value: (int(value_counts[value]), str(type(value)), str(value)),
            )

        # Prefix filtering keeps the Jaccard result exact while reducing the pair
        # candidates inserted into the inverted index.
        inverted: dict[object, list[int]] = defaultdict(list)
        for path_id, values in ordered_items.items():
            if threshold <= 0.0:
                prefix_length = len(values)
            else:
                prefix_length = len(values) - int(math.ceil(threshold * len(values))) + 1
            prefix_length = max(0, min(len(values), int(prefix_length)))
            for value in values[:prefix_length]:
                inverted[value].append(int(path_id))

        potential_pairs: set[tuple[int, int]] = set()
        for ids in inverted.values():
            if len(ids) < 2:
                continue
            ids = sorted(ids)
            for left_idx, left_id in enumerate(ids[:-1]):
                for right_id in ids[left_idx + 1:]:
                    left_len = len(items[int(left_id)])
                    right_len = len(items[int(right_id)])
                    if threshold > 0.0 and float(min(left_len, right_len)) <= threshold * float(max(left_len, right_len)):
                        continue
                    pair = (int(left_id), int(right_id))
                    potential_pairs.add(pair)

        candidates: set[tuple[int, int]] = set()
        for pair in potential_pairs:
            left_id, right_id = pair
            intersection_count = len(items[left_id] & items[right_id])
            union_count = len(items[left_id]) + len(items[right_id]) - int(intersection_count)
            if union_count > 0 and (float(intersection_count) / float(union_count)) > threshold:
                candidates.add(pair)
        return candidates, len(potential_pairs)

    def _dataset_atom_sets_from_preprocessed_pkl(
        self,
        dataset_key: str,
        path_ids: list[int],
    ) -> tuple[dict[int, set[str]], str]:
        dataset_info = self._dataset_info_for_statistics(dataset_key)
        if not dataset_info:
            return {}, ""
        pkl_path = self._find_named_file_near_dataset(dataset_info, "preprocessed_paths.pkl")
        if not pkl_path:
            return {}, ""

        abs_path = os.path.abspath(pkl_path)
        stat = os.stat(abs_path)
        cache_key = ("preprocessed_pkl_atom_sets", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        path_base = int(getattr(dataset_binding, "path_id_base", 0) or 0) if dataset_binding is not None else 0
        local_ids = sorted({int(path_id) - path_base for path_id in path_ids if int(path_id) - path_base >= 0})
        if cache_key in cache:
            cached_sets = cache[cache_key]
            return {
                int(local_id) + path_base: set(cached_sets[int(local_id)])
                for local_id in local_ids
                if int(local_id) in cached_sets
            }, abs_path

        with open(abs_path, "rb") as handle:
            data = pickle.load(handle)
        paths = list((data or {}).get("PATHS", []) if isinstance(data, dict) else [])
        cached_sets: dict[int, frozenset[str]] = {}
        for local_id, path_payload in enumerate(paths):
            if not isinstance(path_payload, dict):
                continue
            atom_set = self._atom_set_from_path_payload(path_payload)
            if atom_set:
                cached_sets[int(local_id)] = frozenset(atom_set)
        cache[cache_key] = cached_sets
        self._statistics_atom_csv_cache = cache

        result: dict[int, set[str]] = {}
        for local_id in local_ids:
            atom_set = cached_sets.get(int(local_id))
            if atom_set:
                result[int(local_id) + path_base] = set(atom_set)
        return result, abs_path

    def _dataset_atom_sets_from_network_csv(
        self,
        dataset_key: str,
        path_ids: list[int],
    ) -> tuple[dict[int, set[str]], str]:
        dataset_info = self._dataset_info_for_statistics(dataset_key)
        if not dataset_info:
            return {}, ""
        atom_network_path = self._find_named_file_near_dataset(dataset_info, "atom_network.csv")
        if not atom_network_path:
            atom_network_path = self._find_named_file_near_dataset(dataset_info, "1.atom_network.csv")
        all_points_path = self._find_named_file_near_dataset(dataset_info, "mapped_path_points.csv")
        if not all_points_path:
            all_points_path = self._find_named_file_near_dataset(dataset_info, "4.all_points.csv")
        if not atom_network_path or not all_points_path:
            return {}, atom_network_path or ""

        path_keys = self._all_points_path_atom_keys(all_points_path)
        key_atom_sets = self._atom_network_key_atom_sets(atom_network_path)
        if not path_keys or not key_atom_sets:
            return {}, atom_network_path

        result: dict[int, set[str]] = {}
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        path_base = int(getattr(dataset_binding, "path_id_base", 0) or 0) if dataset_binding is not None else 0
        for path_id in path_ids:
            local_path_id = int(path_id) - path_base
            key = path_keys.get(int(local_path_id))
            if key is None:
                continue
            atom_set = key_atom_sets.get(key)
            if atom_set:
                result[int(path_id)] = set(atom_set)
        return result, atom_network_path

    def _compute_dataset_cluster_results_export(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection
        path_ids: list[int] = []
        cluster_rows: list[tuple[int, int]] = []
        for cluster_id in target_clusters:
            ids = sorted(int(path_id) for path_id in cluster_groups.get(int(cluster_id), set()) if int(path_id) > 0)
            for path_id in ids:
                path_ids.append(int(path_id))
                cluster_rows.append((int(cluster_id), int(path_id)))
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_distances: dict[int, float] = {}
        original_distances: dict[int, float] = {}
        path_id_set = set(path_ids)
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) in path_id_set
            }
            if not cluster_path_ids:
                continue
            _current_centerline, cluster_current_distances = self._mean_centerline_and_distances({
                int(path_id): coords
                for path_id, coords in current.items()
                if int(path_id) in cluster_path_ids
            })
            _original_centerline, cluster_original_distances = self._mean_centerline_and_distances({
                int(path_id): coords
                for path_id, coords in original.items()
                if int(path_id) in cluster_path_ids
            })
            current_distances.update(cluster_current_distances)
            original_distances.update(cluster_original_distances)

        headers = ["Cluster", "Path", "Current Sum Dist", "Original Sum Dist", "Delta Current-Original"]
        rows: list[dict] = []
        for cluster_id, path_id in cluster_rows:
            current_value = current_distances.get(int(path_id))
            original_value = original_distances.get(int(path_id))
            delta_value = None
            if current_value is not None and original_value is not None:
                delta_value = current_value - original_value
            rows.append({
                "Cluster": str(cluster_id),
                "Path": self._format_path_label(int(path_id)),
                "Current Sum Dist": self._format_stat_float(current_value) if current_value is not None else "",
                "Original Sum Dist": self._format_stat_float(original_value) if original_value is not None else "",
                "Delta Current-Original": self._format_stat_float(delta_value) if delta_value is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        present_clusters = sorted({int(cluster_id) for cluster_id, _path_id in cluster_rows})
        current_total = float(np.sum(list(current_distances.values()))) if current_distances else float("nan")
        original_total = float(np.sum(list(original_distances.values()))) if original_distances else float("nan")
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster centerline distances for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"{len(rows):,} paths across clusters {', '.join(str(cid) for cid in present_clusters)}; "
                f"mapped/current total summed distance {self._format_stat_float(current_total)}; "
                f"original total summed distance {self._format_stat_float(original_total)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster results for export: {dataset_label} ({len(rows)} paths, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _compute_dataset_cluster_compactness(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection

        path_ids: list[int] = []
        for cluster_id in target_clusters:
            path_ids.extend(
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            )
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        threshold_dialog = QDialog(self)
        threshold_dialog.setWindowTitle("Jaccard Thresholds")
        threshold_layout = QVBoxLayout(threshold_dialog)
        threshold_form = QGridLayout()
        residue_threshold_spin = QDoubleSpinBox(threshold_dialog)
        atom_threshold_spin = QDoubleSpinBox(threshold_dialog)
        for spin in (residue_threshold_spin, atom_threshold_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
        residue_threshold_spin.setValue(0.500)
        atom_threshold_spin.setValue(0.500)
        threshold_form.addWidget(QLabel("Residue Jaccard >"), 0, 0)
        threshold_form.addWidget(residue_threshold_spin, 0, 1)
        threshold_form.addWidget(QLabel("Atom Jaccard >"), 1, 0)
        threshold_form.addWidget(atom_threshold_spin, 1, 1)
        threshold_layout.addLayout(threshold_form)
        threshold_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, threshold_dialog)
        threshold_buttons.accepted.connect(threshold_dialog.accept)
        threshold_buttons.rejected.connect(threshold_dialog.reject)
        threshold_layout.addWidget(threshold_buttons)
        if threshold_dialog.exec() != QDialog.Accepted:
            return
        jaccard_threshold = float(residue_threshold_spin.value())
        atom_jaccard_threshold = float(atom_threshold_spin.value())

        profiles = self.db.get_path_profiles(path_ids)
        residue_sets_by_id: dict[int, set[int]] = {}
        for profile in profiles:
            path_id = int(profile.get("pathIndex", 0) or 0)
            if path_id <= 0:
                continue
            residue_set = profile_residue_set(profile)
            if residue_set:
                residue_sets_by_id[int(path_id)] = set(residue_set)
        if not residue_sets_by_id:
            QMessageBox.information(self, "Statistics", "No residue composition data were available for the selected clusters.")
            return

        self._statusbar.showMessage("Loading path atom sets from preprocessed data...", 4000)
        QApplication.processEvents()
        atom_sets_by_id, atom_source_path = self._dataset_atom_sets_from_preprocessed_pkl(dataset_key, path_ids)
        if not atom_sets_by_id:
            atom_sets_by_id, atom_source_path = self._dataset_atom_sets_from_network_csv(dataset_key, path_ids)
        residue_atom_key_map: dict[int, set[tuple[int, int]]] = {}
        if not atom_sets_by_id:
            residue_atom_key_map = self._dataset_residue_atom_key_map_for_statistics(dataset_key)
            if residue_atom_key_map:
                for path_id, residue_set in residue_sets_by_id.items():
                    atom_set: set[tuple[int, int]] = set()
                    for residue_id in residue_set:
                        atom_set.update(residue_atom_key_map.get(int(residue_id), set()))
                    if atom_set:
                        atom_sets_by_id[int(path_id)] = atom_set
        if not atom_sets_by_id:
            protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
            aligned_atom_count = len(list(protein_cache.get("atoms_aligned", []) or []))
            selected_residue_ids = set().union(*residue_sets_by_id.values()) if residue_sets_by_id else set()
            atom_residue_ids = set(residue_atom_key_map)
            matched_residue_ids = selected_residue_ids & atom_residue_ids
            sample_path_residues = ", ".join(str(value) for value in sorted(selected_residue_ids)[:8])
            sample_atom_residues = ", ".join(str(value) for value in sorted(atom_residue_ids)[:8])
            dataset_info = self._dataset_info_for_statistics(dataset_key)
            dataset_folder = str(dataset_info.get("folder") or "")
            QMessageBox.information(
                self,
                "Statistics",
                (
                    "No atom composition data were available for the selected paths.\n\n"
                    f"Dataset folder: {dataset_folder or 'unknown'}\n"
                    f"Atom source: {atom_source_path or 'not found'}\n"
                    f"Loaded protein atoms: {aligned_atom_count:,}\n"
                    f"Path residue IDs: {len(selected_residue_ids):,}"
                    + (f" (sample: {sample_path_residues})" if sample_path_residues else "")
                    + "\n"
                    f"Protein atom residue IDs: {len(atom_residue_ids):,}"
                    + (f" (sample: {sample_atom_residues})" if sample_atom_residues else "")
                    + "\n"
                    f"Matched residue IDs: {len(matched_residue_ids):,}\n\n"
                    "Load a protein model first, or check that the protein residue numbering matches the path residue IDs."
                ),
            )
            return

        headers = [
            "Cluster",
            "Path Count",
            "Residue Path Count",
            "Candidate Path Count",
            "Can Test",
            "Pair Count",
            "Residue Candidate Pairs",
            "Atom Candidate Pairs",
            "Residue Shared Pairs",
            "Atom Shared Pairs",
            "Skipped Pair Count",
            "Residue Skipped Pairs",
            "Atom Skipped Pairs",
            "Missing Atom Pairs",
            "Total Pair Count",
            "Mean Residue Jaccard",
            "Min Residue Jaccard",
            "Max Residue Jaccard",
            "Atom Path Count",
            "Atom Pair Count",
            "Mean Atom Jaccard",
            "Median Atom Jaccard",
            "Min Atom Jaccard",
            "Max Atom Jaccard",
            "Std Atom Jaccard",
            "Topo-Spatial Gap Mean",
            "Topo-Spatial Gap Median",
            "Abs Gap Mean",
            "Abs Gap Median",
            "Gap Min",
            "Gap Max",
            "Gap Std",
            "Topo Greater Pairs",
            "Topo Smaller Pairs",
            "Gap Stable Pairs",
            "Topo Pair Dist Mean",
            "Spatial Pair Dist Mean",
            "Topo Pair Dist Median",
            "Spatial Pair Dist Median",
            "Topo Pair Dist Min",
            "Spatial Pair Dist Min",
            "Topo Pair Dist Max",
            "Spatial Pair Dist Max",
            "Topo Pair Dist Std",
            "Spatial Pair Dist Std",
        ]
        rows: list[dict] = []
        current_weighted_total = 0.0
        current_weighted_count = 0
        original_weighted_total = 0.0
        original_weighted_count = 0
        delta_weighted_total = 0.0
        delta_weighted_count = 0
        abs_delta_weighted_total = 0.0
        abs_delta_weighted_count = 0
        atom_jaccard_weighted_total = 0.0
        atom_jaccard_weighted_count = 0
        testable_clusters: list[int] = []
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            }
            cluster_residue_sets = {
                int(path_id): residue_set
                for path_id, residue_set in residue_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            }
            residue_path_count = len(cluster_residue_sets)
            cluster_atom_sets = {
                int(path_id): atom_set
                for path_id, atom_set in atom_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            }
            candidate_info = self._jaccard_filtered_pair_candidates(
                cluster_residue_sets,
                float(jaccard_threshold),
                cluster_atom_sets,
                float(atom_jaccard_threshold),
            )
            candidate_path_ids = sorted({
                int(path_id)
                for pair in set(candidate_info.get("_candidate_pairs", set()) or set())
                for path_id in pair
            })
            if candidate_path_ids:
                current, original = self._current_and_original_coords_for_statistics(candidate_path_ids)
            else:
                current, original = {}, {}
            stats = self._jaccard_filtered_pairwise_path_compactness(
                current,
                original,
                cluster_residue_sets,
                float(jaccard_threshold),
                cluster_atom_sets,
                float(atom_jaccard_threshold),
                precomputed_candidates=candidate_info,
            )
            current_mean = stats.get("current_mean")
            original_mean = stats.get("original_mean")
            current_median = stats.get("current_median")
            original_median = stats.get("original_median")
            delta_pair_mean = stats.get("distance_delta_mean")
            abs_delta_pair_mean = stats.get("abs_distance_delta_mean")

            pair_count = int(stats.get("pair_count", 0) or 0)
            if pair_count > 0:
                testable_clusters.append(int(cluster_id))
            atom_pair_count = int(stats.get("atom_pair_count", 0) or 0)
            atom_jaccard_mean = stats.get("atom_jaccard_mean")
            if atom_jaccard_mean is not None and atom_pair_count > 0:
                atom_jaccard_weighted_total += float(atom_jaccard_mean) * atom_pair_count
                atom_jaccard_weighted_count += atom_pair_count
            if current_mean is not None and pair_count > 0:
                current_weighted_total += float(current_mean) * pair_count
                current_weighted_count += pair_count
            if original_mean is not None and pair_count > 0:
                original_weighted_total += float(original_mean) * pair_count
                original_weighted_count += pair_count
            if delta_pair_mean is not None and pair_count > 0:
                delta_weighted_total += float(delta_pair_mean) * pair_count
                delta_weighted_count += pair_count
            if abs_delta_pair_mean is not None and pair_count > 0:
                abs_delta_weighted_total += float(abs_delta_pair_mean) * pair_count
                abs_delta_weighted_count += pair_count

            rows.append({
                "Cluster": str(cluster_id),
                "Path Count": str(len(cluster_path_ids)),
                "Residue Path Count": str(residue_path_count),
                "Candidate Path Count": str(int(stats.get("path_count", 0) or 0)),
                "Can Test": "Yes" if pair_count > 0 else "No",
                "Pair Count": str(pair_count),
                "Residue Candidate Pairs": str(int(stats.get("residue_candidate_pair_count", 0) or 0)),
                "Atom Candidate Pairs": str(int(stats.get("atom_candidate_pair_count", 0) or 0)),
                "Residue Shared Pairs": str(int(stats.get("residue_shared_pair_count", 0) or 0)),
                "Atom Shared Pairs": str(int(stats.get("atom_shared_pair_count", 0) or 0)),
                "Skipped Pair Count": str(int(stats.get("skipped_pair_count", 0) or 0)),
                "Residue Skipped Pairs": str(int(stats.get("residue_skipped_pair_count", 0) or 0)),
                "Atom Skipped Pairs": str(int(stats.get("atom_skipped_pair_count", 0) or 0)),
                "Missing Atom Pairs": str(int(stats.get("missing_atom_pair_count", 0) or 0)),
                "Total Pair Count": str(int(stats.get("total_pair_count", 0) or 0)),
                "Mean Residue Jaccard": self._format_stat_float(stats.get("jaccard_mean")) if stats.get("jaccard_mean") is not None else "",
                "Min Residue Jaccard": self._format_stat_float(stats.get("jaccard_min")) if stats.get("jaccard_min") is not None else "",
                "Max Residue Jaccard": self._format_stat_float(stats.get("jaccard_max")) if stats.get("jaccard_max") is not None else "",
                "Atom Path Count": str(sum(1 for path_id in cluster_path_ids if int(path_id) in atom_sets_by_id)),
                "Atom Pair Count": str(atom_pair_count),
                "Mean Atom Jaccard": self._format_stat_float(atom_jaccard_mean) if atom_jaccard_mean is not None else "",
                "Median Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_median")) if stats.get("atom_jaccard_median") is not None else "",
                "Min Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_min")) if stats.get("atom_jaccard_min") is not None else "",
                "Max Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_max")) if stats.get("atom_jaccard_max") is not None else "",
                "Std Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_std")) if stats.get("atom_jaccard_std") is not None else "",
                "Topo-Spatial Gap Mean": self._format_stat_float(delta_pair_mean) if delta_pair_mean is not None else "",
                "Topo-Spatial Gap Median": self._format_stat_float(stats.get("distance_delta_median")) if stats.get("distance_delta_median") is not None else "",
                "Abs Gap Mean": self._format_stat_float(abs_delta_pair_mean) if abs_delta_pair_mean is not None else "",
                "Abs Gap Median": self._format_stat_float(stats.get("abs_distance_delta_median")) if stats.get("abs_distance_delta_median") is not None else "",
                "Gap Min": self._format_stat_float(stats.get("distance_delta_min")) if stats.get("distance_delta_min") is not None else "",
                "Gap Max": self._format_stat_float(stats.get("distance_delta_max")) if stats.get("distance_delta_max") is not None else "",
                "Gap Std": self._format_stat_float(stats.get("distance_delta_std")) if stats.get("distance_delta_std") is not None else "",
                "Topo Greater Pairs": str(int(stats.get("increased_pair_count", 0) or 0)),
                "Topo Smaller Pairs": str(int(stats.get("decreased_pair_count", 0) or 0)),
                "Gap Stable Pairs": str(int(stats.get("near_zero_pair_count", 0) or 0)),
                "Topo Pair Dist Mean": self._format_stat_float(current_mean) if current_mean is not None else "",
                "Spatial Pair Dist Mean": self._format_stat_float(original_mean) if original_mean is not None else "",
                "Topo Pair Dist Median": self._format_stat_float(current_median) if current_median is not None else "",
                "Spatial Pair Dist Median": self._format_stat_float(original_median) if original_median is not None else "",
                "Topo Pair Dist Min": self._format_stat_float(stats.get("current_min")) if stats.get("current_min") is not None else "",
                "Spatial Pair Dist Min": self._format_stat_float(stats.get("original_min")) if stats.get("original_min") is not None else "",
                "Topo Pair Dist Max": self._format_stat_float(stats.get("current_max")) if stats.get("current_max") is not None else "",
                "Spatial Pair Dist Max": self._format_stat_float(stats.get("original_max")) if stats.get("original_max") is not None else "",
                "Topo Pair Dist Std": self._format_stat_float(stats.get("current_std")) if stats.get("current_std") is not None else "",
                "Spatial Pair Dist Std": self._format_stat_float(stats.get("original_std")) if stats.get("original_std") is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        current_mean = (current_weighted_total / current_weighted_count) if current_weighted_count > 0 else float("nan")
        original_mean = (original_weighted_total / original_weighted_count) if original_weighted_count > 0 else float("nan")
        delta_mean = (delta_weighted_total / delta_weighted_count) if delta_weighted_count > 0 else float("nan")
        abs_delta_mean = (
            abs_delta_weighted_total / abs_delta_weighted_count
            if abs_delta_weighted_count > 0
            else float("nan")
        )
        atom_jaccard_mean = (
            atom_jaccard_weighted_total / atom_jaccard_weighted_count
            if atom_jaccard_weighted_count > 0
            else float("nan")
        )
        if testable_clusters:
            visible_testable_clusters = ", ".join(str(cluster_id) for cluster_id in testable_clusters[:50])
            if len(testable_clusters) > 50:
                visible_testable_clusters += f", ... (+{len(testable_clusters) - 50} more)"
        else:
            visible_testable_clusters = "none"
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster compactness for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"residue Jaccard > {float(jaccard_threshold):.3f}; "
                f"atom Jaccard > {float(atom_jaccard_threshold):.3f}; "
                f"{len(rows):,} clusters; "
                f"{len(testable_clusters):,} can be tested ({visible_testable_clusters}); "
                f"weighted atom Jaccard {self._format_stat_float(atom_jaccard_mean)}; "
                f"weighted topo pair distance {self._format_stat_float(current_mean)}; "
                f"weighted spatial pair distance {self._format_stat_float(original_mean)}; "
                f"weighted topo-spatial gap {self._format_stat_float(delta_mean)}; "
                f"weighted abs gap {self._format_stat_float(abs_delta_mean)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster compactness: {dataset_label} ({len(rows)} clusters, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _compute_dataset_cluster_residue_similarity(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection

        path_ids: list[int] = []
        for cluster_id in target_clusters:
            path_ids.extend(
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            )
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        profiles = self.db.get_path_profiles(path_ids)
        residue_sets_by_id: dict[int, set[int]] = {}
        for profile in profiles:
            path_id = int(profile.get("pathIndex", 0) or 0)
            if path_id <= 0:
                continue
            residue_set = profile_residue_set(profile)
            if residue_set:
                residue_sets_by_id[int(path_id)] = set(residue_set)
        if not residue_sets_by_id:
            QMessageBox.information(self, "Statistics", "No residue composition data were available for the selected clusters.")
            return

        headers = [
            "Cluster",
            "Path Count",
            "Residue Path Count",
            "Pair Count",
            "Unique Residue Count",
            "Core Residue Count",
            "Mean Similarity",
            "Median Similarity",
            "Min Similarity",
            "Max Similarity",
            "Std Similarity",
        ]
        rows: list[dict] = []
        weighted_total = 0.0
        weighted_count = 0
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            }
            stats = self._pairwise_residue_similarity_stats({
                int(path_id): residue_set
                for path_id, residue_set in residue_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            })
            pair_count = int(stats.get("pair_count", 0) or 0)
            mean_similarity = stats.get("mean")
            if mean_similarity is not None and pair_count > 0:
                weighted_total += float(mean_similarity) * pair_count
                weighted_count += pair_count
            rows.append({
                "Cluster": str(cluster_id),
                "Path Count": str(len(cluster_path_ids)),
                "Residue Path Count": str(int(stats.get("path_count", 0) or 0)),
                "Pair Count": str(pair_count),
                "Unique Residue Count": str(int(stats.get("unique_residue_count", 0) or 0)),
                "Core Residue Count": str(int(stats.get("core_residue_count", 0) or 0)),
                "Mean Similarity": self._format_stat_float(mean_similarity) if mean_similarity is not None else "",
                "Median Similarity": self._format_stat_float(stats.get("median")) if stats.get("median") is not None else "",
                "Min Similarity": self._format_stat_float(stats.get("min")) if stats.get("min") is not None else "",
                "Max Similarity": self._format_stat_float(stats.get("max")) if stats.get("max") is not None else "",
                "Std Similarity": self._format_stat_float(stats.get("std")) if stats.get("std") is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        weighted_mean = (weighted_total / weighted_count) if weighted_count > 0 else float("nan")
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster residue similarity for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"{len(rows):,} clusters; "
                f"weighted mean similarity {self._format_stat_float(weighted_mean)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster residue similarity: {dataset_label} ({len(rows)} clusters, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _export_statistics_csv(self):
        headers = list(getattr(self, "_statistics_last_headers", []) or [])
        rows = list(getattr(self, "_statistics_last_rows", []) or [])
        if not headers or not rows:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Statistics CSV",
            "path_statistics.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow({header: row.get(header, "") for header in headers})
        except OSError as exc:
            QMessageBox.warning(self, "Export Statistics CSV", f"Failed to export CSV:\n{exc}")
            return
        self._statusbar.showMessage(f"Statistics exported: {path}", 4000)

    @staticmethod
    def _resample_path_coords(coords: np.ndarray, sample_count: int = 64) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[:, :3]
        if arr.shape[0] <= sample_count:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, int(sample_count), dtype=np.int64)
        return arr[idx]

    @staticmethod
    def _resample_path_coords_interpolated(coords: np.ndarray, sample_count: int = 64) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[:, :3]
        if int(sample_count) <= 1:
            return arr[:1].copy()

        segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 1e-8:
            return np.repeat(arr[:1], int(sample_count), axis=0)

        targets = np.linspace(0.0, total_length, int(sample_count), dtype=np.float64)
        result = np.empty((int(sample_count), 3), dtype=np.float64)
        for axis in range(3):
            result[:, axis] = np.interp(targets, cumulative, arr[:, axis])
        return result

    def _path_change_signature(self, path_ids: set[int]) -> tuple:
        return (
            tuple(sorted(int(path_id) for path_id in path_ids)),
            self.state.frame_min,
            self.state.frame_max,
            self._path_coordinate_mode,
        )

    def _path_change_distance_metric(self, path_ids: set[int]) -> dict[int, float]:
        path_ids = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        signature = self._path_change_signature(path_ids)
        if signature == self._path_change_metric_cache_signature:
            return dict(self._path_change_metric_cache)
        if not path_ids:
            return {}

        loaded_coords = dict(getattr(self._viewer, "_bg_data_dict", {}) or {})
        if self._path_coordinate_mode == "current":
            current_by_id = {
                int(path_id): coords
                for path_id, coords in loaded_coords.items()
                if int(path_id) in path_ids
            }
            missing_current = path_ids - set(current_by_id)
            if missing_current:
                current_by_id.update(self.db.get_render_coords(list(missing_current)))
            original_by_id = self.db.get_original_render_coords(list(path_ids))
        else:
            original_by_id = {
                int(path_id): coords
                for path_id, coords in loaded_coords.items()
                if int(path_id) in path_ids
            }
            missing_original = path_ids - set(original_by_id)
            if missing_original:
                original_by_id.update(self.db.get_original_render_coords(list(missing_original)))
            current_by_id = self.db.get_render_coords(list(path_ids))
        metrics: dict[int, float] = {}
        for path_id in sorted(path_ids):
            current = current_by_id.get(int(path_id))
            original = original_by_id.get(int(path_id))
            if current is None or original is None:
                continue
            current_sample = self._resample_path_coords(current)
            original_sample = self._resample_path_coords(original, sample_count=len(current_sample) or 64)
            count = min(len(current_sample), len(original_sample))
            if count <= 0:
                continue
            diff = current_sample[:count] - original_sample[:count]
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            if distances.size:
                metrics[int(path_id)] = float(np.mean(distances))

        self._path_change_metric_cache_signature = signature
        self._path_change_metric_cache = dict(metrics)
        return metrics

    def _exit_delta_signature(self, path_ids: set[int]) -> tuple:
        return (
            tuple(sorted(int(path_id) for path_id in path_ids)),
            self.state.frame_min,
            self.state.frame_max,
        )

    def _exit_cluster_distance_delta_metric(self, path_ids: set[int]) -> dict[int, float]:
        path_ids = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        signature = self._exit_delta_signature(path_ids)
        if signature == self._exit_delta_metric_cache_signature:
            return dict(self._exit_delta_metric_cache)
        if not path_ids or not hasattr(self.db, "get_path_exit_points"):
            return {}

        endpoints = self.db.get_path_exit_points(list(path_ids))
        groups: dict[tuple[str, int], list[int]] = {}
        for path_id, payload in endpoints.items():
            dataset_key = str(
                payload.get("dataset_key")
                or getattr(self.db, "get_dataset_key_for_path_id", lambda _pid: "")(int(path_id))
                or ""
            )
            cluster_id = int(payload.get("cluster_id", 0) or 0)
            groups.setdefault((dataset_key, cluster_id), []).append(int(path_id))

        metrics: dict[int, float] = {}
        for _group_key, group_path_ids in groups.items():
            valid_path_ids = [
                int(path_id)
                for path_id in group_path_ids
                if int(path_id) in endpoints
            ]
            if not valid_path_ids:
                continue
            current_points = np.vstack([
                np.asarray(endpoints[path_id]["current_exit"], dtype=np.float64)
                for path_id in valid_path_ids
            ])
            original_points = np.vstack([
                np.asarray(endpoints[path_id]["original_exit"], dtype=np.float64)
                for path_id in valid_path_ids
            ])
            current_center = np.mean(current_points, axis=0)
            original_center = np.mean(original_points, axis=0)
            for path_id in valid_path_ids:
                current_exit = np.asarray(endpoints[path_id]["current_exit"], dtype=np.float64)
                original_exit = np.asarray(endpoints[path_id]["original_exit"], dtype=np.float64)
                current_distance = float(np.linalg.norm(current_exit - current_center))
                original_distance = float(np.linalg.norm(original_exit - original_center))
                metrics[int(path_id)] = abs(current_distance - original_distance)

        self._exit_delta_metric_cache_signature = signature
        self._exit_delta_metric_cache = dict(metrics)
        return metrics

    @staticmethod
    def _format_path_delta_threshold(value: float) -> str:
        value = float(value)
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def _base_display_excluded_path_ids(self) -> set[int]:
        return (
            self.state.get_excluded_path_ids()
            | self._hidden_dataset_path_ids()
        )

    def _sync_path_delta_filter_controls(self, metrics: dict[int, float] | None = None):
        slider = getattr(self, "_path_delta_filter_slider", None)
        label = getattr(self, "_path_delta_filter_label", None)
        if slider is None or label is None:
            return
        metrics = metrics or {}
        active_label = "Exit Δ" if self._exit_delta_overlay_active else "Path Δ"
        overlay_active = bool(self._path_change_overlay_active or self._exit_delta_overlay_active)
        enabled = bool(overlay_active and metrics)
        slider.setEnabled(enabled)
        if not enabled or slider.value() <= 0:
            label.setText(f"{active_label} off" if overlay_active else "Δ off")
            label.setToolTip(f"{active_label} exclusion threshold" if overlay_active else "Active Δ exclusion threshold")
            return
        threshold = float(getattr(self, "_path_delta_filter_threshold", 0.0) or 0.0)
        label.setText(f"{active_label}≤{self._format_path_delta_threshold(threshold)}")
        label.setToolTip(
            f"Excluding paths with {active_label} <= {self._format_path_delta_threshold(threshold)}"
        )

    def _clear_path_delta_exclusion(self, *, reset_slider: bool = False, refresh: bool = True):
        changed = bool(getattr(self, "_path_delta_excluded_path_ids", set()))
        self._path_delta_excluded_path_ids.clear()
        self._path_delta_filter_threshold = 0.0
        slider = getattr(self, "_path_delta_filter_slider", None)
        if reset_slider and slider is not None and slider.value() != 0:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self._sync_path_delta_filter_controls({})
        if changed and refresh:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()

    def _apply_path_delta_exclusion(self, metrics: dict[int, float]) -> bool:
        slider = getattr(self, "_path_delta_filter_slider", None)
        overlay_active = bool(self._path_change_overlay_active or self._exit_delta_overlay_active)
        if slider is None or not overlay_active or not metrics or slider.value() <= 0:
            previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
            self._path_delta_excluded_path_ids.clear()
            self._path_delta_filter_threshold = 0.0
            self._sync_path_delta_filter_controls(metrics if overlay_active else {})
            return bool(previous)

        finite_values = [
            float(value)
            for value in metrics.values()
            if np.isfinite(float(value))
        ]
        if not finite_values:
            previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
            self._path_delta_excluded_path_ids.clear()
            self._path_delta_filter_threshold = 0.0
            self._sync_path_delta_filter_controls({})
            return bool(previous)

        max_delta = max(finite_values)
        threshold = max_delta * (float(slider.value()) / 100.0)
        self._path_delta_filter_threshold = threshold
        excluded = {
            int(path_id)
            for path_id, value in metrics.items()
            if np.isfinite(float(value)) and float(value) <= threshold
        }
        previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
        self._path_delta_excluded_path_ids = excluded
        self._sync_path_delta_filter_controls(metrics)
        return previous != excluded

    def _on_path_delta_filter_changed(self, _value: int):
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
            return
        if self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
            return
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=False, refresh=True)
            return

    def _apply_path_change_overlay(self):
        visible_ids = set(getattr(self._viewer, "_bg_data_dict", {}).keys())
        visible_ids -= self._base_display_excluded_path_ids()
        metrics = self._path_change_distance_metric(visible_ids)
        delta_exclusion_changed = self._apply_path_delta_exclusion(metrics)
        self._viewer.set_path_change_metric_map(
            metrics,
            self._display_excluded_path_ids(),
            cmap="Reds",
        )
        if delta_exclusion_changed:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
        self._viewer.set_bg_visible(True)
        self._viewer.set_dataset_visible(False)
        if metrics:
            max_distance = max(metrics.values())
            hidden_count = len(getattr(self, "_path_delta_excluded_path_ids", set()))
            hidden_suffix = f", Δ-filter hidden {hidden_count}" if hidden_count else ""
            self._statusbar.showMessage(
                f"Path change distance overlay: {len(metrics)} paths, max mean distance {max_distance:.3f}{hidden_suffix}",
                4000,
            )
        else:
            self._statusbar.showMessage("No matching original/current path coordinates were available", 4000)

    def _toggle_path_change_overlay(self, checked: bool):
        self._path_change_overlay_active = bool(checked)
        if self._path_change_overlay_active:
            exit_btn = getattr(self, "_exit_delta_btn", None)
            if exit_btn is not None and exit_btn.isChecked():
                exit_btn.blockSignals(True)
                exit_btn.setChecked(False)
                exit_btn.blockSignals(False)
            self._exit_delta_overlay_active = False
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
            if not self._exit_delta_overlay_active:
                self._viewer.set_path_change_metric_map(None, self._display_excluded_path_ids())
                self._last_3d_overlay_signature = None
                self._refresh_3d_overlays()
                self._statusbar.showMessage("Path change distance overlay disabled", 2500)
            return
        self._apply_path_change_overlay()

    def _apply_exit_delta_overlay(self):
        visible_ids = set(getattr(self._viewer, "_bg_data_dict", {}).keys())
        visible_ids -= self._base_display_excluded_path_ids()
        metrics = self._exit_cluster_distance_delta_metric(visible_ids)
        delta_exclusion_changed = self._apply_path_delta_exclusion(metrics)
        self._viewer.set_path_change_metric_map(
            metrics,
            self._display_excluded_path_ids(),
            cmap="bwr",
            clim_percentile=95.0,
        )
        if delta_exclusion_changed:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
        self._viewer.set_bg_visible(True)
        self._viewer.set_dataset_visible(False)
        if metrics:
            max_delta = max(metrics.values())
            hidden_count = len(getattr(self, "_path_delta_excluded_path_ids", set()))
            hidden_suffix = f", Exit Δ-filter hidden {hidden_count}" if hidden_count else ""
            self._statusbar.showMessage(
                f"Exit Δ overlay: {len(metrics)} paths, max cluster-distance difference {max_delta:.3f}{hidden_suffix}",
                4000,
            )
        else:
            self._statusbar.showMessage("No clustered current/original exit coordinates were available", 4000)

    def _toggle_exit_delta_overlay(self, checked: bool):
        self._exit_delta_overlay_active = bool(checked)
        if self._exit_delta_overlay_active:
            path_btn = getattr(self, "_path_change_btn", None)
            if path_btn is not None and path_btn.isChecked():
                path_btn.blockSignals(True)
                path_btn.setChecked(False)
                path_btn.blockSignals(False)
            self._path_change_overlay_active = False
            self._apply_exit_delta_overlay()
            return
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
            self._viewer.set_path_change_metric_map(None, self._display_excluded_path_ids())
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
            self._statusbar.showMessage("Exit Δ overlay disabled", 2500)

    def _visible_category_payload(self):
        visible = {}
        for name in self.state.visible_categories:
            if name in self.state.categories:
                path_ids = self._filter_display_path_set(self.state.categories[name])
                color = self.state.category_colors.get(name, "#00CC96")
                if path_ids:
                    visible[name] = (path_ids, color)
        return visible

    def _visible_dataset_payload(self, *, exclude_effective: bool = False):
        visible = {}
        excluded = self._display_excluded_path_ids()
        # Effective/current paths are rendered by the dedicated bright layer.
        # Remove them from the dim dataset layer so a selected path is not
        # drawn twice (which makes it look displaced or thicker during pick).
        effective_ids = (
            set(getattr(self.state, "effective_path_ids", set()) or set()) - excluded
            if exclude_effective else set()
        )
        for dataset in self.db.list_datasets():
            key = dataset["key"]
            if key not in self.state.visible_datasets:
                continue
            if self.state.dataset_cluster_display_key == key and hasattr(self.db, "get_dataset_cluster_path_groups"):
                cluster_groups = self.db.get_dataset_cluster_path_groups(
                    key,
                    frame_min=self.state.frame_min,
                    frame_max=self.state.frame_max,
                )
                for cluster_id, cluster_path_ids in cluster_groups.items():
                    path_ids = set(cluster_path_ids) - excluded - effective_ids
                    if path_ids:
                        visible[f"dataset:{key}:cluster:{cluster_id}"] = (
                            path_ids,
                            self.state.get_auto_color(int(cluster_id)),
                        )
                continue
            path_ids = self._dataset_path_ids(key) - excluded - effective_ids
            if path_ids:
                visible[f"dataset:{key}"] = (
                    path_ids,
                    self.state.dataset_colors.get(key, self.state.get_auto_color(0)),
                )
        return visible

    def _visible_dataset_path_ids(self) -> set[int]:
        visible_ids = set()
        for path_ids, _color in self._visible_dataset_payload().values():
            visible_ids.update(path_ids)
        return visible_ids

    def _visible_dataset_path_color_map(self, path_ids=None) -> dict[int, str]:
        color_map: dict[int, str] = {}
        requested = None if path_ids is None else {int(path_id) for path_id in path_ids}
        for dataset_path_ids, color in self._visible_dataset_payload().values():
            selected_ids = dataset_path_ids if requested is None else requested.intersection(dataset_path_ids)
            for path_id in selected_ids:
                color_map[int(path_id)] = str(color)
        return color_map

    def _chart_path_color_map(self, profiles) -> dict[int, str]:
        color_map: dict[int, str] = {}
        for profile in profiles or []:
            path_id = int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
            if path_id <= 0:
                continue
            dataset_key = getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(path_id)
            if not dataset_key:
                color_map[path_id] = "#4169E1"
                continue
            if self.state.dataset_cluster_display_key == dataset_key:
                cluster_id = int(profile.get("cluster_id", 0) or 0)
                color_map[path_id] = self.state.get_auto_color(cluster_id)
            else:
                color_map[path_id] = self.state.dataset_colors.get(
                    dataset_key,
                    self.state.get_auto_color(0),
                )
        return color_map

    def _visible_cluster_label_payload(self) -> list[dict]:
        dataset_key = str(self.state.dataset_cluster_label_key or "")
        if not dataset_key or not hasattr(self.db, "get_dataset_cluster_path_groups"):
            return []
        excluded = self._display_excluded_path_ids()
        cluster_groups = self.db.get_dataset_cluster_path_groups(
            dataset_key,
            frame_min=self.state.frame_min,
            frame_max=self.state.frame_max,
        )
        labels: list[dict] = []
        for cluster_id, cluster_path_ids in cluster_groups.items():
            exit_points = []
            for path_id in cluster_path_ids:
                if int(path_id) in excluded:
                    continue
                coords = self._viewer._bg_data_dict.get(int(path_id))
                if coords is None or len(coords) == 0:
                    continue
                exit_points.append(np.asarray(coords[-1], dtype=np.float64))
            if not exit_points:
                continue
            mean_point = np.mean(np.vstack(exit_points), axis=0)
            labels.append(
                {
                    "text": f"C{int(cluster_id)}",
                    "position": tuple(float(value) for value in mean_point[:3]),
                    "color": self.state.get_auto_color(int(cluster_id)),
                }
            )
        return labels

    def _dataset_path_ids(self, key: str) -> set[int]:
        cache_key = (key, self.state.frame_min, self.state.frame_max)
        if cache_key not in self._dataset_path_cache:
            self._dataset_path_cache[cache_key] = set(self.db.get_dataset_path_ids(
                key,
                frame_min=self.state.frame_min,
                frame_max=self.state.frame_max,
            ))
        return set(self._dataset_path_cache[cache_key])

    def _hidden_dataset_path_ids(self) -> set[int]:
        hidden = set()
        for dataset in self.db.list_datasets():
            key = dataset["key"]
            if key not in self.state.visible_datasets:
                hidden.update(self._dataset_path_ids(key))
        return hidden

    def _display_excluded_path_ids(self) -> set[int]:
        return self._base_display_excluded_path_ids() | set(getattr(self, "_path_delta_excluded_path_ids", set()))

    def _filter_hidden_dataset_path_set(self, path_ids) -> set[int]:
        return set(path_ids) - self._hidden_dataset_path_ids()

    def _filter_hidden_dataset_paths(self, path_ids):
        hidden = self._hidden_dataset_path_ids()
        return [int(pid) for pid in path_ids if int(pid) not in hidden]

    def _filter_display_path_set(self, path_ids) -> set[int]:
        return set(path_ids) - self._display_excluded_path_ids()

    def _filter_display_paths(self, path_ids):
        excluded = self._display_excluded_path_ids()
        return [int(pid) for pid in path_ids if int(pid) not in excluded]

    def _visible_effective_path_ids(self) -> set[int]:
        return self._filter_display_path_set(self.state.effective_path_ids)

    def _path_panel_selected_path_ids(self) -> set[int]:
        panel = getattr(self, "_path_panel", None)
        if panel is None or not hasattr(panel, "selected_path_ids"):
            return set()
        try:
            return {
                int(path_id)
                for path_id in panel.selected_path_ids()
                if int(path_id) > 0
            }
        except Exception:
            return set()

    def _residue_combination_context_path_ids(
        self,
        *,
        prefer_explicit_selection: bool = False,
    ) -> set[int]:
        selected = self._path_panel_selected_path_ids()
        effective = {
            int(path_id)
            for path_id in getattr(self.state, "effective_path_ids", set())
            if int(path_id) > 0
        }
        preview = {
            int(path_id)
            for path_id in (getattr(self, "_last_preview_path_ids", []) or [])
            if int(path_id) > 0
        }
        # Reload/manual context refresh must not be narrowed by a table-row click.
        if prefer_explicit_selection and selected:
            return selected
        if effective:
            return effective
        if prefer_explicit_selection and preview:
            return preview
        return set()

    def _get_display_path_coords(self, path_ids):
        if self._path_coordinate_mode == "original":
            return self.db.get_original_render_coords(path_ids)
        return self.db.get_render_coords(path_ids)

    def _payload_signature(self, payload: dict) -> tuple:
        items = []
        for key, value in (payload or {}).items():
            path_ids, color = value
            items.append((str(key), tuple(sorted(int(path_id) for path_id in (path_ids or ()))), str(color)))
        return tuple(sorted(items))

    def _cluster_label_signature(self, labels: list[dict]) -> tuple:
        result = []
        for entry in labels or []:
            position = entry.get("position", ())
            try:
                pos = tuple(round(float(value), 4) for value in position)
            except Exception:
                pos = ()
            result.append((str(entry.get("text", "") or ""), pos, str(entry.get("color", "") or "")))
        return tuple(result)

    def _entry_exit_signature_values(self) -> tuple:
        display_ids = self._get_entry_exit_display_ids()
        return (
            tuple(sorted(int(path_id) for path_id in self.state.effective_path_ids)),
            None if display_ids is None else tuple(sorted(int(path_id) for path_id in display_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in self._visible_dataset_path_color_map().items())),
        )

    def _render_effective_paths_if_needed(self, path_ids=None, color_map=None) -> None:
        visible_ids = set(path_ids) if path_ids is not None else self._visible_effective_path_ids()
        source_color_map = (
            dict(color_map)
            if color_map is not None
            else self._visible_dataset_path_color_map(visible_ids)
        )
        # The dataset color map can contain every loaded path.  Effective-path
        # rendering only needs colors for the selected subset; shrinking it here
        # avoids sorting and hashing the whole database on every click.
        visible_color_map = {
            int(path_id): str(source_color_map[int(path_id)])
            for path_id in visible_ids
            if int(path_id) in source_color_map
        }
        signature = (
            tuple(sorted(int(path_id) for path_id in visible_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in visible_color_map.items())),
        )
        if signature == self._last_effective_render_signature:
            return
        self._last_effective_render_signature = signature
        self._viewer.render_effective_paths(visible_ids, color_map=visible_color_map)

    def _refresh_3d_overlays(self):
        excluded = self._display_excluded_path_ids()
        dataset_payload = self._visible_dataset_payload(exclude_effective=True)
        cluster_payload = self._visible_cluster_label_payload()
        category_payload = self._visible_category_payload()
        effective_ids = self._visible_effective_path_ids()
        effective_color_map = self._visible_dataset_path_color_map(effective_ids)
        visible_preview_ids = (
            tuple(self._filter_display_paths(self._last_preview_path_ids))
            if self._last_preview_path_ids else ()
        )
        highlighted_preview_ids = tuple(sorted(self._filter_display_path_set(self._last_highlighted_preview_ids)))
        effective_id_set = set(int(value) for value in effective_ids)
        preview_only_signature = tuple(
            path_id for path_id in visible_preview_ids
            if int(path_id) not in effective_id_set
        )
        signature = (
            tuple(sorted(int(path_id) for path_id in excluded)),
            self._payload_signature(dataset_payload),
            self._cluster_label_signature(cluster_payload),
            self._payload_signature(category_payload),
            tuple(sorted(int(path_id) for path_id in effective_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in effective_color_map.items())),
            preview_only_signature,
            tuple(sorted(set(highlighted_preview_ids).intersection(preview_only_signature))),
            tuple(sorted((int(path_id), str(color)) for path_id, color in (self._last_preview_color_map or {}).items())),
            bool(self.state.focus_mode),
            bool(self.state.show_entry_exit_points),
            self._entry_exit_signature_values(),
        )
        if signature == self._last_3d_overlay_signature:
            return
        self._last_3d_overlay_signature = signature
        if hasattr(self._viewer, "begin_render_batch"):
            self._viewer.begin_render_batch()
        try:
            self._viewer.set_bg_excluded_paths(set(excluded))
            self._viewer.render_dataset_paths(dataset_payload)
            self._viewer.render_cluster_labels(cluster_payload)
            self._viewer.render_category_paths(category_payload)
            self._last_effective_render_signature = None
            self._render_effective_paths_if_needed(effective_ids, effective_color_map)

            # A selected/current path is already rendered by the effective
            # path layer. Do not add it again as a preview actor, otherwise
            # the same line is drawn twice and appears offset/thicker.
            preview_only_ids = tuple(
                path_id for path_id in visible_preview_ids
                if int(path_id) not in effective_id_set
            )
            if preview_only_ids:
                coords = self._get_display_path_coords(list(preview_only_ids))
                self._viewer.render_matched_paths(
                    coords,
                    color_map=self._last_preview_color_map or None,
                    highlighted=set(highlighted_preview_ids).intersection(preview_only_ids),
                )
            else:
                self._viewer.clear_matched()

            if self.state.focus_mode:
                self._viewer.set_bg_visible(False)
                self._viewer.set_dataset_visible(False)
            elif self._path_change_overlay_active or self._exit_delta_overlay_active:
                self._viewer.set_bg_visible(True)
                self._viewer.set_dataset_visible(False)
            else:
                self._viewer.set_bg_visible(True)
                self._viewer.set_dataset_visible(True)
            if self.state.show_entry_exit_points:
                self._last_entry_exit_signature = None
                self._refresh_entry_exit_points()
            self._render_dataset_compare_overlay()
        finally:
            if hasattr(self._viewer, "end_render_batch"):
                self._viewer.end_render_batch()

    def _on_viewer_coordinate_mode_changed(self, mode: str):
        mode = "original" if mode == "original" else "current"
        if self._path_coordinate_load_busy:
            self._viewer.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)
            return
        if self._path_coordinate_mode == mode:
            return
        self._path_coordinate_mode = mode
        self._invalidate_render_signatures()
        mode_label = "original paths" if mode == "original" else "current paths"
        self._statusbar.showMessage(f"3D coordinate mode switched to {mode_label}", 2500)
        self._reload_3d()

    # ─── Entry/Exit point display + selection mode ────────
    def _toggle_focus_mode(self, checked: bool):
        self.state.focus_mode = checked
        self._viewer.set_focus_mode(checked)
        self._viewer.set_bg_visible(not checked)
        self._viewer.set_dataset_visible(not checked)
        if checked:
            self._statusbar.showMessage("Focus mode: showing only selected paths", 3000)
        else:
            self._statusbar.showMessage("Focus mode off: showing all paths", 3000)
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()

    def _toggle_entry_exit_points(self, checked: bool):
        self.state.set_show_entry_exit_points(checked)

    def _toggle_entry_exit_scope(self, checked: bool):
        self.state.set_entry_exit_current_only(checked)

    def _toggle_exit_cluster_select(self, checked: bool):
        self._viewer.set_exit_cluster_select_mode(bool(checked))
        if checked:
            if not self._entry_exit_btn.isChecked():
                self._entry_exit_btn.setChecked(True)
            self._statusbar.showMessage(
                "Exit Cluster ON - click an exit point to select all connected exit spheres at current zoom",
                5000,
            )
        else:
            self._statusbar.showMessage("Exit Cluster OFF", 2500)

    def _get_entry_exit_display_ids(self):
        if not self.state.entry_exit_current_only:
            return None
        if self.state.effective_path_ids:
            return set(self.state.effective_path_ids)
        visible = self.state.get_visible_category_path_ids()
        return set(visible)

    def _refresh_entry_exit_scope_btn(self):
        current_only = self.state.entry_exit_current_only
        self._entry_exit_scope_btn.blockSignals(True)
        self._entry_exit_scope_btn.setChecked(current_only)
        self._entry_exit_scope_btn.setText("EE: Current" if current_only else "EE: All")
        self._entry_exit_scope_btn.setToolTip(
            "Showing entry/exit points for current paths only"
            if current_only else
            "Showing entry/exit points for all paths"
        )
        self._entry_exit_scope_btn.blockSignals(False)

    def _refresh_entry_exit_points(self):
        display_ids = self._get_entry_exit_display_ids()
        signature = self._entry_exit_signature_values()
        if signature == self._last_entry_exit_signature:
            return
        self._last_entry_exit_signature = signature
        self._viewer.render_entry_exit_points(
            show=True,
            effective_ids=self.state.effective_path_ids,
            display_ids=display_ids,
            path_color_map=self._visible_dataset_path_color_map(),
        )

    def _on_entry_exit_display_changed(self, show: bool):
        if show:
            self._refresh_entry_exit_points()
            counts = self._viewer._rendered_entry_exit_counts
            scope = "current" if self.state.entry_exit_current_only else "all"
            self._statusbar.showMessage(
                f"Entry/exit points ({scope}): {counts['entries']} entries + "
                f"{counts['exits']} exits",
                3000,
            )
        else:
            self._viewer.clear_entry_exit_points()
            self._statusbar.showMessage("Entry/exit points hidden", 3000)

    def _on_entry_exit_scope_changed(self, current_only: bool):
        self._refresh_entry_exit_scope_btn()
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
            counts = self._viewer._rendered_entry_exit_counts
            scope = "current" if current_only else "all"
            self._statusbar.showMessage(
                f"Entry/exit scope: {scope} "
                f"({counts['entries']} entries + {counts['exits']} exits)",
                3000,
            )

    def _toggle_selection_mode(self, checked: bool):
        mode = "entrance_exit" if checked else "path"
        self.state.set_selection_mode(mode)
        if checked:
            # Auto-show entry/exit points when switching to point select mode
            if not self._entry_exit_btn.isChecked():
                self._entry_exit_btn.setChecked(True)
            self._statusbar.showMessage(
                "Point Select ON — lasso selects entry/exit points (coarse filter)", 5000
            )
        else:
            self._statusbar.showMessage(
                "Path Select — lasso selects paths (fine filter)", 3000
            )

    def _on_exit_cluster_selected(self, path_ids):
        selected = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        if not selected:
            self._statusbar.showMessage("No connected exit points found at this zoom", 3000)
            return
        self.state.apply_lasso(selected, "replace")
        if self._legacy_path_panel_enabled:
            self._path_panel.refresh()
            self._path_panel.set_highlighted_paths(sorted(selected))
        self._last_highlighted_preview_ids = set(selected)
        self._status_selected.setText(f"Effective: {len(self.state.effective_path_ids):,} paths")
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
        self._statusbar.showMessage(
            f"Selected {len(selected)} paths from connected exit points",
            4000,
        )

    def _toggle_residue_positions(self, checked: bool):
        self.state.show_all_residues = checked
        if checked:
            positions = self._residue_positions_with_dataset_colors()
            if positions:
                self._viewer.set_residue_data(positions)
                self._viewer.render_residues(self.state.selected_residues)
                self._refresh_residue_labels()
                if self._legacy_residue_combination_enabled:
                    self._refresh_residue_combination_markers()
                self._statusbar.showMessage(
                    f"Showing {len(positions)} residues", 3000
                )
        else:
            self._viewer.clear_residues()
            self._statusbar.showMessage("Residue positions hidden", 3000)

    def _refresh_residue_3d(self):
        """Re-render residue spheres when selection changes (if toggle is on)."""
        if self.state.show_all_residues:
            self._viewer.render_residues(self.state.selected_residues)
            self._refresh_residue_labels()
        if self._legacy_residue_combination_enabled:
            self._refresh_residue_combination_markers()

    def _toggle_residue_labels(self, checked: bool):
        if not checked:
            self._viewer.clear_residue_labels()
            self._statusbar.showMessage("Residue labels hidden", 3000)
            return
        self._refresh_residue_labels()
        self._statusbar.showMessage("Residue labels shown for selected residues", 3000)

    def _refresh_residue_labels(self):
        if not getattr(self, "_residue_labels_btn", None):
            return
        if not self._residue_labels_btn.isChecked() or not self.state.show_all_residues:
            self._viewer.clear_residue_labels()
            return
        self._viewer.render_residue_labels(self.state.selected_residues)

    def _on_residue_3d_clicked(self, residue_id: int):
        """Residue sphere clicked in 3D → toggle selection in panel."""
        self.state.toggle_residue(residue_id)
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _on_residue_compare_selection_requested(self, dataset_key: str, local_residue_id: int):
        if not dataset_key or int(local_residue_id) <= 0:
            return
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        if dataset is None:
            return
        global_residue_id = dataset.residue_id_base + int(local_residue_id)
        self.state.toggle_residue(global_residue_id)
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _on_residue_compare_clear_requested(self):
        self.state.clear_residues()
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _sync_residue_compare_selection_state(self):
        if not self._legacy_residue_compare_enabled:
            return
        selected_keys: set[tuple[str, int]] = set()
        for residue_id in self.state.selected_residues:
            dataset, local_id = getattr(self.db, "_decode_residue_id", lambda _rid: (None, None))(int(residue_id))
            if dataset is None or local_id is None:
                continue
            selected_keys.add((str(dataset.key), int(local_id)))
        if getattr(self, "_residue_compare_panel", None):
            self._residue_compare_panel.set_selected_residues(selected_keys)

    def _prime_primary_residue_positions(self):
        positions = []
        primary_db = self._primary_analysis_db()
        if primary_db is not None:
            positions = primary_db.get_residue_positions()
        self._primary_residue_position_cache = {
            int(row["residue_id"]): np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            for row in positions
            if row.get("residue_id") is not None
        }

    def _residue_positions_with_dataset_colors(self) -> list[dict]:
        positions = self.db.get_residue_positions()
        for row in positions:
            dataset_key = str(row.get("dataset_key", "") or "")
            row["dataset_color"] = self.state.dataset_colors.get(dataset_key, "#4169E1")
        return positions

    def _dataset_residue_position_maps(self) -> dict[str, dict[int, tuple[float, float, float]]]:
        result: dict[str, dict[int, tuple[float, float, float]]] = {}
        for dataset_info in self.db.list_datasets() if hasattr(self.db, "list_datasets") else []:
            dataset_key = str(dataset_info.get("key", "") or "")
            binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
            dataset_db = binding.db if binding is not None else None
            if not dataset_key or dataset_db is None:
                continue
            result[dataset_key] = {
                int(row["residue_id"]): (float(row["x"]), float(row["y"]), float(row["z"]))
                for row in dataset_db.get_residue_positions()
                if row.get("residue_id") is not None
            }
        self._dataset_residue_position_cache = result
        return result

    def _dataset_residue_atom_position_maps(self) -> dict[str, dict[int, list[tuple[float, float, float]]]]:
        result: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
        protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
        aligned_atoms = list(protein_cache.get("atoms_aligned", []) or [])
        if not aligned_atoms:
            return result

        residue_atoms: dict[int, list[tuple[float, float, float]]] = {}
        for atom in aligned_atoms:
            try:
                residue_id = int(atom.get("res_seq", 0) or 0)
                position = np.asarray(atom.get("position"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            if residue_id <= 0:
                continue
            residue_atoms.setdefault(residue_id, []).append(
                (float(position[0]), float(position[1]), float(position[2]))
            )

        if not residue_atoms:
            return result

        for dataset_info in self.db.list_datasets() if hasattr(self.db, "list_datasets") else []:
            dataset_key = str(dataset_info.get("key", "") or "")
            if dataset_key:
                result[dataset_key] = {
                    int(residue_id): list(atom_positions)
                    for residue_id, atom_positions in residue_atoms.items()
                }
        return result

    def _encode_residue_ids_for_all_datasets(self, residue_ids) -> list[int]:
        local_ids = [int(rid) for rid in residue_ids if int(rid) > 0]
        if not local_ids or not hasattr(self.db, "list_datasets") or not hasattr(self.db, "_datasets_by_key"):
            return local_ids

        encoded_ids: list[int] = []
        for dataset_info in self.db.list_datasets():
            dataset_key = str(dataset_info.get("key") or "")
            dataset = self.db._datasets_by_key.get(dataset_key)
            if dataset is None:
                continue
            encoded_ids.extend(dataset.residue_id_base + local_id for local_id in local_ids)
        return encoded_ids or local_ids

    def _dataset_combination_matched_path_ids(self, dataset_key: str, residue_ids) -> list[int]:
        residue_tuple = tuple(sorted(int(rid) for rid in residue_ids if int(rid) > 0))
        if len(residue_tuple) < 2:
            return []
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        dataset_db = dataset_binding.db if dataset_binding is not None else None
        if dataset_binding is None or dataset_db is None:
            return []

        local_path_ids = dataset_db.get_path_ids()
        profiles = dataset_db.get_path_profiles(local_path_ids)
        matched_local_ids: list[int] = []
        for profile in profiles:
            residue_columns = (
                profile.get("res_1", []),
                profile.get("res_2", []),
                profile.get("res_3", []),
                profile.get("res_4", []),
            )
            point_count = max((len(column) for column in residue_columns), default=0)
            has_combination = False
            for idx in range(point_count):
                point_residue_ids = tuple(sorted(
                    {
                        int(column[idx])
                        for column in residue_columns
                        if idx < len(column) and int(column[idx]) > 0
                    }
                ))
                if point_residue_ids == residue_tuple:
                    has_combination = True
                    break
            if has_combination:
                local_path_id = int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
                if local_path_id > 0:
                    matched_local_ids.append(local_path_id)
        return [
            dataset_binding.path_id_base + int(local_path_id)
            for local_path_id in matched_local_ids
        ]

    def _build_residue_combination_markers(self) -> list[dict]:
        if not self._primary_residue_position_cache:
            self._prime_primary_residue_positions()
        if not self._dataset_residue_position_cache:
            self._dataset_residue_position_maps()

        property_key = self._residue_combination_panel.selected_property_key()
        property_label = self._residue_combination_panel.selected_property_label()
        compare_mode = self._residue_combination_panel.is_compare_mode()
        active_keys = self._residue_combination_panel.active_dataset_keys()
        marker_position_maps = [
            self._dataset_residue_position_cache.get(key, {})
            for key in reversed(active_keys)
            if self._dataset_residue_position_cache.get(key)
        ]
        marker_position_maps.append(self._primary_residue_position_cache)
        markers = []
        for row in self._residue_combination_panel.visible_rows():
            residue_ids = tuple(int(rid) for rid in row.get("residue_ids", ()) if int(rid) > 0)
            centroid = None
            for position_map in marker_position_maps:
                centroid = combination_centroid(residue_ids, position_map)
                if centroid is not None:
                    break
            if centroid is None:
                continue
            trend = combination_property_trend(row, property_key, compare_mode=compare_mode)
            value = combination_property_value(row, property_key, compare_mode=compare_mode)
            markers.append(
                {
                    "marker_key": "combo_" + "_".join(str(rid) for rid in residue_ids),
                    "residue_ids": residue_ids,
                    "residue_names": tuple(row.get("residue_names", ())),
                    "combination_label": row.get("combination_label", ""),
                    "affected_path_count": int(row.get("affected_path_count", row.get("metrics", {}).get("affected_path_count", {}).get("right", 0)) or 0),
                    "property_key": property_key,
                    "property_label": property_label,
                    "property_value": value,
                    "property_trend": trend,
                    "property_trend_label": {
                        "increase": "Increase",
                        "decrease": "Decrease",
                        "neutral": "Neutral" if compare_mode else "",
                    }.get(trend, ""),
                    "centroid": centroid,
                }
            )
        return markers

    def _refresh_residue_combination_current_paths(
        self,
        *,
        prefer_explicit_selection: bool = False,
    ):
        if not getattr(self, "_residue_combination_panel", None):
            return {}
        combo_keys: set[tuple[int, ...]] = set()
        rows_by_dataset: dict[str, list[dict]] = {}
        total_paths_by_dataset: dict[str, int] = {}
        profiles_by_dataset: dict[str, list[dict]] = {}
        panel = self._residue_combination_panel
        active_dataset_keys = set(panel.active_dataset_keys())
        context_path_ids = self._residue_combination_context_path_ids(
            prefer_explicit_selection=prefer_explicit_selection,
        )
        if context_path_ids and hasattr(self.db, "list_datasets") and hasattr(self.db, "get_local_path_ids_for_dataset"):
            for dataset_info in self.db.list_datasets():
                dataset_key = str(dataset_info.get("key") or "")
                if not dataset_key:
                    continue
                if active_dataset_keys and dataset_key not in active_dataset_keys:
                    continue
                local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                if not local_path_ids:
                    continue
                dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    continue
                profiles = dataset_db.get_path_profiles(local_path_ids)
                profiles_by_dataset[dataset_key] = list(profiles)
                rows = combination_rows_from_profiles(profiles)
                rows_by_dataset[dataset_key] = rows
                combo_keys.update(tuple(row.get("residue_ids", ())) for row in rows)
                total_paths_by_dataset[dataset_key] = len(set(int(path_id) for path_id in local_path_ids))
        elif context_path_ids:
            profiles = self.db.get_path_profiles(list(context_path_ids))
            primary_key = self.db.primary_dataset_key() if hasattr(self.db, "primary_dataset_key") else ""
            if primary_key:
                profiles_by_dataset[str(primary_key)] = list(profiles)
                rows = combination_rows_from_profiles(profiles)
                rows_by_dataset[str(primary_key)] = rows
                combo_keys = {tuple(row.get("residue_ids", ())) for row in rows}
                total_paths_by_dataset[str(primary_key)] = len(set(int(path_id) for path_id in context_path_ids))
        self._residue_combination_panel.set_current_path_combination_keys(combo_keys)
        self._residue_combination_panel.set_current_path_combination_rows(
            rows_by_dataset,
            total_paths_by_dataset,
        )
        return profiles_by_dataset

    def _refresh_residue_combination_current_bottlenecks(
        self,
        *,
        prefer_explicit_selection: bool = False,
        profiles_by_dataset_key: dict[str, list[dict]] | None = None,
    ):
        if not getattr(self, "_residue_combination_panel", None):
            return
        combo_keys: set[tuple[int, ...]] = set()
        rows_by_dataset: dict[str, list[dict]] = {}
        total_paths_by_dataset: dict[str, int] = {}
        cached_profiles_by_dataset = {
            str(dataset_key): list(profiles or [])
            for dataset_key, profiles in dict(profiles_by_dataset_key or {}).items()
        }
        panel = self._residue_combination_panel
        active_dataset_keys = set(panel.active_dataset_keys())
        context_path_ids = self._residue_combination_context_path_ids(
            prefer_explicit_selection=prefer_explicit_selection,
        )
        if context_path_ids and hasattr(self.db, "list_datasets") and hasattr(self.db, "get_local_path_ids_for_dataset"):
            for dataset_info in self.db.list_datasets():
                dataset_key = str(dataset_info.get("key") or "")
                if not dataset_key:
                    continue
                if active_dataset_keys and dataset_key not in active_dataset_keys:
                    continue
                local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                if not local_path_ids:
                    continue
                dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    continue
                cached_profiles = cached_profiles_by_dataset.get(dataset_key)
                profiles = (
                    list(cached_profiles)
                    if cached_profiles is not None
                    else dataset_db.get_path_profiles(local_path_ids)
                )
                rows = combination_rows_from_profiles(
                    profiles,
                    bottleneck_only=True,
                )
                rows_by_dataset[dataset_key] = rows
                combo_keys.update(tuple(row.get("residue_ids", ())) for row in rows)
                total_paths_by_dataset[dataset_key] = len(set(int(path_id) for path_id in local_path_ids))
        elif context_path_ids:
            primary_key = self.db.primary_dataset_key() if hasattr(self.db, "primary_dataset_key") else ""
            profiles = (
                list(cached_profiles_by_dataset.get(primary_key, []))
                if primary_key and primary_key in cached_profiles_by_dataset
                else self.db.get_path_profiles(list(context_path_ids))
            )
            if primary_key:
                rows = combination_rows_from_profiles(
                    profiles,
                    bottleneck_only=True,
                )
                rows_by_dataset[str(primary_key)] = rows
                combo_keys = {tuple(row.get("residue_ids", ())) for row in rows}
                total_paths_by_dataset[str(primary_key)] = len(set(int(path_id) for path_id in context_path_ids))
        self._residue_combination_panel.set_current_bottleneck_combination_keys(combo_keys)
        self._residue_combination_panel.set_current_bottleneck_combination_rows(
            rows_by_dataset,
            total_paths_by_dataset,
        )

    def _combination_present_frames_from_profiles(
        self,
        profiles: list[dict],
        *,
        max_size: int = 4,
        bottleneck_only: bool = False,
    ) -> dict[tuple[int, ...], set[int]]:
        from itertools import combinations

        result: dict[tuple[int, ...], set[int]] = {}
        for profile in profiles or ():
            try:
                frame_id = int(profile.get("frameId", profile.get("frame_id", 0)) or 0)
            except (TypeError, ValueError):
                continue
            if frame_id <= 0:
                continue
            residue_columns = (
                profile.get("res_1", []),
                profile.get("res_2", []),
                profile.get("res_3", []),
                profile.get("res_4", []),
            )
            radii = profile.get("radius", [])
            point_count = max((len(column) for column in residue_columns), default=0)
            allowed_indices: set[int] | None = None
            if bottleneck_only and len(radii) > 0:
                allowed_indices = bottleneck_point_indices(radii, point_count=point_count)
            for idx in range(point_count):
                if allowed_indices is not None and idx not in allowed_indices:
                    continue
                residue_ids = sorted(
                    {
                        int(column[idx])
                        for column in residue_columns
                        if idx < len(column) and int(column[idx] or 0) > 0
                    }
                )
                for size in range(2, min(int(max_size), len(residue_ids)) + 1):
                    for combo in combinations(residue_ids, size):
                        result.setdefault(tuple(int(item) for item in combo), set()).add(frame_id)
        return result

    def _profiles_for_dataset_path_scope(self, dataset_key: str, frame_ids: set[int] | None = None) -> list[dict]:
        dataset_key = str(dataset_key or "").strip()
        if not dataset_key:
            return []
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        dataset_db = dataset_binding.db if dataset_binding is not None else None
        if dataset_db is None:
            return []
        normalized_frames = sorted({int(frame) for frame in (frame_ids or set()) if int(frame) > 0})
        global_ids = []
        try:
            if hasattr(self.db, "get_dataset_path_ids"):
                if normalized_frames:
                    global_ids = self.db.get_dataset_path_ids(dataset_key, normalized_frames[0], normalized_frames[-1])
                else:
                    global_ids = self.db.get_dataset_path_ids(dataset_key, self.state.frame_min, self.state.frame_max)
        except Exception:
            global_ids = []
        try:
            local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, global_ids) if global_ids else []
        except Exception:
            local_ids = []
        if not local_ids:
            try:
                if normalized_frames:
                    local_ids = dataset_db.get_path_ids_for_frames(normalized_frames)
                else:
                    local_ids = dataset_db.get_path_ids()
            except Exception:
                local_ids = []
        if not local_ids:
            return []
        try:
            return dataset_db.get_path_profiles(local_ids)
        except Exception:
            return []

    def _dataset_residue_pair_distance_csv_path(self, dataset_key: str) -> str:
        dataset_key = str(dataset_key or "").strip()
        cache_key = ("distance", dataset_key)
        cached = self._dataset_residue_pair_csv_path_cache.get(cache_key)
        if cached is not None:
            return cached
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset_binding is None:
            self._dataset_residue_pair_csv_path_cache[cache_key] = ""
            return ""
        target_names = (
            "residue_pair_distances.npy",
            "residue_pair_distance.npy",
            "residue_pari_distance.npy",
            "residue_pari_distances.npy",
            "residue_pair_distances.csv",
            "residue_pair_distance.csv",
            "residue_pari_distance.csv",
            "residue_pari_distances.csv",
        )
        result = self._find_dataset_named_file(dataset_binding, target_names)
        if result:
            self._dataset_residue_pair_csv_path_cache[cache_key] = result
            return result
        self._dataset_residue_pair_csv_path_cache[cache_key] = ""
        return ""

    def _find_dataset_named_file(self, dataset_binding, target_names: tuple[str, ...]) -> str:
        search_roots: list[str] = []
        for candidate in (
            os.path.dirname(str(getattr(dataset_binding, "residue_combination_statistics_path", "") or "")),
            os.path.dirname(str(getattr(dataset_binding, "residue_statistics_path", "") or "")),
            os.path.dirname(str(getattr(dataset_binding, "path", "") or "")),
            getattr(dataset_binding, "folder", ""),
        ):
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and os.path.isdir(normalized) and normalized not in search_roots:
                search_roots.append(normalized)
        normalized_targets = {str(name).lower(): str(name) for name in target_names}
        for root in search_roots:
            try:
                lower_map = {name.lower(): name for name in os.listdir(root)}
            except OSError:
                continue
            for lower_name in normalized_targets:
                actual = lower_map.get(lower_name)
                if actual:
                    return os.path.abspath(os.path.join(root, actual))
        return ""

    def _dataset_residue_pair_presence_stats_csv_path(self, dataset_key: str) -> str:
        dataset_key = str(dataset_key or "").strip()
        cache_key = ("presence_stats", dataset_key)
        cached = self._dataset_residue_pair_csv_path_cache.get(cache_key)
        if cached is not None:
            return cached
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset_binding is None:
            self._dataset_residue_pair_csv_path_cache[cache_key] = ""
            return ""
        target_names = (
            "residue_pair_path_presence_stats.csv",
            "residue_pari_path_presence_stats.csv",
        )
        result = self._find_dataset_named_file(dataset_binding, target_names)
        if result:
            self._dataset_residue_pair_csv_path_cache[cache_key] = result
            return result
        self._dataset_residue_pair_csv_path_cache[cache_key] = ""
        return ""

    def _residue_combination_presence_stats_ready(self) -> bool:
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is None:
            return False
        try:
            file_a_key = panel.left_dataset_key() if hasattr(panel, "left_dataset_key") else ""
            file_b_key = panel.right_dataset_key() if hasattr(panel, "right_dataset_key") else ""
        except Exception:
            return False
        return bool(
            self._dataset_residue_pair_presence_stats_csv_path(file_a_key)
            and self._dataset_residue_pair_presence_stats_csv_path(file_b_key)
        )

    def _sync_residue_combination_observer_frame_distances(
        self,
        profiles_by_dataset_key: dict[str, list[dict]] | None = None,
    ) -> None:
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is None or not hasattr(panel, "set_observer_frame_distance_inputs"):
            return
        top_slot = getattr(self, "_protein_viewer_panel_top", None)
        bottom_slot = getattr(self, "_protein_viewer_panel_bottom", None)
        top_panel = top_slot.get("panel") if isinstance(top_slot, dict) else None
        bottom_panel = bottom_slot.get("panel") if isinstance(bottom_slot, dict) else None
        file_a_key = panel.left_dataset_key() if hasattr(panel, "left_dataset_key") else ""
        file_b_key = panel.right_dataset_key() if hasattr(panel, "right_dataset_key") else ""

        def frame_paths_for(observer_panel) -> dict[int, str]:
            if observer_panel is None or not hasattr(observer_panel, "sequence_frame_paths"):
                return {}
            try:
                return observer_panel.sequence_frame_paths()
            except Exception:
                return {}

        left_frame_paths = frame_paths_for(top_panel)
        right_frame_paths = frame_paths_for(bottom_panel)
        left_presence_stats_csv_path = self._dataset_residue_pair_presence_stats_csv_path(file_a_key)
        right_presence_stats_csv_path = self._dataset_residue_pair_presence_stats_csv_path(file_b_key)
        left_distance_csv_path = self._dataset_residue_pair_distance_csv_path(file_a_key)
        right_distance_csv_path = self._dataset_residue_pair_distance_csv_path(file_b_key)
        scope_mode = (
            str(panel.loaded_scope_mode() or "dataset")
            if hasattr(panel, "loaded_scope_mode")
            else "dataset"
        )
        prefer_context_stats = bool(profiles_by_dataset_key) or scope_mode == "path" or bool(
            getattr(panel, "current_path_filter_active", lambda: False)()
        ) or bool(
            getattr(panel, "current_bottleneck_filter_active", lambda: False)()
        )
        context_profiles_by_dataset: dict[str, list[dict]] = {
            str(dataset_key): list(profiles or [])
            for dataset_key, profiles in dict(profiles_by_dataset_key or {}).items()
        }
        if prefer_context_stats and not context_profiles_by_dataset:
            context_path_ids = self._residue_combination_context_path_ids(
                prefer_explicit_selection=scope_mode == "path",
            )
            if context_path_ids and hasattr(self.db, "get_local_path_ids_for_dataset"):
                for dataset_key in {str(file_a_key or "").strip(), str(file_b_key or "").strip()}:
                    if not dataset_key:
                        continue
                    dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                    dataset_db = dataset_binding.db if dataset_binding is not None else None
                    if dataset_db is None:
                        continue
                    local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                    if not local_path_ids:
                        context_profiles_by_dataset[dataset_key] = []
                        continue
                    context_profiles_by_dataset[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
        frame_distances_ready = bool(
            (left_presence_stats_csv_path and right_presence_stats_csv_path)
            or (left_distance_csv_path and right_distance_csv_path)
            or (left_frame_paths and right_frame_paths)
        )
        presence_stats_ready = bool(left_presence_stats_csv_path and right_presence_stats_csv_path) and not prefer_context_stats

        def present_frames_for(dataset_key: str) -> dict[tuple[int, ...], set[int]]:
            if not frame_distances_ready:
                return {}
            if presence_stats_ready:
                return {}
            dataset_key = str(dataset_key or "").strip()
            if not dataset_key:
                return {}
            profiles = context_profiles_by_dataset.get(dataset_key) if prefer_context_stats else None
            if profiles is None:
                frames = set(left_frame_paths.keys()) if dataset_key == file_a_key else set(right_frame_paths.keys())
                bottleneck_only = bool(
                    getattr(panel, "current_bottleneck_filter_active", lambda: False)()
                )
                cache_key = (dataset_key, tuple(sorted(int(frame) for frame in frames)), bottleneck_only)
                cached = self._residue_combo_present_frame_cache.get(cache_key)
                if cached is not None:
                    return {
                        tuple(combo): set(frame_set)
                        for combo, frame_set in cached.items()
                    }
                profiles = self._profiles_for_dataset_path_scope(dataset_key, frames)
                present_frames = self._combination_present_frames_from_profiles(
                    profiles,
                    bottleneck_only=bottleneck_only,
                )
                self._residue_combo_present_frame_cache[cache_key] = {
                    tuple(combo): set(frame_set)
                    for combo, frame_set in present_frames.items()
                }
                if len(self._residue_combo_present_frame_cache) > 64:
                    self._residue_combo_present_frame_cache.clear()
                return present_frames
            bottleneck_only = bool(
                getattr(panel, "current_bottleneck_filter_active", lambda: False)()
            )
            profile_path_ids = tuple(sorted({
                int(profile.get("pathIndex", profile.get("localPathIndex", 0)) or 0)
                for profile in profiles
                if int(profile.get("pathIndex", profile.get("localPathIndex", 0)) or 0) > 0
            }))
            profile_frame_ids = tuple(sorted({
                int(profile.get("frameId", profile.get("frame_id", 0)) or 0)
                for profile in profiles
                if int(profile.get("frameId", profile.get("frame_id", 0)) or 0) > 0
            }))
            cache_key = ("profiles", dataset_key, profile_path_ids, profile_frame_ids, bottleneck_only)
            cached = self._residue_combo_present_frame_cache.get(cache_key)
            if cached is not None:
                return {
                    tuple(combo): set(frame_set)
                    for combo, frame_set in cached.items()
                }
            present_frames = self._combination_present_frames_from_profiles(
                profiles,
                bottleneck_only=bottleneck_only,
            )
            self._residue_combo_present_frame_cache[cache_key] = {
                tuple(combo): set(frame_set)
                for combo, frame_set in present_frames.items()
            }
            if len(self._residue_combo_present_frame_cache) > 64:
                self._residue_combo_present_frame_cache.clear()
            return present_frames

        panel.set_observer_frame_distance_inputs(
            left={
                "dataset_key": file_a_key,
                "frame_paths": left_frame_paths,
                "combo_present_frames": present_frames_for(file_a_key),
                "precomputed_distance_csv_path": left_distance_csv_path,
                "precomputed_presence_stats_csv_path": left_presence_stats_csv_path,
                "prefer_context_stats": prefer_context_stats,
            },
            right={
                "dataset_key": file_b_key,
                "frame_paths": right_frame_paths,
                "combo_present_frames": present_frames_for(file_b_key),
                "precomputed_distance_csv_path": right_distance_csv_path,
                "precomputed_presence_stats_csv_path": right_presence_stats_csv_path,
                "prefer_context_stats": prefer_context_stats,
            },
        )

    def _refresh_residue_combination_contained_match_status(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        panel.set_contained_match_status(
            locked=bool(self._contained_match_current_target_key),
            locked_count=len(self._contained_match_current_target_path_ids),
            candidate_count=len(self._contained_match_current_target_path_ids),
        )

    def _clear_residue_combination_contained_match_preview(self):
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids.clear()
        self._refresh_residue_combination_contained_match_status()

    def _clear_residue_combination_contained_match(self):
        if self._contained_match_current_target_path_ids:
            self.state.apply_lasso(set(self._contained_match_current_target_path_ids), "remove")
            self._path_panel.refresh()
        self._contained_match_last_compare_summary = ""
        if getattr(self, "_residue_combination_panel", None):
            self._residue_combination_panel.clear_contained_match_result()
        self._path_panel.set_highlighted_paths([])
        self._clear_residue_combination_contained_match_preview()
        self._viewer.set_selected_combination(None)
        profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
        self._refresh_residue_combination_current_bottlenecks(
            profiles_by_dataset_key=profiles_by_dataset_key,
        )
        self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
        self._refresh_residue_combination_markers()
        self._statusbar.showMessage("Cleared matched paths and residue-pair difference results", 3000)

    def _query_residue_combination_contained_match(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        if not panel.is_compare_mode():
            self._statusbar.showMessage("Load File A and File B before querying matched paths", 3000)
            return
        self._begin_loading("Querying matched paths...")
        finished = False
        try:
            source_key = panel.contained_match_source_dataset_key()
            if not source_key:
                self._statusbar.showMessage("Select a source dataset first", 3000)
                return

            active_dataset_keys = panel.active_dataset_keys()
            target_candidates = [
                str(dataset_key)
                for dataset_key in active_dataset_keys
                if str(dataset_key) and str(dataset_key) != source_key
            ]
            if not target_candidates:
                self._statusbar.showMessage("Select a target dataset in File B before querying matched paths", 3000)
                return
            target_key = str(target_candidates[0])

            context_path_ids = self._residue_combination_context_path_ids(
                prefer_explicit_selection=True,
            )
            source_path_ids = [
                int(path_id)
                for path_id in context_path_ids
                if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == source_key
            ]
            if not source_path_ids:
                self._statusbar.showMessage("No paths are currently selected in the source dataset, so matched paths cannot be queried", 3000)
                return

            source_local_path_ids = (
                self.db.get_local_path_ids_for_dataset(source_key, source_path_ids)
                if hasattr(self.db, "get_local_path_ids_for_dataset") else []
            )
            source_binding = getattr(self.db, "_datasets_by_key", {}).get(source_key)
            target_binding = getattr(self.db, "_datasets_by_key", {}).get(target_key)
            source_db = source_binding.db if source_binding is not None else None
            target_db = target_binding.db if target_binding is not None else None
            if source_db is None or target_db is None:
                self._statusbar.showMessage("Dataset binding is unavailable, so matched paths cannot be queried", 3000)
                return

            source_profiles = source_db.get_path_profiles(source_local_path_ids)
            source_residue_ids = residue_set_from_profiles(source_profiles)
            if not source_residue_ids:
                self._statusbar.showMessage("No valid residues were extracted from the source paths, so matched paths cannot be queried", 3000)
                return

            keep_ratio = panel.contained_match_keep_ratio()

            source_exit_region = trimmed_exit_region_from_path_coords(
                source_db.get_render_coords(source_local_path_ids),
                keep_ratio,
            )
            if not source_exit_region:
                self._statusbar.showMessage("No valid exit positions were extracted from the source paths, so matched paths cannot be queried", 3000)
                return
            source_length_map = source_db.get_path_length_map(source_local_path_ids)
            source_length_region = trimmed_length_range(source_length_map, keep_ratio)
            if not source_length_region:
                self._statusbar.showMessage("No valid path lengths were extracted from the source paths, so matched paths cannot be queried", 3000)
                return
            source_length_min = float(source_length_region["min"])
            source_length_max = float(source_length_region["max"])

            target_local_path_ids = (
                self.db.get_local_path_ids_for_dataset(
                    target_key,
                    self.db.get_dataset_path_ids(target_key, self.state.frame_min, self.state.frame_max),
                )
                if hasattr(self.db, "get_dataset_path_ids") and hasattr(self.db, "get_local_path_ids_for_dataset")
                else []
            )
            target_local_path_ids = filter_path_ids_by_exit_region(
                target_db.get_render_coords(target_local_path_ids),
                source_exit_region,
            )
            target_length_map = target_db.get_path_length_map(target_local_path_ids)
            target_local_path_ids = [
                int(path_id)
                for path_id, path_length in target_length_map.items()
                if float(path_length) >= source_length_min and float(path_length) <= source_length_max
            ]
            target_local_path_ids.sort()
            target_profiles = target_db.get_path_profiles(target_local_path_ids)
            matched_target_profiles = filter_profiles_by_residue_subset(target_profiles, source_residue_ids)
            matched_target_local_ids = [
                int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
                for profile in matched_target_profiles
                if int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1) > 0
            ]
            target_path_id_base = int(getattr(target_binding, "path_id_base", 0) or 0)
            matched_target_global_ids = [
                target_path_id_base + int(local_id)
                for local_id in matched_target_local_ids
            ]
            source_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == source_key), {})
            target_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == target_key), {})
            previous_target_path_ids = set(self._contained_match_current_target_path_ids)
            self._contained_match_current_target_key = target_key
            self._contained_match_current_target_path_ids = set(matched_target_global_ids)
            self._contained_match_last_compare_summary = (
                f"Match query completed | Source [{source_dataset.get('prefix', source_key)}]: {len(source_profiles)} paths | "
                f"Keep ratio {keep_ratio:.2f} | "
                f"Exit radius {float(source_exit_region.get('radius', 0.0)):.3f} | "
                f"Length range [{source_length_min:.3f}, {source_length_max:.3f}] | "
                f"Target [{target_dataset.get('prefix', target_key)}] prefiltered: {len(target_profiles)} paths | "
                f"Residue-matched: {len(matched_target_profiles)} paths"
            )
            self._suppress_next_path_preview = True
            next_selection = set(source_path_ids) | set(matched_target_global_ids)
            if not next_selection:
                next_selection = set(context_path_ids) - previous_target_path_ids
            self.state.apply_lasso(next_selection, "replace")
            self._path_panel.refresh()
            self._path_panel.set_highlighted_paths(matched_target_global_ids)
            self._refresh_residue_combination_contained_match_status()
            if matched_target_global_ids:
                self._end_loading(
                    f"Found {len(matched_target_global_ids)} target matched paths; source-path selection was kept and this round of matched paths was added to the current selection",
                    4000,
                )
            else:
                self._end_loading(
                    "No target matched paths were found; residue-pair differences can still be computed and the target side will be treated as 0",
                    4000,
                )
            finished = True
        finally:
            if not finished:
                self._end_loading()

    def _compute_residue_combination_contained_match(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        if not panel.is_compare_mode():
            self._statusbar.showMessage("Load File A and File B before computing differences", 3000)
            return
        self._begin_loading("Computing residue-pair differences...")
        finished = False
        try:
            file_a_key = panel.left_dataset_key()
            file_b_key = panel.right_dataset_key()
            if not file_a_key or not file_b_key:
                self._statusbar.showMessage("File A/File B dataset mapping is unavailable, so differences cannot be computed", 3000)
                return

            using_matched_paths = bool(self._contained_match_current_target_key)
            profiles_by_dataset_key: dict[str, list[dict]] = {}
            summary: str

            if using_matched_paths:
                source_key = panel.contained_match_source_dataset_key()
                if not source_key:
                    self._statusbar.showMessage("Select a source dataset first", 3000)
                    return

                context_path_ids = self._residue_combination_context_path_ids()
                target_path_ids = [
                    int(path_id)
                    for path_id in context_path_ids
                    if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == self._contained_match_current_target_key
                    and int(path_id) in self._contained_match_current_target_path_ids
                ]
                source_path_ids = [
                    int(path_id)
                    for path_id in context_path_ids
                    if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == source_key
                ]
                source_binding = getattr(self.db, "_datasets_by_key", {}).get(source_key)
                target_binding = getattr(self.db, "_datasets_by_key", {}).get(self._contained_match_current_target_key)
                source_db = source_binding.db if source_binding is not None else None
                target_db = target_binding.db if target_binding is not None else None
                if source_db is None or target_db is None:
                    self._statusbar.showMessage("Dataset binding is unavailable, so differences cannot be computed", 3000)
                    return
                if not source_path_ids:
                    self._statusbar.showMessage("No source-dataset paths are currently selected, so differences cannot be computed", 3000)
                    return

                source_local_path_ids = self.db.get_local_path_ids_for_dataset(
                    source_key,
                    source_path_ids,
                )
                target_local_path_ids = self.db.get_local_path_ids_for_dataset(
                    self._contained_match_current_target_key,
                    target_path_ids,
                )
                profiles_by_dataset_key = {
                    str(source_key): source_db.get_path_profiles(source_local_path_ids),
                    str(self._contained_match_current_target_key): target_db.get_path_profiles(target_local_path_ids),
                }
                if file_a_key not in profiles_by_dataset_key or file_b_key not in profiles_by_dataset_key:
                    self._statusbar.showMessage("Matched-path source/target datasets do not match the current File A/File B selection", 3000)
                    return
                summary = (
                    f"{self._contained_match_last_compare_summary} | "
                    f"Target paths kept after manual filtering: {len(profiles_by_dataset_key.get(self._contained_match_current_target_key, []))} | "
                    f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
                )
            else:
                context_path_ids = list(self._residue_combination_context_path_ids())
                if not context_path_ids:
                    self._statusbar.showMessage("No paths are currently selected, so differences cannot be computed", 3000)
                    return
                for dataset_key in (file_a_key, file_b_key):
                    dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                    dataset_db = dataset_binding.db if dataset_binding is not None else None
                    if dataset_db is None:
                        self._statusbar.showMessage("Dataset binding is unavailable, so differences cannot be computed", 3000)
                        return
                    local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                    profiles_by_dataset_key[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
                if not profiles_by_dataset_key.get(file_a_key) and not profiles_by_dataset_key.get(file_b_key):
                    self._statusbar.showMessage("The current selection does not contain any File A or File B paths to compare", 3000)
                    return
                summary = (
                    f"Current selection compare | "
                    f"File A: {len(profiles_by_dataset_key.get(file_a_key, []))} paths | "
                    f"File B: {len(profiles_by_dataset_key.get(file_b_key, []))} paths | "
                    f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
                )

            file_a_profiles = profiles_by_dataset_key.get(file_a_key, [])
            file_b_profiles = profiles_by_dataset_key.get(file_b_key, [])
            compare_rows = compare_combination_row_sets(
                combination_rows_from_profiles(
                    file_a_profiles,
                    bottleneck_only=False,
                ),
                combination_rows_from_profiles(
                    file_b_profiles,
                    bottleneck_only=False,
                ),
                include_right_only=True,
                left_total_paths=len(file_a_profiles),
                right_total_paths=len(file_b_profiles),
            )
            bottleneck_compare_rows = compare_combination_row_sets(
                combination_rows_from_profiles(
                    file_a_profiles,
                    bottleneck_only=True,
                ),
                combination_rows_from_profiles(
                    file_b_profiles,
                    bottleneck_only=True,
                ),
                include_right_only=True,
                left_total_paths=len(file_a_profiles),
                right_total_paths=len(file_b_profiles),
            )
            panel.set_contained_match_result(
                compare_rows,
                summary,
                active=True,
                bottleneck_rows=bottleneck_compare_rows,
            )
            self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
            self._refresh_residue_combination_markers()
            self._end_loading(
                f"Computed residue-pair differences as File B - File A using {len(file_a_profiles)} File A paths and {len(file_b_profiles)} File B paths",
                4000,
            )
            finished = True
        finally:
            if not finished:
                self._end_loading()

    def _refresh_residue_combination_markers(self):
        if (
            not getattr(self, "_legacy_residue_combination_enabled", True)
            or not getattr(self, "_residue_combination_panel", None)
        ):
            return
        if not self._residue_combination_panel.combination_markers_enabled():
            self._viewer.clear_combination_markers()
            return
        markers = self._build_residue_combination_markers()
        self._viewer.render_combination_markers(markers)

    def _on_combination_marker_clicked(self, payload):
        residue_ids = tuple(int(rid) for rid in (payload or {}).get("residue_ids", ()) if int(rid) > 0)
        if len(residue_ids) < 2:
            return
        matched_ids = self.db.get_matched_path_ids(set(self._encode_residue_ids_for_all_datasets(residue_ids)))
        self.state.apply_lasso(set(matched_ids), "replace")
        self._path_panel.set_highlighted_paths(matched_ids)
        self._last_highlighted_preview_ids = set(matched_ids)
        self._statusbar.showMessage(
            f"{payload.get('combination_label', 'Residue Combination')} linked to {len(matched_ids)} paths",
            4000,
        )

    def _on_combination_selection_requested(self, payload):
        payload = dict(payload or {})
        self._observer_pending_payload = payload
        residue_ids = tuple(int(rid) for rid in payload.get("residue_ids", ()) if int(rid) > 0)
        if len(residue_ids) < 2:
            return
        if (
            getattr(self, "_protein_observer_container", None) is not None
            and not self._protein_observer_container.isVisible()
        ):
            self._toggle_protein_observer_panel(True)
        self._observer_residue_input.setText(",".join(str(rid) for rid in residue_ids))
        self._update_combination_observer_views(payload)
        self._statusbar.showMessage(
            f"Residue Observer updated for residues {', '.join(str(rid) for rid in residue_ids)}",
            4000,
        )

    def _on_left_residue_set_requested(self, payload):
        payload = dict(payload or {})
        left_residue_ids = tuple(int(rid) for rid in payload.get("left_residue_ids", ()) if int(rid) > 0)
        right_residue_ids = tuple(int(rid) for rid in payload.get("right_residue_ids", ()) if int(rid) > 0)
        if len(left_residue_ids) < 2 and len(right_residue_ids) < 2:
            self._statusbar.showMessage("No visible reference/target residue sets to show", 3000)
            return

        normalized_payload = dict(payload)
        normalized_payload["present_left"] = bool(
            str(payload.get("left_dataset_key", "") or "").strip() and len(left_residue_ids) >= 2
        )
        normalized_payload["present_right"] = bool(
            str(payload.get("right_dataset_key", "") or "").strip() and len(right_residue_ids) >= 2
        )
        normalized_payload["right_dataset_key"] = str(payload.get("right_dataset_key", "") or "")
        self._observer_pending_payload = normalized_payload

        if (
            getattr(self, "_protein_observer_container", None) is not None
            and not self._protein_observer_container.isVisible()
        ):
            self._toggle_protein_observer_panel(True)

        merged_residue_ids = tuple(
            sorted(set(int(rid) for rid in (*left_residue_ids, *right_residue_ids) if int(rid) > 0))
        )
        self._observer_residue_input.setText(",".join(str(rid) for rid in merged_residue_ids))
        self._update_combination_observer_views(normalized_payload)
        self._statusbar.showMessage(
            f"Residue Observer updated with {len(left_residue_ids)} reference and "
            f"{len(right_residue_ids)} target residues",
            4000,
        )

    # ─── Protein overlay ─────────────────────────────────
    def _set_protein_controls_enabled(self, enabled: bool):
        self._protein_toggle_btn.setEnabled(enabled)
        self._protein_style_combo.setEnabled(enabled)
        self._protein_opacity_slider.setEnabled(enabled)
        self._clear_protein_btn.setEnabled(enabled)

    def _build_residue_anchor_map(self) -> dict:
        anchors = {}
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        primary_prefix = datasets[0]["prefix"] if datasets else None
        for row in self.db.get_residue_positions():
            try:
                if primary_prefix is not None and row.get("dataset_prefix") != primary_prefix:
                    continue
                rid = int(row.get("local_residue_id", row["residue_id"]))
                anchors[rid] = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            except Exception:
                continue
        return anchors

    def _cleanup_pse_extracted_pdb(self) -> None:
        path = self._pse_extracted_pdb_path
        self._pse_extracted_pdb_path = None
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    def _extract_pse_protein_to_pdb_file(self, molecule) -> str:
        self._cleanup_pse_extracted_pdb()
        pdb_text = pse_protein_to_pdb_string(molecule)
        fd, path = tempfile.mkstemp(suffix=".pdb", prefix="new_path_pse_")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(pdb_text)
        self._pse_extracted_pdb_path = path
        return path

    def _can_reuse_loaded_pdb_for_pse(self) -> bool:
        if not self._protein_path or self._viewer._protein_model_cache is None:
            return False
        if self._protein_path == self._pse_extracted_pdb_path:
            return False
        if not self._protein_path.lower().endswith(".pdb"):
            return False
        return os.path.exists(self._protein_path)

    def _sync_protein_controls_after_load(self, source_key: str, keep_toggle_state: bool) -> None:
        self._protein_path = source_key
        self._set_protein_controls_enabled(True)
        visible = self._protein_toggle_btn.isChecked() if keep_toggle_state else True
        self._protein_toggle_btn.blockSignals(True)
        self._protein_toggle_btn.setChecked(visible)
        self._protein_toggle_btn.blockSignals(False)
        self._viewer.set_protein_opacity(self._protein_opacity_slider.value() / 100.0)
        self._viewer.set_protein_visible(visible)

    def _reload_current_protein(self, keep_toggle_state: bool = True) -> None:
        if self._pse_path:
            self._load_pse_file(self._pse_path, keep_toggle_state=keep_toggle_state)
        elif self._protein_path:
            self._load_protein_file(self._protein_path, keep_toggle_state=keep_toggle_state)

    def _load_protein_file(self, pdb_path: str, keep_toggle_state: bool = False) -> bool:
        style = self._protein_style_combo.currentText().strip() or "cartoon"
        align_mode = self._protein_align_combo.currentData() or self._protein_align_combo.currentText()
        anchors = self._build_residue_anchor_map()
        result = self._viewer.load_protein_from_file(
            pdb_path=pdb_path,
            style=style,
            align_mode=str(align_mode),
            residue_anchor_positions=anchors,
        )
        if not result.get("ok"):
            msg = str(result.get("error", "unknown_error"))
            self._statusbar.showMessage(f"Failed to load protein: {msg}", 5000)
            QMessageBox.warning(self, "Load Protein Failed", f"Failed to load protein model:\n{msg}")
            return False

        if pdb_path != self._pse_extracted_pdb_path:
            self._cleanup_pse_extracted_pdb()
            self._pse_path = None
        self._sync_protein_controls_after_load(pdb_path, keep_toggle_state)
        self._residue_combination_panel.set_residue_atom_position_maps(
            self._dataset_residue_atom_position_maps()
        )
        rmsd = result.get("rmsd")
        rmsd_text = f"{float(rmsd):.3f}" if isinstance(rmsd, (int, float)) else "n/a"
        self._statusbar.showMessage(
            (
                f"Protein loaded: atoms={result.get('atom_count', 0)}, "
                f"match={result.get('matched_count', 0)}, "
                f"align={result.get('alignment_mode', 'n/a')}, rmsd={rmsd_text}"
            ),
            6000,
        )
        return True

    def _on_load_protein(self):
        pdb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Protein Structure",
            "",
            "Supported Files (*.pdb *.pse);;PDB Files (*.pdb);;PyMOL Session (*.pse);;All Files (*)",
        )
        if not pdb_path:
            return
        if pdb_path.lower().endswith(".pse"):
            self._load_pse_file(pdb_path)
        else:
            self._load_protein_file(pdb_path, keep_toggle_state=False)

    def _on_load_observer_protein(self):
        pdb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Observer Protein PDB",
            "",
            "PDB Files (*.pdb);;All Files (*)",
        )
        if not pdb_path:
            return
        self._protein_observer_path = pdb_path
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if panel is not None:
                panel.load_pdb_file(pdb_path, first_model_only=True)
                panel.set_highlight_residues(())
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | Manual PDB",
            )
        self._statusbar.showMessage(
            f"Residue observer loaded first frame: {os.path.basename(pdb_path)}",
            4000,
        )

    def _set_bottom_analysis_mode(self, mode: str) -> None:
        panel = getattr(self, "_evidence_panel", None)
        if panel is None:
            return
        normalized = "evidence" if str(mode or "").strip().lower() == "evidence" else "charts"
        if normalized == "evidence":
            panel.select_tab("overview")
        else:
            panel.select_tab("charts")
        # Tunnel Properties and Overview share the same lower pane. Switching
        # R1/R2 must preserve the user's current splitter height.
        self._bottom_analysis_mode = normalized

    def _set_main_workspace(self, workspace: str) -> None:
        stack = getattr(self, "_main_view_stack", None)
        if stack is None:
            return
        normalized = str(workspace or "main").strip().lower()
        if normalized != "observer":
            self._observer_load_generation += 1
            self._observer_load_queue = []
            self._observer_loading_dataset_key = ""
            for panel in self._iter_protein_observer_panels():
                if hasattr(panel, "set_sequence_prefetch_enabled"):
                    panel.set_sequence_prefetch_enabled(False)
        if normalized == "observer":
            target = getattr(self, "_protein_observer_container", None)
            if target is None:
                return
            preferred_observer_keys = list(
                getattr(self, "_observer_compare_dataset_keys", []) or []
            )
            self._refresh_observer_dataset_multiselect(
                preferred_keys=preferred_observer_keys or None
            )
            self._refresh_observer_change_event_selector()
            stack.setCurrentWidget(target)
            self._set_bottom_analysis_mode("evidence")
            if not self._load_uninitialized_observer_datasets():
                self._finalize_observer_dataset_loads()
                generation = int(self._observer_load_generation)
                QTimer.singleShot(
                    650,
                    lambda token=generation: self._enable_observer_sequence_prefetch(token),
                )
            message = "Switched to Residue Observer"
        elif normalized == "evidence":
            target = getattr(self, "_evidence_panel", None)
            if target is None:
                return
            viewer = getattr(self, "_viewer", None)
            if viewer is None:
                return
            stack.setCurrentWidget(viewer)
            self._set_bottom_analysis_mode("evidence")
            target.select_tab("overview")
            self._refresh_evidence_panel(include_temporal=True)
            compare_panel = getattr(self, "_dataset_compare_panel", None)
            if compare_panel is not None and hasattr(compare_panel, "set_discovery_stage"):
                compare_panel.set_discovery_stage(
                    4,
                    "Evidence linked. Check mapping robustness, independent datasets, spatial separation, and within-trajectory persistence.",
                )
            message = "Switched to Evidence Overview"
        else:
            normalized = "main"
            target = getattr(self, "_viewer", None)
            if target is None:
                return
            stack.setCurrentWidget(target)
            self._set_bottom_analysis_mode("evidence")
            message = "Switched to main 3D view"

        buttons = (
            (getattr(self, "_main_view_btn", None), normalized == "main"),
            (getattr(self, "_load_observer_protein_btn", None), normalized == "observer"),
            (getattr(self, "_evidence_view_btn", None), normalized == "evidence"),
        )
        for button, active in buttons:
            if button is None:
                continue
            button.blockSignals(True)
            button.setChecked(active)
            button.blockSignals(False)
        self._statusbar.showMessage(message, 2500)

    def _toggle_protein_observer_panel(self, checked: bool | None = None):
        container = getattr(self, "_protein_observer_container", None)
        stack = getattr(self, "_main_view_stack", None)
        if container is None or stack is None:
            return
        if checked is None:
            checked = stack.currentWidget() is not container
        self._set_main_workspace("observer" if bool(checked) else "main")

    def _toggle_evidence_panel(self, checked: bool | None = None):
        panel = getattr(self, "_evidence_panel", None)
        if panel is None:
            return
        if checked is None:
            checked = panel.current_tab_name() == "tunnel properties"
        self._set_main_workspace("evidence" if bool(checked) else "main")

    def _on_evidence_marker_requested(self, payload: dict) -> None:
        payload = dict(payload or {})
        residue_ids = tuple(
            dict.fromkeys(int(value) for value in payload.get("residue_ids", ()) or () if int(value) > 0)
        )
        if not residue_ids:
            return
        self._observer_residue_input.setText(",".join(str(value) for value in residue_ids))
        self._apply_observer_debug_selection()
        membership = self._observer_active_path_membership(residue_ids)
        marker_lines = []
        for slot in self._protein_observer_slots():
            dataset_key = str(slot.get("dataset_key") or "")
            if not dataset_key:
                continue
            centroids = self._observer_residue_centroids_for_dataset(dataset_key, residue_ids)
            distances = []
            for left, right in combinations(residue_ids, 2):
                if left in centroids and right in centroids:
                    distances.append(
                        f"MD{left}–MD{right} {float(np.linalg.norm(centroids[left] - centroids[right])):.2f} Å"
                    )
            path_state = dict(membership.get(dataset_key, {}) or {})
            path_text = ""
            if path_state:
                path_text = (
                    f" · group on path {int(path_state.get('group_count', 0) or 0)}/"
                    f"{int(path_state.get('path_count', 0) or 0)}"
                )
            marker_lines.append(
                f"{self._dataset_compare_label(dataset_key)}: "
                + (", ".join(distances) if distances else "single-residue identity/path membership")
                + path_text
            )
        evidence_panel = getattr(self, "_evidence_panel", None)
        if evidence_panel is not None and hasattr(evidence_panel, "set_marker_result"):
            evidence_panel.set_marker_result(
                f"{payload.get('name', 'Functional marker')} | current aligned Observer frame\n"
                + ("\n".join(marker_lines) if marker_lines else "Load Observer structures to calculate marker geometry.")
            )
        explanation = getattr(self, "_observer_change_explanation", None)
        if explanation is not None:
            explanation.setText(
                f"Functional marker: {payload.get('name', 'Marker')} | MD residues "
                + ", ".join(str(value) for value in residue_ids)
                + ". Compare identities, path membership, and aligned spatial arrangement across datasets."
            )

    def _on_evidence_dataset_selected(self, target_key: str) -> None:
        active = dict(getattr(self, "_dataset_compare_active", {}) or {})
        reference_cluster = int(active.get("reference_cluster", 0) or 0)
        if not target_key or reference_cluster <= 0:
            return
        panel = getattr(self, "_dataset_compare_panel", None)
        if panel is not None and hasattr(panel, "select_comparison"):
            if panel.select_comparison(reference_cluster, str(target_key)):
                self._statusbar.showMessage(
                    f"Selected {self._dataset_compare_label(str(target_key))} from the ensemble evidence plot",
                    3500,
                )

    def _toggle_protein_visible(self, checked: bool):
        self._viewer.set_protein_visible(checked)
        if checked:
            self._statusbar.showMessage("Protein overlay enabled", 3000)
        else:
            self._statusbar.showMessage("Protein overlay hidden", 3000)

    def _on_protein_style_changed(self, style: str):
        self._viewer.set_protein_style(style)
        if self._protein_path:
            self._statusbar.showMessage(f"Protein style: {style}", 2000)

    def _on_protein_opacity_changed(self, value: int):
        self._viewer.set_protein_opacity(value / 100.0)

    def _sync_main_protein_opacity(self, opacity: float) -> None:
        """Keep the legacy hidden control in sync across protein reloads."""
        slider = getattr(self, "_protein_opacity_slider", None)
        if slider is None:
            return
        value = max(0, min(100, int(round(float(opacity) * 100.0))))
        was_blocked = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(was_blocked)

    def _on_protein_align_mode_changed(self, index: int):
        _ = index
        if not self._protein_path and not self._pse_path:
            return
        self._reload_current_protein(keep_toggle_state=True)

    def _on_clear_protein(self):
        self._viewer.clear_protein()
        self._viewer.clear_binding_sites()
        self._cleanup_pse_extracted_pdb()
        self._protein_path = None
        self._pse_path = None
        self._protein_toggle_btn.blockSignals(True)
        self._protein_toggle_btn.setChecked(False)
        self._protein_toggle_btn.blockSignals(False)
        self._set_protein_controls_enabled(False)
        self._binding_sites_btn.blockSignals(True)
        self._binding_sites_btn.setChecked(True)
        self._binding_sites_btn.setEnabled(False)
        self._binding_sites_btn.blockSignals(False)
        self._residue_combination_panel.set_residue_atom_position_maps({})
        self._statusbar.showMessage("Protein cleared", 3000)

    def _load_pse_file(self, pse_path: str, keep_toggle_state: bool = False):
        """Load a PyMOL .pse session: protein as cartoon + binding sites as surfaces."""
        try:
            molecules = parse_pse_file(pse_path)
        except Exception as exc:
            QMessageBox.warning(self, "Load PSE Failed", f"Failed to parse PSE file:\n{exc}")
            return

        if not molecules:
            QMessageBox.warning(self, "Load PSE Failed", "No molecular objects found in PSE file.")
            return

        # Separate protein(s) and binding site objects
        proteins = [m for m in molecules if m.is_protein]
        binding_sites = [m for m in molecules if not m.is_protein]

        # Prefer an already loaded real PDB, because its ribbon appearance is usually better.
        protein_loaded = False
        reused_existing_pdb = False
        protein_error: Optional[str] = None
        if self._can_reuse_loaded_pdb_for_pse():
            protein_loaded = True
            reused_existing_pdb = True
            self._sync_protein_controls_after_load(self._protein_path, keep_toggle_state)
        elif proteins:
            try:
                extracted_pdb_path = self._extract_pse_protein_to_pdb_file(proteins[0])
            except Exception as exc:
                extracted_pdb_path = ""
                protein_error = str(exc)

            if extracted_pdb_path:
                protein_loaded = self._load_protein_file(
                    extracted_pdb_path,
                    keep_toggle_state=keep_toggle_state,
                )
                if not protein_loaded:
                    protein_error = "failed_to_load_extracted_pdb"

        if not protein_loaded and not binding_sites:
            message = "No protein or binding sites found."
            if protein_error:
                message = f"Failed to load protein model:\n{protein_error}"
            QMessageBox.warning(self, "Load PSE", message)
            return

        if protein_loaded:
            self._pse_path = pse_path

        # Load binding sites
        if binding_sites:
            # Get alignment transform from protein loading
            align_R = None
            align_t = None
            align_scale = 1.0
            diag = self._viewer._protein_last_alignment_diag
            if diag and self._viewer._protein_model_cache:
                cache = self._viewer._protein_model_cache
                backbone = cache.get("backbone_raw", {})
                from TopoTunnel_UI.vis.protein_model import estimate_alignment
                anchors = self._build_residue_anchor_map()
                align_info = estimate_alignment(
                    backbone.get("ca_by_resseq", {}),
                    anchors,
                    mode=diag.get("mode", "auto_kabsch"),
                )
                align_R = np.asarray(align_info["R"], dtype=np.float64)
                align_t = np.asarray(align_info["t"], dtype=np.float64)
                align_scale = float(align_info.get("scale", 1.0))

            mol_data = [
                {"name": m.name, "coords": m.coords, "color": m.color}
                for m in binding_sites
            ]
            result = self._viewer.load_binding_sites(
                mol_data,
                align_R=align_R,
                align_t=align_t,
                align_scale=align_scale,
            )

            site_count = result.get("count", 0)
            self._binding_sites_btn.setEnabled(site_count > 0)
            self._binding_sites_btn.blockSignals(True)
            self._binding_sites_btn.setChecked(True)
            self._binding_sites_btn.blockSignals(False)

            self._statusbar.showMessage(
                (
                    f"PSE loaded: {len(proteins)} protein(s), "
                    f"{site_count} binding site surface(s)"
                    + ("; reused current PDB protein" if reused_existing_pdb else "")
                ),
                6000,
            )
        elif protein_loaded:
            self._statusbar.showMessage(
                "PSE loaded: protein only (no binding sites found)"
                + ("; reused current PDB protein" if reused_existing_pdb else ""),
                6000,
            )
        elif protein_error:
            self._statusbar.showMessage(
                f"PSE protein load failed, binding sites loaded: {protein_error}",
                6000,
            )

    def _toggle_binding_sites(self, checked: bool):
        self._viewer.set_binding_sites_visible(checked)
        if checked:
            self._statusbar.showMessage("Binding site surfaces shown", 3000)
        else:
            self._statusbar.showMessage("Binding site surfaces hidden", 3000)

    def _on_export(self, cat_name, path_ids):
        """Export category paths to PDB files."""
        split_frame_paths = self._prompt_export_options()
        if split_frame_paths is None:
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not dir_path:
            return

        exported = export_paths_to_pdb(
            self.db,
            path_ids,
            cat_name,
            dir_path,
            split_frame_paths=split_frame_paths,
        )
        mode_text = "split by path" if split_frame_paths else "merged by frame"
        metadata_text = (
            "\nAn additional TXT metadata summary file was generated for split exports."
            if split_frame_paths
            else ""
        )
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {len(exported)} PDB files to:\n{dir_path}\n"
            f"Mode: {mode_text}{metadata_text}"
        )

    def _prompt_export_options(self) -> Optional[bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle("PDB Export Options")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        description = QLabel(
            "Choose whether paths from the same frame are exported into one PDB "
            "file or split into multiple files."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        split_checkbox = QCheckBox("Split same-frame paths into separate files")
        split_checkbox.setChecked(False)
        layout.addWidget(split_checkbox)

        hint = QLabel(
            "Off: Category_234606_33_centroids.pdb\n"
            "On: Category_234606_33_centroids_1.pdb, "
            "Category_234606_33_centroids_2.pdb, ..."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        return split_checkbox.isChecked()

    # ─── Frame range ────────────────────────────────────
    def _apply_frame_range(self):
        try:
            fmin = int(self._frame_min_combo.currentText())
            fmax = int(self._frame_max_combo.currentText())
            self.state.set_frame_range(fmin, fmax)
            self._path_panel.refresh()
        except ValueError:
            self._statusbar.showMessage("Invalid frame range", 3000)

    def _clear_frame_range(self):
        self.state.set_frame_range(None, None)
        self._path_panel.refresh()
        self._reload_3d()

    def _format_path_label(self, path_id: int) -> str:
        if hasattr(self.db, "format_path_label"):
            return self.db.format_path_label(path_id)
        return str(path_id)

    def _primary_analysis_db(self):
        if hasattr(self.db, "primary_db"):
            return self.db.primary_db
        return self.db

    def _sync_residue_combination_context(self):
        primary_key = getattr(self.db, "primary_dataset_key", None)
        local_residue_ids = []
        if primary_key and hasattr(self.db, "get_local_residue_ids_for_dataset"):
            local_residue_ids = self.db.get_local_residue_ids_for_dataset(
                primary_key,
                self.state.selected_residues,
            )
        else:
            local_residue_ids = [int(rid) for rid in self.state.selected_residues]
        self._residue_combination_panel.set_current_selected_residues(local_residue_ids)

    def _reset_state_for_dataset_change(self):
        self.state.selected_residues.clear()
        self.state.residue_filtered_path_ids.clear()
        self.state.lasso_added_path_ids.clear()
        self.state.lasso_removed_path_ids.clear()
        self.state.effective_path_ids.clear()
        self.state.selected_path_ids.clear()
        self.state.matched_path_ids.clear()
        self.state.categories.clear()
        self.state.category_colors.clear()
        self.state.visible_categories.clear()
        self.state.excluded_categories.clear()
        self.state.active_category = None
        self.state.frame_min = None
        self.state.frame_max = None
        self.state.current_session_id = None
        self.state.current_session_name = None
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        self._primary_residue_position_cache.clear()
        self._dataset_residue_position_cache.clear()
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids.clear()
        self._contained_match_last_compare_summary = ""
        self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
        self._status_selected.setText("Selected: 0 paths")
        self._viewer.clear_effective()
        self._viewer.clear_categories()
        self._viewer.clear_matched()
        self._clear_chart_views()
        for slot in self._protein_observer_slots():
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is not None and hasattr(panel, "clear_path_overlay"):
                panel.clear_path_overlay()
        self.state.residue_selection_changed.emit()
        self.state.category_changed.emit()
        self.state.category_visibility_changed.emit()
        self.state.effective_paths_changed.emit()

    def _on_datasets_changed(self, change_info=None):
        info = change_info if isinstance(change_info, dict) else {}
        self._observer_load_generation += 1
        self._observer_load_queue = []
        self._observer_loading_dataset_key = ""
        self._observer_dataset_layout_signature = ()
        self._observer_sequence_source_cache.clear()
        self._observer_alignment_metadata_cache.clear()
        self._observer_frame_transform_cache.clear()
        self._dataset_compare_residue_profile_cache.clear()
        self._dataset_compare_path_scope_cache.clear()
        if hasattr(self, "_dataset_compare_relation_cache"):
            self._dataset_compare_relation_cache.clear()
        if getattr(self, "_dataset_compare_active", None):
            self._clear_dataset_compare()
        action = str(info.get("action") or "")
        reset_state = bool(info.get("reset_state", True))
        refresh_3d = bool(info.get("refresh_3d", True))
        removed_keys = [
            str(key)
            for key in (info.get("removed_keys") or [])
            if str(key)
        ]
        removed_key = str(info.get("removed_key") or "")
        if removed_key and removed_key not in removed_keys:
            removed_keys.append(removed_key)
        if removed_keys:
            removed_key_set = set(removed_keys)
            self._dataset_compare_mapping_locks = {
                pair: locks
                for pair, locks in self._dataset_compare_mapping_locks.items()
                if not (set(pair) & removed_key_set)
            }
        removed_path_ids = {
            int(path_id)
            for path_id in (info.get("removed_path_ids") or [])
        }

        for key in removed_keys:
            self.state.visible_datasets.discard(key)
            if self.state.dataset_cluster_display_key == key:
                self.state.dataset_cluster_display_key = None
            if self.state.dataset_cluster_label_key == key:
                self.state.dataset_cluster_label_key = None
            self.state.dataset_colors.pop(key, None)

        if reset_state:
            self._reset_state_for_dataset_change()
        elif removed_path_ids:
            self._prune_removed_dataset_state(removed_path_ids)
        if not refresh_3d:
            self._refresh_dataset_views_without_3d()
        else:
            self._refresh_dataset_views()
        if self.state.show_all_residues:
            positions = self._residue_positions_with_dataset_colors()
            self._viewer.set_residue_data(positions)
            self._viewer.render_residues(set())
            self._refresh_residue_labels()
        self._prime_primary_residue_positions()
        if self._legacy_residue_combination_enabled:
            profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
            self._refresh_residue_combination_current_bottlenecks(
                profiles_by_dataset_key=profiles_by_dataset_key,
            )
            self._refresh_residue_combination_markers()
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays()
        self._refresh_observer_dataset_multiselect()
        if self._protein_observer_page_active():
            self._load_uninitialized_observer_datasets()
        if self._protein_path or self._pse_path:
            self._reload_current_protein(keep_toggle_state=True)
        dataset_count = len(self.db.list_datasets())
        if action == "remove" and not reset_state:
            self._statusbar.showMessage(
                f"Removed hidden dataset; kept current view ({dataset_count} dataset(s) loaded)",
                4000,
            )
        elif reset_state:
            self._statusbar.showMessage(
                f"Loaded {dataset_count} dataset(s); analysis state was reset",
                4000,
            )
        else:
            self._statusbar.showMessage(
                f"Loaded {dataset_count} dataset(s)",
                3000,
            )

    def _prune_removed_dataset_state(self, removed_path_ids: set[int]):
        if not removed_path_ids:
            return

        self.state.residue_filtered_path_ids.difference_update(removed_path_ids)
        self.state.lasso_added_path_ids.difference_update(removed_path_ids)
        self.state.lasso_removed_path_ids.difference_update(removed_path_ids)
        self.state.effective_path_ids.difference_update(removed_path_ids)
        self.state.selected_path_ids.difference_update(removed_path_ids)
        self.state.box_selected_paths.difference_update(removed_path_ids)
        self.state.global_excluded_paths.difference_update(removed_path_ids)
        self.state.matched_path_ids = [
            int(path_id)
            for path_id in self.state.matched_path_ids
            if int(path_id) not in removed_path_ids
        ]

        for name, path_ids in list(self.state.categories.items()):
            filtered = [
                int(path_id)
                for path_id in path_ids
                if int(path_id) not in removed_path_ids
            ]
            self.state.categories[name] = filtered

        self._last_preview_path_ids = [
            int(path_id)
            for path_id in self._last_preview_path_ids
            if int(path_id) not in removed_path_ids
        ]
        self._last_highlighted_preview_ids.difference_update(removed_path_ids)
        for path_id in list(self._last_preview_color_map.keys()):
            if int(path_id) in removed_path_ids:
                self._last_preview_color_map.pop(path_id, None)
        self._contained_match_current_target_path_ids.difference_update(removed_path_ids)
        self._path_delta_excluded_path_ids.difference_update(removed_path_ids)
        self._chart_operation_history = [
            [int(path_id) for path_id in path_ids if int(path_id) not in removed_path_ids]
            for path_ids in self._chart_operation_history
        ]

        self.state.recompute_effective()
        self.state.effective_paths_changed.emit()
        self.state.category_changed.emit()

    def _refresh_dataset_views_without_3d(self):
        self._invalidate_render_signatures()
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        n_paths = self.db.path_count()
        fmin, fmax = self.db.frame_range()

        self._status_paths.setText(f"Paths: {n_paths:,}")
        n_res = int(self.db.get_meta("residue_count", "0"))
        self._status_residues.setText(f"Residues: {n_res:,}")

        self._frame_min_combo.clear()
        self._frame_max_combo.clear()
        self._frame_min_combo.addItem(str(fmin))
        self._frame_max_combo.addItem(str(fmax))
        self._refresh_combo_popup_width(self._frame_min_combo)
        self._refresh_combo_popup_width(self._frame_max_combo)

        if self._legacy_residue_panel_enabled:
            self._residue_panel.set_db(self.db)
            self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
            self._residue_panel.load_options()

        if self._legacy_path_panel_enabled:
            self._path_panel.set_db(self.db)
            self._path_panel.init_model()

        self.state.ensure_dataset_defaults([
            dataset["key"] for dataset in self.db.list_datasets()
        ])

        self._dataset_panel.refresh()
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        if self._legacy_residue_compare_enabled:
            self._residue_compare_panel.set_dataset_choices(datasets)
        if self._legacy_residue_combination_enabled:
            self._residue_combination_panel.set_db(self._primary_analysis_db())
            self._residue_combination_panel.set_dataset_choices(datasets)
            self._residue_combination_panel.set_residue_position_maps(
                self._dataset_residue_position_maps()
            )
            self._residue_combination_panel.set_residue_atom_position_maps(
                self._dataset_residue_atom_position_maps()
            )
        self._refresh_dataset_compare_cluster_options()
        if self._legacy_residue_compare_enabled:
            self._sync_residue_compare_selection_state()
        self._prime_primary_residue_positions()
        if self._legacy_residue_combination_enabled:
            profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
            self._refresh_residue_combination_current_bottlenecks(
                profiles_by_dataset_key=profiles_by_dataset_key,
            )
            self._refresh_residue_combination_contained_match_status()
            self._sync_residue_combination_context()

    # ─── Cleanup ────────────────────────────────────────
    def closeEvent(self, event):
        for panel in self._iter_protein_observer_panels():
            try:
                panel.shutdown()
            except Exception:
                pass
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "shutdown"):
            try:
                viewer.shutdown()
            except Exception:
                pass
        self.db.close()
        super().closeEvent(event)


class _BackgroundRoleDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        background = index.data(Qt.BackgroundRole)
        if isinstance(background, QBrush):
            option.backgroundBrush = background


class _ResidueSummaryHeatmapDialog(QDialog):
    def __init__(self, owner: ChartWorkspaceDialog):
        super().__init__(owner)
        self._owner = owner
        self.setWindowTitle("Residue Summary Compare")
        self.setModal(False)
        self.setMinimumSize(960, 680)
        self.resize(1120, 780)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Residue Summary Compare")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self._summary = QLabel("No heatmap data.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #606266; font-size: 11px;")
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionsClickable(True)
        self._table.horizontalHeader().sectionClicked.connect(self._on_delta_header_clicked)
        self._table.setSortingEnabled(False)
        self._table.setItemDelegate(_BackgroundRoleDelegate(self._table))
        self._table.setObjectName("ResidueSummaryCompareTable")
        self._table.setAutoFillBackground(True)
        self._table.setStyleSheet(
            "QTableWidget#ResidueSummaryCompareTable { background: #ffffff; }"
            "QTableWidget#ResidueSummaryCompareTable::item { padding: 4px 8px; }"
            "QTableWidget#ResidueSummaryCompareTable::item:hover { background: rgba(0, 0, 0, 0); }"
        )
        layout.addWidget(self._table, 1)

        self._legend = QLabel("")
        self._legend.setWordWrap(True)
        self._legend.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(self._legend)

        self._payload = {"datasets": [], "rows": [], "kind": "path"}
        self._sort_column = 6
        self._sort_descending = True

    def refresh(self, payload: dict):
        self._payload = dict(payload or {"datasets": [], "rows": [], "kind": "path"})
        self._rebuild_table()

    def _on_delta_header_clicked(self, section: int):
        if section == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = section
            self._sort_descending = section not in {0}
        self._rebuild_table()

    def _rebuild_table(self):
        datasets = list(self._payload.get("datasets", []) or [])
        delta_rows = list(self._payload.get("delta_rows", []) or [])
        kind = str(self._payload.get("kind", "path") or "path")
        kind_label = "Path Residues" if kind == "path" else "Bottleneck"

        if len(datasets) < 2 or not delta_rows:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._summary.setText(f"Need two datasets with {kind_label} data for comparison.")
            self._legend.setText("")
            return

        self._rebuild_delta_table(datasets, delta_rows, kind_label)

    def _rebuild_delta_table(self, datasets: list, delta_rows: list, kind_label: str):
        base_dataset = str(datasets[0])
        compare_dataset = str(datasets[1])
        headers = [
            "Residue",
            f"{base_dataset} Count",
            f"{base_dataset} %",
            f"{compare_dataset} Count",
            f"{compare_dataset} %",
            "Delta Count",
            "Delta %",
            "Status",
        ]
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setRowCount(len(delta_rows))
        self._table.setHorizontalHeaderLabels(headers)

        delta_rows = self._sorted_delta_rows(delta_rows)
        max_abs_delta = max(abs(int(row.get("delta_count", 0) or 0)) for row in delta_rows)
        for row_idx, row in enumerate(delta_rows):
            status = str(row.get("status", "same") or "same")
            delta_count = int(row.get("delta_count", 0) or 0)
            delta_percent = float(row.get("delta_percent", 0.0) or 0.0)
            background = self._delta_status_color(status, delta_count, max_abs_delta)
            status_text = self._delta_status_text(row)
            values = [
                str(int(row.get("residue_id", 0) or 0)),
                str(int(row.get("base_count", 0) or 0)),
                f"{float(row.get('base_percent', 0.0) or 0.0):.1f}%",
                str(int(row.get("compare_count", 0) or 0)),
                f"{float(row.get('compare_percent', 0.0) or 0.0):.1f}%",
                f"{delta_count:+d}",
                f"{delta_percent:+.1f}%",
                status_text,
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.BackgroundRole, QBrush(background))
                item.setData(Qt.ForegroundRole, QColor("#1f2937"))
                item.setBackground(QBrush(background))
                if col_idx == 7:
                    item.setForeground(QBrush(self._delta_status_text_color(status)))
                self._table.setItem(row_idx, col_idx, item)

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._summary.setText(
            f"{kind_label} delta table comparing {compare_dataset} against {base_dataset}. "
            f"Percent = residue path count / current selected path count."
        )
        self._legend.setText(
            "Legend: red = comparison increased, blue = comparison did not increase/decreased, "
            "green = new Bottleneck residue."
        )

    def _sorted_delta_rows(self, delta_rows: list) -> list:
        status_rank = {"new": 3, "up": 2, "same": 1, "down": 0}

        def key(row):
            status = str(row.get("status", "same") or "same")
            keys = [
                int(row.get("residue_id", 0) or 0),
                int(row.get("base_count", 0) or 0),
                float(row.get("base_percent", 0.0) or 0.0),
                int(row.get("compare_count", 0) or 0),
                float(row.get("compare_percent", 0.0) or 0.0),
                int(row.get("delta_count", 0) or 0),
                float(row.get("delta_percent", 0.0) or 0.0),
                status_rank.get(status, 1),
            ]
            primary = keys[self._sort_column] if 0 <= self._sort_column < len(keys) else keys[6]
            return (primary, int(row.get("residue_id", 0) or 0))

        return sorted(delta_rows, key=key, reverse=self._sort_descending)

    def _delta_status_text(self, row: dict) -> str:
        status = str(row.get("status", "same") or "same")
        base_rank = row.get("base_rank")
        compare_rank = row.get("compare_rank")
        if status == "new":
            return f"NEW {compare_rank}" if compare_rank is not None else "NEW"
        if status == "up":
            return f"{base_rank} UP {compare_rank}" if base_rank is not None and compare_rank is not None else "UP"
        if status == "down":
            return f"{base_rank} DOWN {compare_rank}" if base_rank is not None and compare_rank is not None else "DOWN"
        return f"{base_rank} SAME {compare_rank}" if base_rank is not None and compare_rank is not None else "SAME"

    def _delta_status_text_color(self, status: str) -> QColor:
        if status == "new":
            return QColor("#1B5E20")
        if status == "up":
            return QColor("#7F1D1D")
        return QColor("#1E3A8A")

    def _delta_status_color(self, status: str, delta_count: int, max_abs_delta: int) -> QColor:
        if status == "new":
            return QColor("#B7E4A8")
        if status == "up":
            strength = min(1.0, abs(delta_count) / max(1, max_abs_delta))
            return self._owner._blend_with_white((220, 38, 38), 0.45 + 0.35 * strength)
        strength = 0.55 if delta_count == 0 else min(0.90, 0.45 + abs(delta_count) / max(1, max_abs_delta) * 0.35)
        return self._owner._blend_with_white((37, 99, 235), strength)
