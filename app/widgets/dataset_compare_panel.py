"""Controls for comparing the currently selected cluster across datasets."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QBrush, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QToolTip, QComboBox,
    QDoubleSpinBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QSizePolicy,
)


class ResidueReplacementTrack(QWidget):
    """Compact full-path heatmap with selectable residue-change hotspots."""

    hotspot_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bins: list[dict] = []
        self._hotspots: list[dict] = []
        self._reference_bottleneck: dict = {}
        self._target_bottleneck: dict = {}
        self._selected_index = -1
        self.setMinimumHeight(112)
        self.setMouseTracking(True)
        # Hovering the map should expose the underlying evidence immediately.
        # Do not let a static interaction hint obscure the dynamic tooltip.
        self.setToolTip("")

    def set_profile(self, profile: dict | None):
        profile = dict(profile or {})
        self._bins = [dict(row) for row in profile.get("bins", []) if isinstance(row, dict)]
        self._hotspots = [dict(row) for row in profile.get("hotspots", []) if isinstance(row, dict)]
        self._reference_bottleneck = dict(profile.get("reference_bottleneck", {}) or {})
        self._target_bottleneck = dict(profile.get("target_bottleneck", {}) or {})
        self._selected_index = next(
            (
                index
                for index, hotspot in enumerate(self._hotspots)
                if str(
                    hotspot.get("region_id") or hotspot.get("hotspot_id") or ""
                ).strip().upper() == "R1"
            ),
            0 if self._hotspots else -1,
        )
        self.update()

    def clear(self):
        self._bins = []
        self._hotspots = []
        self._reference_bottleneck = {}
        self._target_bottleneck = {}
        self._selected_index = -1
        self.update()

    def selected_hotspot(self) -> dict | None:
        if 0 <= self._selected_index < len(self._hotspots):
            return dict(self._hotspots[self._selected_index])
        return None

    def select_hotspot(self, index: int, *, emit: bool = False) -> dict | None:
        """Select a replacement region programmatically and optionally emit it."""
        if not self._hotspots:
            self._selected_index = -1
            self.update()
            return None
        self._selected_index = max(0, min(int(index), len(self._hotspots) - 1))
        payload = dict(self._hotspots[self._selected_index])
        self.update()
        if emit:
            self.hotspot_selected.emit(payload)
        return payload

    def select_hotspot_by_id(self, region_id: str, *, emit: bool = False) -> dict | None:
        normalized = str(region_id or "")
        for index, hotspot in enumerate(self._hotspots):
            candidate = str(hotspot.get("region_id") or hotspot.get("hotspot_id") or "")
            if normalized and candidate == normalized:
                return self.select_hotspot(index, emit=emit)
        return self.selected_hotspot()

    def step_hotspot(self, delta: int, *, emit: bool = False) -> dict | None:
        if not self._hotspots:
            return None
        current = self._selected_index if self._selected_index >= 0 else 0
        return self.select_hotspot((current + int(delta)) % len(self._hotspots), emit=emit)

    def _track_rect(self) -> QRectF:
        return QRectF(18.0, 34.0, max(1.0, float(self.width()) - 36.0), 24.0)

    def _bottleneck_label_rect(
        self,
        fraction: float,
        top: float,
        slot_index: int = 0,
        slot_count: int = 1,
    ) -> QRectF:
        track = self._track_rect()
        if slot_count <= 1:
            width = min(220.0, track.width())
            center_x = track.left() + max(0.0, min(1.0, fraction)) * track.width()
            left = max(track.left(), min(center_x - width / 2.0, track.right() - width))
            return QRectF(left, top, width, 24.0)

        gap = 8.0
        slot_width = max(1.0, (track.width() - gap * (slot_count - 1)) / slot_count)
        width = min(250.0, slot_width)
        slot_left = track.left() + slot_index * (slot_width + gap)
        left = slot_left + (slot_width - width) / 2.0
        return QRectF(left, top, width, 24.0)

    @staticmethod
    def _bottleneck_values(payload: dict) -> tuple[float, float, float] | None:
        if not payload or payload.get("fraction") is None:
            return None
        fraction = max(0.0, min(1.0, float(payload.get("fraction", 0.0))))
        q25 = max(0.0, min(1.0, float(payload.get("fraction_q25", fraction))))
        q75 = max(0.0, min(1.0, float(payload.get("fraction_q75", fraction))))
        return fraction, min(q25, q75), max(q25, q75)

    @staticmethod
    def _score_color(score: float) -> QColor:
        value = max(0.0, min(1.0, float(score)))
        if value < 0.35:
            ratio = value / 0.35
            start = QColor("#E5E7EB")
            end = QColor("#FDE68A")
        else:
            ratio = (value - 0.35) / 0.65
            start = QColor("#FDE68A")
            end = QColor("#DC2626")
        return QColor(
            round(start.red() + (end.red() - start.red()) * ratio),
            round(start.green() + (end.green() - start.green()) * ratio),
            round(start.blue() + (end.blue() - start.blue()) * ratio),
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        readable_font = painter.font()
        readable_font.setFamily("Arial")
        if readable_font.pointSizeF() > 0:
            readable_font.setPointSizeF(max(10.5, readable_font.pointSizeF()))
        readable_font.setBold(True)
        painter.setFont(readable_font)
        track = self._track_rect()
        painter.setPen(QPen(QColor("#D1D5DB"), 1.0))
        painter.setBrush(QColor("#F3F4F6"))
        painter.drawRoundedRect(track, 4.0, 4.0)

        if self._bins:
            count = len(self._bins)
            bin_width = track.width() / max(1, count)
            painter.setPen(Qt.PenStyle.NoPen)
            for index, row in enumerate(self._bins):
                rect = QRectF(
                    track.left() + index * bin_width,
                    track.top(),
                    bin_width + 0.6,
                    track.height(),
                )
                painter.setBrush(self._score_color(float(row.get("score", 0.0))))
                painter.drawRect(rect)

        bottlenecks = (
            ("BN-Ref", self._reference_bottleneck, QColor("#2563EB"), 68.0, True, 0),
            ("BN-Target", self._target_bottleneck, QColor("#EA580C"), 68.0, False, 1),
        )
        for label, payload, color, label_top, upper_half, slot_index in bottlenecks:
            values = self._bottleneck_values(payload)
            if values is None:
                continue
            fraction, q25, q75 = values
            band_color = QColor(color)
            band_color.setAlpha(105)
            band_top = track.top() if upper_half else track.center().y()
            band = QRectF(
                track.left() + q25 * track.width(),
                band_top,
                max(2.0, (q75 - q25) * track.width()),
                track.height() / 2.0,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(band_color)
            painter.drawRect(band)

            x = track.left() + fraction * track.width()
            painter.setPen(QPen(color, 2.0, Qt.PenStyle.DashLine))
            painter.drawLine(int(x), int(track.top() - 2.0), int(x), int(track.bottom() + 3.0))

            label_rect = self._bottleneck_label_rect(
                fraction,
                label_top,
                slot_index=slot_index,
                slot_count=2,
            )
            label_background = QColor(color)
            label_background.setAlpha(28)
            painter.setBrush(label_background)
            painter.setPen(QPen(color, 1.0))
            painter.drawRoundedRect(label_rect, 4.0, 4.0)
            radius = payload.get("radius")
            radius_text = f" · r={float(radius):.2f} Å" if radius is not None else ""
            painter.setPen(color)
            painter.drawText(
                label_rect,
                Qt.AlignCenter,
                f"{label} {fraction * 100:.1f}%{radius_text}",
            )

        for index, hotspot in enumerate(self._hotspots):
            peak = max(0.0, min(1.0, float(hotspot.get("peak_fraction", 0.0))))
            x = track.left() + peak * track.width()
            selected = index == self._selected_index
            painter.setPen(QPen(QColor("#991B1B"), 2.5 if selected else 1.5))
            painter.drawLine(int(x), int(track.top() - 5.0), int(x), int(track.bottom() + 3.0))
            marker = QRectF(x - 13.0, 7.0, 26.0, 21.0)
            painter.setBrush(QColor("#DC2626") if selected else QColor("#FEE2E2"))
            painter.setPen(QPen(QColor("#991B1B"), 1.0))
            painter.drawRoundedRect(marker, 4.0, 4.0)
            painter.setPen(QColor("#FFFFFF") if selected else QColor("#991B1B"))
            painter.drawText(marker, Qt.AlignCenter, str(hotspot.get("region_id") or hotspot.get("hotspot_id") or f"R{index + 1}"))

    def _bottleneck_at(self, x: float, y: float) -> tuple[str, dict] | None:
        track = self._track_rect()
        if y < track.top() - 4.0 or y > 100.0:
            return None
        candidates = (
            ("BN-Ref", self._reference_bottleneck, 68.0, 0),
            ("BN-Target", self._target_bottleneck, 68.0, 1),
        )
        for label, payload, label_top, slot_index in candidates:
            values = self._bottleneck_values(payload)
            if values is None:
                continue
            fraction, q25, q75 = values
            marker_x = track.left() + fraction * track.width()
            band_left = track.left() + q25 * track.width()
            band_right = track.left() + q75 * track.width()
            label_rect = self._bottleneck_label_rect(
                fraction,
                label_top,
                slot_index=slot_index,
                slot_count=2,
            )
            if label_rect.contains(float(x), float(y)) or (
                track.top() <= y <= track.bottom()
                and (abs(float(x) - marker_x) <= 7.0 or band_left <= x <= band_right)
            ):
                return label, payload
        return None

    def _hotspot_index_at(self, x: float) -> int:
        if not self._hotspots:
            return -1
        track = self._track_rect()
        fraction = max(0.0, min(1.0, (float(x) - track.left()) / max(1.0, track.width())))
        containing = [
            index for index, hotspot in enumerate(self._hotspots)
            if float(hotspot.get("start_fraction", 0.0)) <= fraction <= float(hotspot.get("end_fraction", 0.0))
        ]
        if containing:
            return containing[0]
        return min(
            range(len(self._hotspots)),
            key=lambda index: abs(float(self._hotspots[index].get("peak_fraction", 0.0)) - fraction),
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._hotspot_index_at(event.position().x())
            if index >= 0:
                self._selected_index = index
                self.update()
                self.hotspot_selected.emit(dict(self._hotspots[index]))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        bottleneck = self._bottleneck_at(event.position().x(), event.position().y())
        if bottleneck is not None:
            label, payload = bottleneck
            fraction = float(payload.get("fraction", 0.0)) * 100.0
            q25 = float(payload.get("fraction_q25", payload.get("fraction", 0.0))) * 100.0
            q75 = float(payload.get("fraction_q75", payload.get("fraction", 0.0))) * 100.0
            radius = payload.get("radius")
            radius_text = f"{float(radius):.3f} Å" if radius is not None else "not available"
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{label} | median arc position {fraction:.1f}%\n"
                f"Arc-position IQR {min(q25, q75):.1f}–{max(q25, q75):.1f}%\n"
                f"Median minimum radius {radius_text} | paths {int(payload.get('path_count', 0) or 0)}",
                self,
            )
            super().mouseMoveEvent(event)
            return
        index = self._hotspot_index_at(event.position().x())
        if index >= 0:
            hotspot = self._hotspots[index]
            lost = ", ".join(
                f"S{int(rid)}" for rid in hotspot.get("lost_sequence_ids", ()) or ()
            ) or "none"
            gained = ", ".join(
                f"S{int(rid)}" for rid in hotspot.get("gained_sequence_ids", ()) or ()
            ) or "none"
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{hotspot.get('region_label', hotspot.get('hotspot_id', ''))} | replacement {float(hotspot.get('max_score', 0.0)) * 100:.0f}%\n"
                f"Reference/lost: {lost}\nTarget/gained: {gained}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)


class AllostericEvidenceStrip(QWidget):
    """Mutation→path→bottleneck evidence chain for hypothesis formation."""

    hotspot_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile: dict = {}
        self._selected_region_id = ""
        self._hotspot_rect = QRectF()
        self.setMinimumHeight(104)
        self.setToolTip(
            "Mutation-to-active-site remoteness and path-region response localization."
        )

    def set_profile(self, profile: dict | None) -> None:
        self._profile = dict(profile or {})
        region_ids = [
            str(row.get("region_id") or row.get("hotspot_id") or "")
            for row in self._profile.get("hotspots", []) or []
        ]
        preferred = str(self._profile.get("selected_region") or "")
        if preferred in region_ids:
            self._selected_region_id = preferred
        elif self._selected_region_id not in region_ids:
            self._selected_region_id = region_ids[0] if region_ids else ""
        self.update()

    def set_selected_region(self, region_id: str) -> None:
        self._selected_region_id = str(region_id or "")
        self.update()

    def selected_region_id(self) -> str:
        return self._selected_region_id

    def _selected_hotspot(self) -> dict:
        region_id = str(self._selected_region_id or "")
        evidence = [dict(row) for row in self._profile.get("allosteric_evidence", []) or []]
        hotspots = [dict(row) for row in self._profile.get("hotspots", []) or []]
        for rows in (evidence, hotspots):
            selected = next((
                row for row in rows
                if region_id and str(row.get("region_id") or row.get("hotspot_id") or "") == region_id
            ), None)
            if selected:
                return selected
        return evidence[0] if evidence else (hotspots[0] if hotspots else {})

    def clear(self) -> None:
        self._profile = {}
        self._selected_region_id = ""
        self._hotspot_rect = QRectF()
        self.update()

    @staticmethod
    def _card(painter: QPainter, rect: QRectF, title: str, detail: str, color: str) -> None:
        fill = QColor(color)
        fill.setAlpha(24)
        painter.setPen(QPen(QColor(color), 1.2))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 7.0, 7.0)
        painter.setPen(QColor(color))
        painter.drawText(rect.adjusted(8.0, 5.0, -8.0, -rect.height() / 2.0), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.setPen(QColor("#374151"))
        painter.drawText(
            rect.adjusted(8.0, rect.height() / 2.0 - 4.0, -8.0, -4.0),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextFlag.TextWordWrap,
            detail,
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        profile = self._profile
        mutations = [dict(row) for row in profile.get("substitutions", []) or []]
        perturbation_label = str(profile.get("perturbation_label") or "").strip()
        if not mutations and not perturbation_label:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No explicit amino-acid substitution detected between these datasets")
            return
        hotspot = self._selected_hotspot()
        margin = 8.0
        gap = max(12.0, min(34.0, float(self.width()) * 0.055))
        available = max(90.0, float(self.width()) - 2.0 * margin - 2.0 * gap)
        card_width = available / 3.0
        top = 8.0
        card_height = 62.0
        mutation_rect = QRectF(margin, top, card_width, card_height)
        hotspot_rect = QRectF(mutation_rect.right() + gap, top, card_width, card_height)
        bottleneck_rect = QRectF(hotspot_rect.right() + gap, top, card_width, card_height)
        self._hotspot_rect = hotspot_rect

        mutation_text = ", ".join(str(row.get("label") or "mutation") for row in mutations[:2])
        if len(mutations) > 2:
            mutation_text += f" +{len(mutations) - 2}"
        if not mutation_text:
            mutation_text = perturbation_label
        active_site_distance = hotspot.get("active_site_distance")
        if active_site_distance is not None and mutations:
            mutation_text = (
                f"{str(hotspot.get('mutation_label') or mutation_text)} · "
                f"active site {float(active_site_distance):.1f} Å"
            )
        self._card(painter, mutation_rect, "1  Perturbation", mutation_text, "#7C3AED")

        if hotspot:
            region = str(hotspot.get("region_id") or "R1")
            position = float(hotspot.get("peak_fraction", 0.0)) * 100.0
            score = float(hotspot.get("replacement_score", hotspot.get("max_score", 0.0)) or 0.0) * 100.0
            self._card(painter, hotspot_rect, "2  Path-local response", f"{region} at {position:.1f}% · Δres {score:.0f}%", "#DC2626")
        else:
            self._card(painter, hotspot_rect, "2  Path-local response", "distance unavailable", "#9CA3AF")

        fraction_shift = profile.get("bottleneck_fraction_shift")
        radius_shift = profile.get("bottleneck_radius_shift")
        if fraction_shift is None and radius_shift is None:
            bottleneck_text = "bottleneck comparison unavailable"
        else:
            parts = []
            if fraction_shift is not None:
                parts.append(f"arc {float(fraction_shift) * 100:+.1f}%")
            if radius_shift is not None:
                parts.append(f"radius {float(radius_shift):+.2f} Å")
            bottleneck_text = " · ".join(parts)
        self._card(painter, bottleneck_rect, "3  Gate response", bottleneck_text, "#EA580C")

        painter.setPen(QPen(QColor("#6B7280"), 1.6))
        y = top + card_height / 2.0
        painter.drawLine(int(mutation_rect.right() + 4.0), int(y), int(hotspot_rect.left() - 4.0), int(y))
        painter.drawLine(int(hotspot_rect.right() + 4.0), int(y), int(bottleneck_rect.left() - 4.0), int(y))
        distance_value = hotspot.get(
            "region_distance", hotspot.get("distance", hotspot.get("mutation_distance"))
        )
        if distance_value is not None:
            distance = float(distance_value)
            painter.setPen(QColor("#4B5563"))
            painter.drawText(
                QRectF(mutation_rect.right(), top + 2.0, gap, 18.0),
                Qt.AlignCenter,
                f"{distance:.1f} Å",
            )

        remote_candidate = bool(hotspot.get("remote_candidate"))
        threshold = float(profile.get("remote_distance_threshold", 10.0) or 10.0)
        if mutations:
            painter.setPen(QColor("#166534") if remote_candidate else QColor("#6B7280"))
            status = (
                f"Distal perturbation (active-site distance ≥{threshold:.1f} Å) linked to a path response; causality is not inferred"
                if remote_candidate
                else "The active-site distal criterion is not met for the current perturbation"
            )
        else:
            painter.setPen(QColor("#6D28D9"))
            status = "Ensemble-state association; the visual comparison does not by itself establish causality"
        painter.drawText(QRectF(margin, 77.0, max(1.0, self.width() - 2.0 * margin), 21.0), Qt.AlignCenter, status)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._hotspot_rect.contains(event.position()):
            hotspot = self._selected_hotspot()
            if hotspot:
                self.hotspot_selected.emit(hotspot)
                event.accept()
                return
        super().mousePressEvent(event)


class DatasetComparePanel(QWidget):
    """Editable cross-dataset cluster mapping and residue comparison controls."""

    compare_requested = Signal()
    source_dataset_changed = Signal(str)
    target_datasets_changed = Signal(object)
    residue_threshold_changed = Signal(float)
    rmsd_threshold_changed = Signal(float)
    rmsd_max_threshold_changed = Signal(float)
    comparison_selected = Signal(object)
    mapping_lock_requested = Signal(object)
    motif_selected = Signal(object)
    evidence_view_requested = Signal()
    evidence_export_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._residue_change_profile: dict = {}
        # These legacy detail tables are permanently omitted from the focused
        # workflow. Keep compatibility attributes, but never populate or paint
        # them while hidden.
        self._legacy_detail_tables_enabled = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        title = QLabel("Aligned Dataset Comparison")
        title.setStyleSheet("font-weight:600;")
        layout.addWidget(title)
        self._discovery_case = QLabel("Discovery workflow")
        self._discovery_case.setWordWrap(True)
        self._discovery_case.setStyleSheet(
            "background:#F8FAFC;border:1px solid #CBD5E1;border-radius:7px;"
            "padding:7px;color:#334155;font-size:10px;"
        )
        layout.addWidget(self._discovery_case)
        self._discovery_stage_labels: list[QLabel] = []
        discovery_row = QHBoxLayout()
        discovery_row.setSpacing(4)
        for index, label in enumerate((
            "1 Overview",
            "2 Lock one path",
            "3 Locate replacement",
            "4 Verify in Observer",
        ), start=1):
            stage = QLabel(label)
            stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stage.setMinimumHeight(27)
            stage.setProperty("stage_index", index)
            discovery_row.addWidget(stage, 1)
            self._discovery_stage_labels.append(stage)
        layout.addLayout(discovery_row)
        self._discovery_stage = 1
        self._update_discovery_stage_styles()
        self._status = QLabel("Choose a source and target dataset, then build their cluster mappings.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#6b7280;font-size:11px;")
        layout.addWidget(self._status)

        self._available_datasets: list[dict] = []
        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        source_row.addWidget(QLabel("Source Dataset"))
        self._source_dataset_combo = QComboBox()
        self._source_dataset_combo.setMinimumContentsLength(18)
        self._source_dataset_combo.currentIndexChanged.connect(
            self._on_source_dataset_changed
        )
        source_row.addWidget(self._source_dataset_combo, 1)
        source_row.addWidget(QLabel("Target Dataset"))
        self._target_dataset_combo = QComboBox(self)
        self._target_dataset_combo.setMinimumContentsLength(18)
        self._target_dataset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._target_dataset_combo.currentIndexChanged.connect(
            self._on_target_dataset_changed
        )
        source_row.addWidget(self._target_dataset_combo, 1)
        layout.addLayout(source_row)

        # Compatibility controls remain as hidden attributes for older capture
        # scripts, but mapping no longer filters by residue similarity or path
        # scope.  The only automatic candidate-family rule is the calibrated
        # absolute RMSD margin shown below.
        self._residue_threshold_spin = QDoubleSpinBox(self)
        self._residue_threshold_spin.setRange(0.0, 100.0)
        self._residue_threshold_spin.setDecimals(1)
        self._residue_threshold_spin.setSingleStep(2.5)
        self._residue_threshold_spin.setValue(0.0)
        self._residue_threshold_spin.setSuffix("%")
        self._residue_threshold_spin.hide()
        self._path_keep_spin = QDoubleSpinBox(self)
        self._path_keep_spin.setRange(5.0, 100.0)
        self._path_keep_spin.setDecimals(1)
        self._path_keep_spin.setSingleStep(5.0)
        self._path_keep_spin.setValue(100.0)
        self._path_keep_spin.setSuffix("%")
        self._path_keep_spin.hide()

        rmsd_row = QHBoxLayout()
        rmsd_row.addWidget(QLabel("RMSD family margin"))
        self._rmsd_threshold_spin = QDoubleSpinBox()
        self._rmsd_threshold_spin.setRange(0.0, 10.0)
        self._rmsd_threshold_spin.setDecimals(2)
        self._rmsd_threshold_spin.setSingleStep(0.05)
        self._rmsd_threshold_spin.setValue(0.79)
        self._rmsd_threshold_spin.setSuffix(" Å")
        self._rmsd_threshold_spin.setToolTip(
            "A target cluster is selectable only when its center-path RMSD is no greater "
            "than the best RMSD plus this threshold."
        )
        self._rmsd_threshold_spin.editingFinished.connect(self._emit_rmsd_threshold_changed)
        rmsd_row.addWidget(self._rmsd_threshold_spin)
        rmsd_row.addWidget(QLabel("from the best match"))
        rmsd_row.addSpacing(12)
        rmsd_row.addWidget(QLabel("Maximum RMSD"))
        self._rmsd_max_spin = QDoubleSpinBox()
        self._rmsd_max_spin.setRange(0.10, 20.0)
        self._rmsd_max_spin.setDecimals(2)
        self._rmsd_max_spin.setSingleStep(0.10)
        self._rmsd_max_spin.setValue(3.00)
        self._rmsd_max_spin.setSuffix(" Å")
        self._rmsd_max_spin.setToolTip(
            "Absolute path-identity cutoff. If even the best target cluster exceeds "
            "this RMSD, the source path is reported as having no counterpart."
        )
        self._rmsd_max_spin.editingFinished.connect(self._emit_rmsd_max_threshold_changed)
        rmsd_row.addWidget(self._rmsd_max_spin)
        rmsd_row.addStretch(1)
        layout.addLayout(rmsd_row)

        self._mapping_rule = QLabel()
        self._mapping_rule.setWordWrap(True)
        self._mapping_rule.setStyleSheet(
            "background:#F0FDF4;border:1px solid #BBF7D0;border-radius:5px;"
            "padding:5px;font-size:10px;color:#166534;"
        )
        layout.addWidget(self._mapping_rule)
        self._update_mapping_rule_text()

        remote_row = QHBoxLayout()
        remote_row.addWidget(QLabel("Active-site remote threshold"))
        self._remote_distance_spin = QDoubleSpinBox()
        self._remote_distance_spin.setRange(0.0, 100.0)
        self._remote_distance_spin.setDecimals(1)
        self._remote_distance_spin.setSingleStep(1.0)
        self._remote_distance_spin.setValue(10.0)
        self._remote_distance_spin.setSuffix(" Å")
        self._remote_distance_spin.setToolTip(
            "Minimum aligned mutation C-alpha distance to the tunnel seed / active-site origin."
        )
        remote_row.addWidget(self._remote_distance_spin)
        remote_row.addStretch(1)
        layout.addLayout(remote_row)

        self._reference = QLabel("Source dataset: not selected")
        self._reference.setWordWrap(True)
        self._reference.setStyleSheet(
            "color:#111827;background:#EFF6FF;border:1px solid #BFDBFE;"
            "border-radius:6px;padding:6px;font-size:11px;font-weight:600;"
        )
        layout.addWidget(self._reference)

        buttons = QHBoxLayout()
        self._compare_button = QPushButton("Build / Recalculate Mappings")
        self._compare_button.setProperty("primary", True)
        self._compare_button.clicked.connect(self.compare_requested.emit)
        self._clear_button = QPushButton("Clear Highlight")
        self._clear_button.clicked.connect(self.clear_requested.emit)
        buttons.addWidget(self._compare_button)
        buttons.addWidget(self._clear_button)
        layout.addLayout(buttons)

        self._summary = QLabel("No comparison results")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-size:11px;color:#374151;")
        layout.addWidget(self._summary)

        self._candidate_table = QTableWidget(0, 5)
        self._candidate_table.setObjectName("DatasetCompareCandidates")
        self._candidate_table.setHorizontalHeaderLabels(
            ["Source Cluster", "Target Dataset", "Most Relevant Cluster", "RMSD", "Residue Similarity"]
        )
        self._candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._candidate_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._candidate_table.setAlternatingRowColors(True)
        self._candidate_table.verticalHeader().setVisible(False)
        candidate_header = self._candidate_table.horizontalHeader()
        candidate_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        candidate_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3, 4):
            candidate_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self._candidate_table.itemSelectionChanged.connect(self._on_candidate_selection_changed)
        self._candidate_table.cellClicked.connect(self._on_candidate_cell_clicked)
        layout.addWidget(self._candidate_table, 2)

        mapping_editor = QHBoxLayout()
        mapping_hint = QLabel("Choose a target in the Most Relevant Cluster column")
        mapping_hint.setStyleSheet("color:#6b7280;font-size:10px;")
        mapping_editor.addWidget(mapping_hint, 1)
        self._lock_mapping_button = QPushButton("Lock Mapping")
        self._lock_mapping_button.setCheckable(True)
        self._lock_mapping_button.setEnabled(False)
        self._lock_mapping_button.toggled.connect(self._on_mapping_lock_toggled)
        mapping_editor.addWidget(self._lock_mapping_button)
        layout.addLayout(mapping_editor)

        self._path_region_widget = QWidget(self)
        path_region_layout = QVBoxLayout(self._path_region_widget)
        path_region_layout.setContentsMargins(0, 0, 0, 0)
        path_region_layout.setSpacing(0)
        # Full-path replacement is the only exposed region view. Mount it
        # directly so a redundant one-tab frame/title row does not consume
        # Overview space.
        replacement_page = QWidget(self._path_region_widget)
        replacement_layout = QVBoxLayout(replacement_page)
        replacement_layout.setContentsMargins(4, 4, 4, 4)
        replacement_layout.setSpacing(5)
        replacement_hint = QLabel(
            "Red path sections have the strongest composition replacement. "
            "R1/R2/... labels identify replacement regions. Blue BN-Ref and orange BN-Target show "
            "the bottleneck median and interquartile span on the same arc-length axis."
        )
        replacement_hint.setWordWrap(True)
        replacement_hint.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        replacement_hint.setStyleSheet("color:#6b7280;font-size:10px;")
        replacement_layout.addWidget(replacement_hint)
        self._bottleneck_summary = QLabel("Bottleneck markers: not calculated")
        self._bottleneck_summary.setWordWrap(True)
        self._bottleneck_summary.setMinimumHeight(40)
        self._bottleneck_summary.setMaximumHeight(58)
        self._bottleneck_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._bottleneck_summary.setStyleSheet(
            "background:#F0F9FF;border:1px solid #BAE6FD;border-radius:5px;"
            "padding:5px;font-size:10px;color:#0C4A6E;"
        )
        self._replacement_track = ResidueReplacementTrack(self)
        self._replacement_track.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._replacement_track.hotspot_selected.connect(self._on_hotspot_selected)
        replacement_layout.addWidget(self._replacement_track)
        replacement_layout.addWidget(self._bottleneck_summary)
        self._hotspot_detail = QLabel("No strong residue-replacement hotspot detected.")
        self._hotspot_detail.setWordWrap(True)
        self._hotspot_detail.setTextFormat(Qt.TextFormat.RichText)
        # Reserve two lines so long lost/gained residue labels cannot be
        # clipped by the lower edge of the Overview workspace.
        self._hotspot_detail.setMinimumHeight(58)
        self._hotspot_detail.setMaximumHeight(80)
        self._hotspot_detail.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._hotspot_detail.setStyleSheet(
            "background:#F9FAFB;border:1px solid #E5E7EB;border-radius:5px;"
            "padding:5px;font-size:11px;color:#374151;"
        )
        replacement_layout.addWidget(self._hotspot_detail)
        self._combination_regions_page = QWidget(self._path_region_widget)
        self._combination_regions_page.setVisible(False)
        self._combination_regions_layout = QVBoxLayout(self._combination_regions_page)
        self._combination_regions_layout.setContentsMargins(4, 4, 4, 4)
        self._combination_regions_placeholder = QLabel(
            "Sync charts to display residue-combination regions."
        )
        self._combination_regions_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._combination_regions_placeholder.setStyleSheet(
            "color:#6B7280;font-size:11px;padding:16px;"
        )
        self._combination_regions_layout.addWidget(
            self._combination_regions_placeholder,
            1,
        )
        path_region_layout.addWidget(replacement_page, 1)
        layout.addWidget(self._path_region_widget)
        region_actions_widget = QWidget(self)
        region_actions = QHBoxLayout(region_actions_widget)
        region_actions.setContentsMargins(0, 0, 0, 0)
        self._inspect_strongest_button = QPushButton("Inspect Strongest Region")
        self._inspect_strongest_button.setProperty("primary", True)
        self._inspect_strongest_button.setToolTip(
            "Select the highest-scoring full-path residue replacement and open it in Residue Observer."
        )
        self._inspect_strongest_button.clicked.connect(self._inspect_strongest_hotspot)
        self._inspect_strongest_button.setEnabled(False)
        region_actions.addWidget(self._inspect_strongest_button)
        self._previous_region_button = QPushButton("Previous Region")
        self._previous_region_button.clicked.connect(lambda: self.step_hotspot(-1, emit=True))
        self._previous_region_button.setEnabled(False)
        region_actions.addWidget(self._previous_region_button)
        self._next_region_button = QPushButton("Next Region")
        self._next_region_button.clicked.connect(lambda: self.step_hotspot(1, emit=True))
        self._next_region_button.setEnabled(False)
        region_actions.addWidget(self._next_region_button)
        region_actions.addStretch(1)
        # The full-path replacement map itself is the region-selection surface;
        # the redundant strongest/previous/next button row stays available to
        # the implementation but is intentionally omitted from the workflow.
        region_actions_widget.setVisible(False)
        layout.addWidget(region_actions_widget)

        allosteric_label = QLabel("Remote-regulation evidence chain")
        allosteric_label.setStyleSheet("font-weight:600;font-size:11px;color:#374151;")
        allosteric_label.setVisible(False)
        layout.addWidget(allosteric_label)
        self._allosteric_strip = AllostericEvidenceStrip(self)
        self._allosteric_strip.hotspot_selected.connect(self._on_hotspot_selected)
        self._allosteric_strip.setVisible(False)
        layout.addWidget(self._allosteric_strip)
        self._allosteric_summary = QLabel(
            "Select a mapped baseline→variant comparison to connect mutation, path-local replacement, and gate response."
        )
        self._allosteric_summary.setWordWrap(True)
        self._allosteric_summary.setStyleSheet(
            "background:#FAF5FF;border:1px solid #E9D5FF;border-radius:5px;"
            "padding:5px;font-size:10px;color:#581C87;"
        )
        self._allosteric_summary.setVisible(False)
        layout.addWidget(self._allosteric_summary)
        evidence_actions_widget = QWidget(self)
        evidence_actions = QHBoxLayout(evidence_actions_widget)
        evidence_actions.setContentsMargins(0, 0, 0, 0)
        evidence_actions.addStretch(1)
        self._open_evidence_button = QPushButton("Open Evidence")
        self._open_evidence_button.setToolTip(
            "Inspect mapping robustness, ensemble consistency, remote distance, and sequential frame blocks."
        )
        self._open_evidence_button.clicked.connect(self.evidence_view_requested.emit)
        self._open_evidence_button.setEnabled(False)
        evidence_actions.addWidget(self._open_evidence_button)
        self._export_evidence_button = QPushButton("Export Evidence…")
        self._export_evidence_button.setToolTip(
            "Export the active mapping, path scope, replacement regions, mutation distances, and bottleneck response as JSON."
        )
        self._export_evidence_button.clicked.connect(self.evidence_export_requested.emit)
        self._export_evidence_button.setEnabled(False)
        evidence_actions.addWidget(self._export_evidence_button)
        # Evidence opens directly from a selected full-path region.  Retain the
        # actions internally for compatibility, while removing the extra row.
        evidence_actions_widget.setVisible(False)
        layout.addWidget(evidence_actions_widget)

        self._changes_label = QLabel("Path residue-combination changes (2/3/4 residues)")
        self._changes_label.setStyleSheet("font-weight:600;font-size:11px;color:#374151;")
        layout.addWidget(self._changes_label)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(["Size", "Residue combination", "Change", "Ref paths", "Target paths", "Delta", "Affected paths"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 2)
        self._changes_label.setVisible(False)
        self._table.setVisible(False)
        self._table.setUpdatesEnabled(False)

        self._geometry_label = QLabel("Conformational remodeling regions")
        self._geometry_label.setStyleSheet("font-weight:600;font-size:11px;color:#374151;")
        layout.addWidget(self._geometry_label)
        self._geometry_table = QTableWidget(0, 5)
        self._geometry_table.setHorizontalHeaderLabels(
            ["Path region", "Mean shift", "Peak shift", "Ref length", "Target length"]
        )
        self._geometry_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._geometry_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._geometry_table.setAlternatingRowColors(True)
        self._geometry_table.verticalHeader().setVisible(False)
        self._geometry_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._geometry_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._geometry_table, 1)
        self._geometry_label.setVisible(False)
        self._geometry_table.setVisible(False)
        self._geometry_table.setUpdatesEnabled(False)

        self._motif_label = QLabel("Residues responsible for each remodeling region")
        self._motif_label.setStyleSheet("font-weight:600;font-size:11px;color:#374151;")
        layout.addWidget(self._motif_label)
        self._motif_hint = QLabel(
            "Select a row to inspect the reference motif in the upper Residue Observer "
            "and the target motif in the lower Residue Observer."
        )
        self._motif_hint.setWordWrap(True)
        self._motif_hint.setStyleSheet("color:#6b7280;font-size:10px;")
        layout.addWidget(self._motif_hint)
        self._motif_table = QTableWidget(0, 6)
        self._motif_table.setHorizontalHeaderLabels(
            ["Region", "Size", "Reference motif", "Target motif", "Path shift", "Residues to inspect"]
        )
        self._motif_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._motif_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._motif_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._motif_table.setAlternatingRowColors(True)
        self._motif_table.verticalHeader().setVisible(False)
        self._motif_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._motif_table.horizontalHeader().setStretchLastSection(True)
        self._motif_table.itemSelectionChanged.connect(self._emit_selected_motif)
        layout.addWidget(self._motif_table, 2)
        self._motif_label.setVisible(False)
        self._motif_hint.setVisible(False)
        self._motif_table.setVisible(False)
        self._motif_table.setUpdatesEnabled(False)

    def set_combination_regions_widget(self, widget: QWidget | None) -> None:
        """Retain the hidden synchronized chart for backward compatibility."""
        if widget is None:
            return
        placeholder = getattr(self, "_combination_regions_placeholder", None)
        if placeholder is not None:
            self._combination_regions_layout.removeWidget(placeholder)
            placeholder.deleteLater()
            self._combination_regions_placeholder = None
        self._combination_regions_widget = widget
        self._combination_regions_layout.addWidget(widget, 1)

    def combination_regions_visible(self) -> bool:
        """Return whether the optional combination-region view is exposed."""
        return False

    def take_path_region_widget(self) -> QWidget:
        """Detach the linked path-region switch for hosting in Evidence Overview."""
        current_layout = self.layout()
        if current_layout is not None:
            current_layout.removeWidget(self._path_region_widget)
        return self._path_region_widget

    def _update_discovery_stage_styles(self) -> None:
        for index, label in enumerate(self._discovery_stage_labels, start=1):
            if index < self._discovery_stage:
                style = (
                    "background:#DCFCE7;border:1px solid #86EFAC;border-radius:6px;"
                    "padding:4px;color:#166534;font-size:12px;font-weight:600;"
                )
            elif index == self._discovery_stage:
                style = (
                    "background:#DBEAFE;border:2px solid #2563EB;border-radius:6px;"
                    "padding:3px;color:#1D4ED8;font-size:12px;font-weight:700;"
                )
            else:
                style = (
                    "background:#F3F4F6;border:1px solid #D1D5DB;border-radius:6px;"
                    "padding:4px;color:#6B7280;font-size:12px;"
                )
            label.setStyleSheet(style)

    def set_discovery_case(self, title: str = "", question: str = "") -> None:
        """Describe the active case without baking biological conclusions into the UI."""
        title = str(title or "Discovery workflow").strip()
        question = str(question or "").strip()
        self._discovery_case.setText(
            f"<b>{title}</b>" + (f"<br>{question}" if question else "")
        )
        self.set_discovery_stage(1)

    def set_discovery_stage(self, stage: int, detail: str = "") -> None:
        self._discovery_stage = max(1, min(4, int(stage)))
        self._update_discovery_stage_styles()
        if detail:
            self._status.setText(str(detail))

    def set_source_datasets(self, datasets: list[dict], *, selected_key: str = "") -> None:
        previous_source = self.source_dataset_key()
        previous_target = self.target_dataset_keys()[0] if self.target_dataset_keys() else ""
        self._available_datasets = [dict(dataset) for dataset in (datasets or [])]
        current_key = str(selected_key or self.source_dataset_key() or "")
        self._source_dataset_combo.blockSignals(True)
        self._source_dataset_combo.clear()
        for dataset in self._available_datasets:
            key = str(dataset.get("key") or "")
            if not key:
                continue
            label = str(dataset.get("prefix") or dataset.get("name") or key)
            self._source_dataset_combo.addItem(label, key)
        selected_index = self._source_dataset_combo.findData(current_key)
        if selected_index < 0 and self._source_dataset_combo.count():
            selected_index = 0
        if selected_index >= 0:
            self._source_dataset_combo.setCurrentIndex(selected_index)
        self._source_dataset_combo.blockSignals(False)
        source_changed = previous_source != self.source_dataset_key()
        self._refresh_target_dataset_choices(
            selected_key="" if source_changed else previous_target,
        )
        self._update_compare_button_enabled()

    def source_dataset_key(self) -> str:
        return str(self._source_dataset_combo.currentData() or "")

    def target_dataset_keys(self) -> list[str]:
        key = str(self._target_dataset_combo.currentData() or "")
        return [key] if key else []

    def _refresh_target_dataset_choices(
        self,
        *,
        selected_key: str = "",
    ) -> None:
        source_key = self.source_dataset_key()
        choices = [
            (
                str(dataset.get("prefix") or dataset.get("name") or dataset.get("key") or ""),
                str(dataset.get("key") or ""),
            )
            for dataset in self._available_datasets
            if str(dataset.get("key") or "") and str(dataset.get("key") or "") != source_key
        ]
        self._target_dataset_combo.blockSignals(True)
        self._target_dataset_combo.clear()
        for label, key in choices:
            self._target_dataset_combo.addItem(label, key)
        selected_index = self._target_dataset_combo.findData(str(selected_key or ""))
        if selected_index < 0 and self._target_dataset_combo.count():
            selected_index = 0
        if selected_index >= 0:
            self._target_dataset_combo.setCurrentIndex(selected_index)
        self._target_dataset_combo.blockSignals(False)

    def _on_source_dataset_changed(self, _index: int) -> None:
        self._refresh_target_dataset_choices()
        self._update_compare_button_enabled()
        self.source_dataset_changed.emit(self.source_dataset_key())

    def _on_target_dataset_changed(self, _index: int) -> None:
        self._update_compare_button_enabled()
        self.target_datasets_changed.emit(self.target_dataset_keys())

    def _update_compare_button_enabled(self) -> None:
        self._compare_button.setEnabled(
            self._source_dataset_combo.count() > 1 and bool(self.target_dataset_keys())
        )

    def residue_similarity_threshold(self) -> float:
        return 0.0

    def path_constraint_keep_ratio(self) -> float:
        return 1.0

    def rmsd_distance_threshold(self) -> float:
        return max(0.0, float(self._rmsd_threshold_spin.value()))

    def rmsd_max_threshold(self) -> float:
        return max(0.0, float(self._rmsd_max_spin.value()))

    def _update_mapping_rule_text(self) -> None:
        threshold = self.rmsd_distance_threshold()
        maximum = self.rmsd_max_threshold()
        self._mapping_rule.setText(
            f"Mapping rule: first require center-path RMSD ≤ {maximum:.2f} Å; then map "
            f"one source cluster to every valid target within best RMSD + {threshold:.2f} Å. "
            "If no target passes the maximum, the path has no counterpart."
        )

    def _emit_rmsd_threshold_changed(self) -> None:
        self._update_mapping_rule_text()
        self.rmsd_threshold_changed.emit(self.rmsd_distance_threshold())

    def _emit_rmsd_max_threshold_changed(self) -> None:
        self._update_mapping_rule_text()
        self.rmsd_max_threshold_changed.emit(self.rmsd_max_threshold())

    def remote_distance_threshold(self) -> float:
        return max(0.0, float(self._remote_distance_spin.value()))

    def set_reference_context(self, *, dataset_label: str = "", cluster_id: int | None = None, path_count: int | None = None, cluster_count: int | None = None):
        if not dataset_label:
            self._reference.setText("Source dataset: not selected")
            self._compare_button.setEnabled(False)
            return
        suffixes = []
        if cluster_count is not None:
            suffixes.append(f"{int(cluster_count):,} clusters")
        if cluster_id is not None:
            suffixes.append(f"current Cluster {int(cluster_id)}")
        if path_count is not None:
            suffixes.append(f"{int(path_count):,} paths")
        suffix = f" | {' | '.join(suffixes)}" if suffixes else ""
        self._reference.setText(f"Source: {dataset_label}{suffix}")
        self._update_compare_button_enabled()

    @staticmethod
    def _mapping_target_ids(payload: dict) -> tuple[int, ...]:
        target_ids = tuple(
            int(value) for value in (payload.get("target_cluster_ids", ()) or ()) if int(value) > 0
        )
        if not target_ids and int(payload.get("target_cluster_id", 0) or 0) > 0:
            target_ids = (int(payload["target_cluster_id"]),)
        return target_ids

    @staticmethod
    def _mapping_result_label(payload: dict) -> str:
        return str(payload.get("target_label") or payload.get("target_dataset") or "Target")

    def _mapping_metric_text(self, payload: dict, metric: str) -> str:
        """Format one target dataset while keeping cluster-level values inspectable."""
        target_ids = self._mapping_target_ids(payload)
        if not target_ids:
            return "—"
        options = {
            int(option.get("target_cluster_id", 0) or 0): dict(option)
            for option in payload.get("target_options", []) or []
            if isinstance(option, dict) and int(option.get("target_cluster_id", 0) or 0) > 0
        }
        values = []
        for cluster_id in target_ids:
            option = options.get(cluster_id, payload if cluster_id == target_ids[0] else {})
            if metric == "rmsd":
                value = option.get("center_rmsd")
                text = f"{float(value):.3f}" if value is not None else "n/a"
            else:
                value = option.get("residue_similarity")
                text = f"{float(value) * 100:.1f}%" if value is not None else "n/a"
            values.append(f"C{cluster_id} {text}")
        return "; ".join(values)

    def _refresh_candidate_row(self, row_index: int, payload: dict) -> None:
        target_ids = self._mapping_target_ids(payload)
        matched = bool(target_ids)
        values = (
            f"C{int(payload.get('reference_cluster_id', 0) or 0)}",
            self._mapping_result_label(payload),
            ", ".join(f"C{cluster_id}" for cluster_id in target_ids) if matched else "No match",
            self._mapping_metric_text(payload, "rmsd"),
            self._mapping_metric_text(payload, "residue"),
        )
        for column, value in enumerate(values):
            item = self._candidate_table.item(row_index, column)
            if item is None:
                item = QTableWidgetItem()
                self._candidate_table.setItem(row_index, column, item)
            item.setText(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(
                QBrush(QColor("#FEE2E2")) if not matched
                else QBrush(QColor("#DBEAFE")) if bool(payload.get("locked"))
                else QBrush()
            )
        source_item = self._candidate_table.item(row_index, 0)
        source_item.setData(Qt.ItemDataRole.UserRole, dict(payload))
        self._candidate_table.setRowHeight(row_index, 34)
        self._configure_target_cluster_cell(row_index, payload)

    def set_comparison_rows(self, rows: list[dict], *, selected: dict | None = None):
        self._candidate_table.blockSignals(True)
        self._candidate_table.setRowCount(0)
        selected_row = -1
        selected = dict(selected or {})
        for raw_row in rows or []:
            payload = dict(raw_row or {})
            index = self._candidate_table.rowCount()
            self._candidate_table.insertRow(index)
            self._refresh_candidate_row(index, payload)
            if (
                int(payload.get("reference_cluster_id", -2))
                == int(selected.get("reference_cluster_id", -1) or -1)
                and str(payload.get("target_dataset") or "")
                == str(selected.get("target_dataset") or "")
            ):
                selected_row = index
        self._candidate_table.blockSignals(False)
        if selected_row >= 0:
            self._candidate_table.selectRow(selected_row)
        elif self._candidate_table.rowCount():
            self._candidate_table.selectRow(0)
        else:
            self._populate_mapping_editor(None)
        if rows:
            self.set_discovery_stage(
                2,
                "Overview complete. Click Most Relevant Cluster to inspect or revise an RMSD-valid target-cluster family.",
            )

    def _configure_target_cluster_cell(self, row_index: int, payload: dict) -> None:
        """Keep the mapping inactive until the user explicitly clicks column 2."""
        item = self._candidate_table.item(row_index, 2)
        if item is None:
            return
        if payload.get("target_options"):
            item.setToolTip(
                "Click to select one or more RMSD-valid target clusters."
            )
            font = item.font()
            font.setUnderline(True)
            item.setFont(font)
            item.setForeground(QBrush(QColor("#2563EB")))
        else:
            item.setToolTip(
                f"No target cluster is within the {self.rmsd_max_threshold():.2f} Å maximum RMSD."
            )

    def _on_candidate_cell_clicked(self, row_index: int, column: int) -> None:
        item = self._candidate_table.item(row_index, 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(payload, dict):
            return
        self._candidate_table.selectRow(row_index)
        self._populate_mapping_editor(payload)

        # A real table click is the user's request to inspect this mapping.
        # Keep programmatic/default row selection passive so rebuilding the
        # table does not unexpectedly replace the current Overview.
        if column != 2:
            if self._mapping_target_ids(payload):
                self.comparison_selected.emit(dict(payload))
            return

        options = [dict(row) for row in payload.get("target_options", []) if isinstance(row, dict)]
        if not options:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Select mapped target clusters")
        dialog.setMinimumWidth(470)
        dialog_layout = QVBoxLayout(dialog)
        hint = QLabel(
            f"Only clusters satisfying RMSD ≤ {self.rmsd_max_threshold():.2f} Å and "
            f"RMSD ≤ best + {self.rmsd_distance_threshold():.2f} Å are shown. "
            "Select every cluster that represents the same source-path family."
        )
        hint.setWordWrap(True)
        dialog_layout.addWidget(hint)
        option_list = QListWidget(dialog)
        dialog_layout.addWidget(option_list)
        current_targets = set(self._mapping_target_ids(payload))
        for option in options:
            cluster_id = int(option.get("target_cluster_id", 0) or 0)
            similarity = option.get("residue_similarity")
            similarity_text = (
                f" | residues {float(similarity) * 100:.1f}%"
                if similarity is not None
                else " | residues n/a"
            )
            list_item = QListWidgetItem(
                f"C{cluster_id} | RMSD {float(option.get('center_rmsd', 0.0)):.3f}{similarity_text}"
            )
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            list_item.setCheckState(
                Qt.CheckState.Checked if cluster_id in current_targets else Qt.CheckState.Unchecked
            )
            list_item.setData(Qt.ItemDataRole.UserRole, option)
            option_list.addItem(list_item)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_options = []
        for index in range(option_list.count()):
            list_item = option_list.item(index)
            if list_item.checkState() == Qt.CheckState.Checked:
                option = list_item.data(Qt.ItemDataRole.UserRole)
                if isinstance(option, dict):
                    selected_options.append(dict(option))
        self._apply_target_options_to_row(row_index, selected_options)

    def _selected_candidate_payload(self) -> dict | None:
        rows = self._candidate_table.selectedItems()
        if not rows:
            return None
        item = self._candidate_table.item(rows[0].row(), 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item else None
        return dict(payload) if isinstance(payload, dict) else None

    def select_comparison(self, reference_cluster_id: int, target_dataset: str) -> bool:
        """Select a mapping row from another linked view."""
        for row_index in range(self._candidate_table.rowCount()):
            item = self._candidate_table.item(row_index, 0)
            payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(payload, dict):
                continue
            if (
                int(payload.get("reference_cluster_id", -1)) == int(reference_cluster_id)
                and str(payload.get("target_dataset") or "") == str(target_dataset or "")
            ):
                self._candidate_table.selectRow(row_index)
                self._candidate_table.scrollToItem(item)
                self._populate_mapping_editor(payload)
                if self._mapping_target_ids(payload):
                    self.comparison_selected.emit(dict(payload))
                return True
        return False

    def _populate_mapping_editor(self, payload: dict | None) -> None:
        payload = dict(payload or {})
        options = [dict(row) for row in payload.get("target_options", []) if isinstance(row, dict)]
        target_ids = tuple(payload.get("target_cluster_ids", ()) or ())
        enabled = bool(payload and options and target_ids)
        self._lock_mapping_button.blockSignals(True)
        self._lock_mapping_button.setChecked(bool(payload.get("locked")))
        self._lock_mapping_button.setText("Unlock Mapping" if bool(payload.get("locked")) else "Lock Mapping")
        self._lock_mapping_button.setEnabled(enabled)
        self._lock_mapping_button.blockSignals(False)

    def _on_candidate_selection_changed(self):
        """Row selection only changes table context; comparison starts from column 2."""
        payload = self._selected_candidate_payload()
        self._populate_mapping_editor(payload)

    def _apply_target_options_to_row(
        self,
        row_index: int,
        selected_options: list[dict],
    ) -> None:
        item = self._candidate_table.item(row_index, 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(payload, dict):
            return
        self._candidate_table.selectRow(row_index)
        payload = dict(payload)
        selected_options = sorted(
            (dict(option) for option in selected_options if isinstance(option, dict)),
            key=lambda option: (
                float(option.get("center_rmsd", float("inf"))),
                int(option.get("target_cluster_id", 0)),
            ),
        )
        selected_ids = tuple(int(option.get("target_cluster_id", 0)) for option in selected_options)
        if selected_options:
            payload.update(selected_options[0])
            payload["target_cluster_id"] = selected_ids[0]
        else:
            payload["target_cluster_id"] = None
        payload["target_cluster_ids"] = selected_ids
        payload["locked"] = False
        payload["matching_method"] = "manual multi-select" if selected_ids else "No match"
        self._refresh_candidate_row(row_index, payload)
        self._lock_mapping_button.blockSignals(True)
        self._lock_mapping_button.setChecked(False)
        self._lock_mapping_button.setText("Lock Mapping")
        self._lock_mapping_button.setEnabled(bool(selected_ids))
        self._lock_mapping_button.blockSignals(False)
        self.comparison_selected.emit(dict(payload))

    def _on_mapping_lock_toggled(self, checked: bool) -> None:
        payload = self._selected_candidate_payload()
        if not payload:
            return
        payload["locked"] = bool(checked)
        self._lock_mapping_button.setText("Unlock Mapping" if checked else "Lock Mapping")
        self.mapping_lock_requested.emit(payload)

    def set_current_mapping_locked(self, locked: bool) -> None:
        """Update only the selected row; the lock preserves its target-cluster set."""
        row_index = self._candidate_table.currentRow()
        payload = self._selected_candidate_payload()
        if row_index < 0 or not payload:
            return
        item = self._candidate_table.item(row_index, 0)
        if item is None:
            return
        payload["locked"] = bool(locked)
        self._refresh_candidate_row(row_index, payload)
        self._lock_mapping_button.blockSignals(True)
        self._lock_mapping_button.setChecked(bool(locked))
        self._lock_mapping_button.setText("Unlock Mapping" if locked else "Lock Mapping")
        self._lock_mapping_button.blockSignals(False)
        if locked:
            self.set_discovery_stage(
                3,
                "Mapping locked. Use the full-path map to locate the strongest residue replacement and its bottleneck context.",
            )

    def set_selected_critical_cluster(self, cluster_ids, matched_path_count: int) -> None:
        """Show the complete target-cluster family used by the comparison."""
        row_index = self._candidate_table.currentRow()
        if row_index < 0:
            return
        if isinstance(cluster_ids, int):
            cluster_ids = (cluster_ids,)
        cluster_ids = tuple(int(value) for value in (cluster_ids or ()) if int(value) > 0)
        if not cluster_ids:
            return
        labels = ", ".join(f"C{cluster_id}" for cluster_id in cluster_ids)
        item = self._candidate_table.item(row_index, 2)
        if item is not None:
            item.setText(labels)
            item.setToolTip(
                f"Selected target family {labels}: all {int(matched_path_count):,} paths. "
                "Click to change the multi-cluster selection."
            )

    def _emit_selected_motif(self):
        rows = self._motif_table.selectedItems()
        if not rows:
            return
        item = self._motif_table.item(rows[0].row(), 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(payload, dict):
            self.motif_selected.emit(payload)

    def _set_hotspot_detail(self, hotspot: dict | None):
        hotspot = dict(hotspot or {})
        if not hotspot:
            self._hotspot_detail.setText("No strong residue-replacement hotspot detected.")
            return
        lost = ", ".join(
            f"Seq {int(seq_id)} (MD {int(seq_id) - 1})"
            for seq_id in hotspot.get("lost_sequence_ids", ()) or ()
        ) or "none"
        gained = ", ".join(
            f"Seq {int(seq_id)} (MD {int(seq_id) - 1})"
            for seq_id in hotspot.get("gained_sequence_ids", ()) or ()
        ) or "none"
        start = float(hotspot.get("start_fraction", 0.0)) * 100.0
        end = float(hotspot.get("end_fraction", 0.0)) * 100.0
        score = float(hotspot.get("max_score", 0.0)) * 100.0
        self._hotspot_detail.setText(
            f"<b>{hotspot.get('region_id', hotspot.get('hotspot_id', 'Region'))}</b> | physical arc {start:.0f}–{end:.0f}% | "
            f"replacement {score:.0f}% &nbsp; "
            f"<span style='color:#2563EB'><b>Reference/lost:</b> {lost}</span> &nbsp; "
            f"<span style='color:#EA580C'><b>Target/gained:</b> {gained}</span>"
        )

    def _evidence_for_hotspot(self, hotspot: dict | None) -> dict:
        hotspot = dict(hotspot or {})
        region_id = str(hotspot.get("region_id") or hotspot.get("hotspot_id") or "")
        evidence = [
            dict(row) for row in self._residue_change_profile.get("allosteric_evidence", []) or []
        ]
        matched = next((
            row for row in evidence
            if region_id and str(row.get("region_id") or row.get("hotspot_id") or "") == region_id
        ), None)
        if matched:
            return matched
        return hotspot

    def _update_allosteric_summary(self, hotspot: dict | None) -> None:
        profile = dict(self._residue_change_profile or {})
        substitutions = [dict(row) for row in profile.get("substitutions", []) or []]
        evidence = self._evidence_for_hotspot(hotspot)
        if not substitutions:
            perturbation_label = str(profile.get("perturbation_label") or "").strip()
            if perturbation_label:
                self._allosteric_summary.setText(
                    f"{perturbation_label} is treated as an ensemble-state perturbation. "
                    "The path-local replacement and gate response are associated observations; causal transmission requires independent evidence."
                )
            else:
                self._allosteric_summary.setText(
                    "No explicit amino-acid substitution was detected from the loaded residue identities."
                )
            return
        region_distance = evidence.get(
            "region_distance", evidence.get("distance", evidence.get("mutation_distance"))
        )
        active_site_distance = evidence.get("active_site_distance")
        if region_distance is None:
            mutation_text = ", ".join(str(row.get("label") or "mutation") for row in substitutions)
            self._allosteric_summary.setText(
                f"Detected {mutation_text}, but aligned coordinates are not yet available for the distance audit."
            )
            return
        mutation_label = str(
            evidence.get("mutation_label")
            or evidence.get("nearest_mutation_label")
            or substitutions[0].get("label")
            or "Mutation"
        )
        region_id = str(evidence.get("region_id") or evidence.get("hotspot_id") or "selected region")
        replacement = float(
            evidence.get("replacement_score", evidence.get("max_score", 0.0)) or 0.0
        )
        qualifier = (
            "distal perturbation / path-response association candidate"
            if bool(evidence.get("remote_candidate"))
            else "active-site distal criterion not met"
        )
        active_text = (
            f"is {float(active_site_distance):.1f} Å from the active-site origin"
            if active_site_distance is not None
            else "has no available active-site distance"
        )
        self._allosteric_summary.setText(
            f"{mutation_label} {active_text}; its linked {region_id} response is "
            f"{float(region_distance):.1f} Å from the mutation; "
            f"path-local replacement {replacement * 100:.0f}%. "
            f"Interpretation: {qualifier}; causal transmission requires independent trajectory/contact evidence."
        )

    def _sync_selected_hotspot(self, hotspot: dict | None) -> None:
        hotspot = dict(hotspot or {})
        self._set_hotspot_detail(hotspot)
        region_id = str(hotspot.get("region_id") or hotspot.get("hotspot_id") or "")
        self._allosteric_strip.set_selected_region(region_id)
        self._update_allosteric_summary(hotspot)

    def _on_hotspot_selected(self, hotspot: dict):
        self._sync_selected_hotspot(hotspot)
        self.set_discovery_stage(
            4,
            "Replacement region selected. Residue Observer now validates the baseline and target microenvironments side by side.",
        )
        self.motif_selected.emit(dict(hotspot or {}))

    def _inspect_strongest_hotspot(self) -> None:
        hotspot = self._replacement_track.select_hotspot(0, emit=False)
        if hotspot:
            self._on_hotspot_selected(hotspot)

    def select_hotspot(self, region_id: str, *, emit: bool = False) -> dict | None:
        hotspot = self._replacement_track.select_hotspot_by_id(region_id, emit=False)
        self._sync_selected_hotspot(hotspot)
        if emit and hotspot:
            self._on_hotspot_selected(hotspot)
        return hotspot

    def step_hotspot(self, delta: int, *, emit: bool = False) -> dict | None:
        hotspot = self._replacement_track.step_hotspot(delta, emit=False)
        self._sync_selected_hotspot(hotspot)
        if emit and hotspot:
            self._on_hotspot_selected(hotspot)
        return hotspot

    def set_result(
        self,
        *,
        summary: str,
        changes: list[dict],
        geometry_regions: list[dict] | None = None,
        remodeling_changes: list[dict] | None = None,
        motif_transitions: list[dict] | None = None,
        residue_change_profile: dict | None = None,
        reference_dataset_key: str = "",
        target_dataset_key: str = "",
    ):
        self._summary.setText(summary)
        profile = dict(residue_change_profile or {})
        enriched_hotspots = []
        for hotspot in profile.get("hotspots", []) or []:
            payload = dict(hotspot or {})
            payload.setdefault("reference_dataset_key", str(reference_dataset_key or ""))
            payload.setdefault("target_dataset_key", str(target_dataset_key or ""))
            payload.setdefault("present_reference", bool(payload.get("reference_residue_ids")))
            payload.setdefault("present_target", bool(payload.get("target_residue_ids")))
            enriched_hotspots.append(payload)
        profile["hotspots"] = enriched_hotspots
        self._residue_change_profile = profile
        self._replacement_track.set_profile(profile)
        self._allosteric_strip.set_profile(profile)
        has_hotspots = bool(enriched_hotspots)
        self._inspect_strongest_button.setEnabled(has_hotspots)
        self._previous_region_button.setEnabled(len(enriched_hotspots) > 1)
        self._next_region_button.setEnabled(len(enriched_hotspots) > 1)
        self._export_evidence_button.setEnabled(bool(profile.get("bins")))
        self._open_evidence_button.setEnabled(bool(profile.get("bins")))
        self._sync_selected_hotspot(self._replacement_track.selected_hotspot())
        def bottleneck_text(label: str, payload: dict) -> str:
            payload = dict(payload or {})
            if not payload:
                return f"{label}: unavailable"
            fraction = float(payload.get("fraction", 0.0)) * 100.0
            q25 = float(payload.get("fraction_q25", payload.get("fraction", 0.0))) * 100.0
            q75 = float(payload.get("fraction_q75", payload.get("fraction", 0.0))) * 100.0
            radius = float(payload.get("radius", 0.0) or 0.0)
            return f"{label}: arc {fraction:.1f}% (IQR {q25:.1f}–{q75:.1f}%), median radius {radius:.2f} Å"
        self._bottleneck_summary.setText(
            bottleneck_text("BN-Ref", profile.get("reference_bottleneck", {}))
            + "  |  "
            + bottleneck_text("BN-Target", profile.get("target_bottleneck", {}))
        )
        if not self._legacy_detail_tables_enabled:
            # The visible workflow ends at the full-path map. Do not construct
            # QTableWidgetItems for the retired change/geometry/motif tables.
            self._status.setText(
                str(profile.get("scope_summary") or "")
                + (" | " if profile.get("scope_summary") else "")
                + f"{len(enriched_hotspots)} residue-replacement regions loaded. "
                "Click R1/R2/... to inspect Reference and Target residues."
            )
            if enriched_hotspots:
                self.set_discovery_stage(
                    3,
                    "Path comparison ready. Select R1/R2/... to inspect the largest residue replacement in structural context.",
                )
            return
        self._table.setRowCount(0)
        for row in changes or []:
            index = self._table.rowCount()
            self._table.insertRow(index)
            values = (
                row.get("combination_size", ""),
                row.get("residue_label", row.get("combination_label", "")),
                row.get("display_status", row.get("status", "")),
                f"{float(row.get('reference_frequency', 0.0)) * 100:.1f}%",
                f"{float(row.get('target_frequency', 0.0)) * 100:.1f}%",
                f"{float(row.get('frequency_delta', 0.0)) * 100:+.1f}%",
                f"{int(row.get('reference_path_count', 0)):,} -> {int(row.get('target_path_count', 0)):,}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                status = str(row.get("status", ""))
                if status == "added_in_target":
                    item.setBackground(QBrush(QColor("#DCFCE7")))
                elif status == "removed_in_target":
                    item.setBackground(QBrush(QColor("#FEE2E2")))
                elif status in {"replacement", "shared"} and (status == "replacement" or abs(float(row.get("frequency_delta", 0.0) or 0.0)) >= 0.05):
                    item.setBackground(QBrush(QColor("#FEF3C7")))
                self._table.setItem(index, column, item)
        self._status.setText(
            str(profile.get("scope_summary") or "") +
            (" | " if profile.get("scope_summary") else "") +
            f"{len(enriched_hotspots)} residue-replacement regions loaded. "
            "Click R1/R2/... to inspect Reference and Target residues."
        )
        if enriched_hotspots:
            self.set_discovery_stage(
                3,
                "Path comparison ready. Select R1/R2/... to inspect the largest residue replacement in structural context.",
            )
        self._geometry_table.setRowCount(0)
        for row in geometry_regions or []:
            index = self._geometry_table.rowCount()
            self._geometry_table.insertRow(index)
            values = (
                f"{float(row.get('start_fraction', 0.0)) * 100:.0f}% - {float(row.get('end_fraction', 0.0)) * 100:.0f}%",
                f"{float(row.get('mean_displacement', 0.0)):.3f} A",
                f"{float(row.get('max_displacement', 0.0)):.3f} A",
                f"{float(row.get('reference_length', 0.0)):.3f} A",
                f"{float(row.get('target_length', 0.0)):.3f} A",
            )
            for column, value in enumerate(values):
                self._geometry_table.setItem(index, column, QTableWidgetItem(str(value)))
        self._motif_table.setRowCount(0)
        transition_rows = motif_transitions or remodeling_changes or []
        for row in transition_rows:
            index = self._motif_table.rowCount()
            self._motif_table.insertRow(index)
            reference_ids = row.get("reference_residue_ids")
            target_ids = row.get("target_residue_ids")
            if reference_ids is not None or target_ids is not None:
                residue_display = f"{';'.join(map(str, reference_ids or ())) } -> { ';'.join(map(str, target_ids or ())) }"
                reference_motif = row.get("reference_motif", "")
                target_motif = row.get("target_motif", "")
            else:
                residue_display = row.get("residue_label", "")
                reference_motif = row.get("reference_motif", "")
                target_motif = row.get("target_motif", "")
            values = (
                f"{float(row.get('start_fraction', 0.0)) * 100:.0f}% - {float(row.get('end_fraction', 0.0)) * 100:.0f}%",
                row.get("combination_size", len(reference_ids or target_ids or ())),
                reference_motif,
                target_motif,
                f"{float(row.get('frequency_delta', 0.0)) * 100:+.1f}%",
                residue_display,
            )
            payload = dict(row)
            payload.setdefault("reference_dataset_key", str(reference_dataset_key or ""))
            payload.setdefault("target_dataset_key", str(target_dataset_key or ""))
            payload.setdefault("present_reference", bool(reference_ids))
            payload.setdefault("present_target", bool(target_ids))
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, payload)
                if bool(row.get("identity_change")) or str(row.get("transition_type", "")) in {"replacement", "rewired"}:
                    item.setBackground(QBrush(QColor("#FEE2E2")))
                else:
                    item.setBackground(QBrush(QColor("#FEF3C7")))
                self._motif_table.setItem(index, column, item)

    def clear_result(self):
        self._residue_change_profile = {}
        self._summary.setText("No comparison results")
        self._candidate_table.setRowCount(0)
        self._populate_mapping_editor(None)
        if self._legacy_detail_tables_enabled:
            self._table.setRowCount(0)
            self._geometry_table.setRowCount(0)
            self._motif_table.setRowCount(0)
        self._replacement_track.clear()
        self._allosteric_strip.clear()
        self._inspect_strongest_button.setEnabled(False)
        self._previous_region_button.setEnabled(False)
        self._next_region_button.setEnabled(False)
        self._allosteric_summary.setText(
            "Select a mapped baseline→variant comparison to connect mutation, path-local replacement, and gate response."
        )
        self._export_evidence_button.setEnabled(False)
        self._open_evidence_button.setEnabled(False)
        self._bottleneck_summary.setText("Bottleneck markers: not calculated")
        self._set_hotspot_detail(None)
        self._reference.setText("Source dataset: not selected")
        self._status.setText("Comparison highlight cleared.")
        self.set_discovery_stage(1)
