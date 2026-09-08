"""Linked evidence workspace for a selected dataset comparison."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QBrush, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)


class MappingStabilityMatrix(QWidget):
    """Threshold × evidence-weight matrix for one selected relation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict = {}
        self.setMinimumHeight(290)
        self.setMouseTracking(True)

    def set_payload(self, payload: dict | None) -> None:
        self._payload = dict(payload or {})
        self.update()

    def _geometry(self):
        thresholds = list(self._payload.get("thresholds", []) or [])
        scenarios = list(self._payload.get("scenarios", []) or [])
        left, top, right, bottom = 76.0, 42.0, 20.0, 54.0
        width = max(1.0, self.width() - left - right)
        height = max(1.0, self.height() - top - bottom)
        cell_w = width / max(1, len(scenarios))
        cell_h = height / max(1, len(thresholds))
        return thresholds, scenarios, left, top, cell_w, cell_h

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        thresholds, scenarios, left, top, cell_w, cell_h = self._geometry()
        if not thresholds or not scenarios:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a mapped cluster relation to test robustness")
            return
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(left, 4.0, self.width() - left, 24.0), Qt.AlignmentFlag.AlignCenter, "Selected mapping under threshold and evidence-weight perturbations")
        cells = {
            (float(row.get("threshold", 0.0)), str(row.get("scenario", ""))): dict(row)
            for row in self._payload.get("cells", []) or []
        }
        for col, scenario in enumerate(scenarios):
            painter.setPen(QColor("#374151"))
            painter.drawText(QRectF(left + col * cell_w, 25.0, cell_w, 18.0), Qt.AlignmentFlag.AlignCenter, scenario)
        for row_index, threshold in enumerate(thresholds):
            painter.setPen(QColor("#4B5563"))
            painter.drawText(QRectF(4.0, top + row_index * cell_h, left - 10.0, cell_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"≥ {threshold * 100:.0f}%")
            for col, scenario in enumerate(scenarios):
                cell = cells.get((float(threshold), str(scenario)), {})
                rect = QRectF(left + col * cell_w + 2.0, top + row_index * cell_h + 2.0, max(1.0, cell_w - 4.0), max(1.0, cell_h - 4.0))
                if bool(cell.get("top")):
                    fill, text, label = QColor("#2563EB"), QColor("#FFFFFF"), "#1"
                elif bool(cell.get("eligible")):
                    fill, text, label = QColor("#BFDBFE"), QColor("#1E3A8A"), f"#{int(cell.get('rank', 0))}"
                else:
                    fill, text, label = QColor("#F3F4F6"), QColor("#9CA3AF"), "—"
                painter.setPen(QPen(QColor("#D1D5DB"), 0.8))
                painter.setBrush(fill)
                painter.drawRoundedRect(rect, 4.0, 4.0)
                painter.setPen(text)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        eligible = int(self._payload.get("eligible_count", 0) or 0)
        top_count = int(self._payload.get("top_count", 0) or 0)
        total = int(self._payload.get("total_count", 0) or 0)
        painter.setPen(QColor("#374151"))
        painter.drawText(QRectF(left, self.height() - 43.0, self.width() - left - 8.0, 20.0), Qt.AlignmentFlag.AlignCenter, f"Eligible {eligible}/{total} · ranked first {top_count}/{total}")
        painter.setPen(QColor("#6B7280"))
        painter.drawText(QRectF(left, self.height() - 23.0, self.width() - left - 8.0, 18.0), Qt.AlignmentFlag.AlignCenter, "#1 = selected relation remains first · #n = eligible at rank n · — = below threshold")

    def mouseMoveEvent(self, event):
        thresholds, scenarios, left, top, cell_w, cell_h = self._geometry()
        col = int((event.position().x() - left) // max(1.0, cell_w))
        row = int((event.position().y() - top) // max(1.0, cell_h))
        if 0 <= row < len(thresholds) and 0 <= col < len(scenarios):
            cell = next((dict(item) for item in self._payload.get("cells", []) or [] if float(item.get("threshold", -1.0)) == float(thresholds[row]) and str(item.get("scenario", "")) == str(scenarios[col])), {})
            automatic = cell.get("automatic_cluster")
            status = f"selected rank {cell.get('rank')}" if cell.get("eligible") else "selected relation below threshold"
            QToolTip.showText(event.globalPosition().toPoint(), f"Threshold {thresholds[row] * 100:.0f}% · {scenarios[col]}\n{status}\nAutomatic first: C{automatic if automatic is not None else 'none'}", self)
        super().mouseMoveEvent(event)


class EnsembleConsistencyPlot(QWidget):
    """One point per target dataset/trajectory-like ensemble member."""

    dataset_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: list[dict] = []
        self._point_hits: list[tuple[QRectF, dict]] = []
        self.setMinimumHeight(360)
        self.setMouseTracking(True)

    def set_points(self, points: list[dict] | None) -> None:
        self._points = [dict(row) for row in (points or [])]
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._point_hits = []
        if not self._points:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load multiple mapped datasets to inspect ensemble consistency")
            return
        left, top, right, bottom = 74.0, 34.0, 28.0, 56.0
        plot = QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))
        xs = [float(row["center_rmsd"]) for row in self._points]
        ys = [float(row["residue_similarity"]) * 100.0 for row in self._points]
        x_min, x_max = min(0.0, min(xs)), max(xs)
        x_pad = max(0.25, (x_max - x_min) * 0.12)
        x_max += x_pad
        y_min, y_max = max(0.0, min(ys) - 8.0), min(100.0, max(ys) + 8.0)
        if y_max - y_min < 20.0:
            mid = (y_max + y_min) / 2.0
            y_min, y_max = max(0.0, mid - 10.0), min(100.0, mid + 10.0)

        def px(value):
            return plot.left() + (float(value) - x_min) / max(1e-9, x_max - x_min) * plot.width()

        def py(value):
            return plot.bottom() - (float(value) - y_min) / max(1e-9, y_max - y_min) * plot.height()

        painter.setPen(QPen(QColor("#D1D5DB"), 1.0))
        painter.setBrush(QColor("#F9FAFB"))
        painter.drawRect(plot)
        for index in range(5):
            x_value = x_min + (x_max - x_min) * index / 4.0
            y_value = y_min + (y_max - y_min) * index / 4.0
            painter.setPen(QPen(QColor("#E5E7EB"), 0.8))
            painter.drawLine(int(px(x_value)), int(plot.top()), int(px(x_value)), int(plot.bottom()))
            painter.drawLine(int(plot.left()), int(py(y_value)), int(plot.right()), int(py(y_value)))
            painter.setPen(QColor("#6B7280"))
            painter.drawText(QRectF(px(x_value) - 24.0, plot.bottom() + 4.0, 48.0, 18.0), Qt.AlignmentFlag.AlignCenter, f"{x_value:.1f}")
            painter.drawText(QRectF(4.0, py(y_value) - 9.0, left - 12.0, 18.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{y_value:.0f}")
        painter.setPen(QColor("#374151"))
        painter.drawText(QRectF(plot.left(), self.height() - 28.0, plot.width(), 20.0), Qt.AlignmentFlag.AlignCenter, "Center-path RMSD (Å)")
        painter.save()
        painter.translate(18.0, plot.center().y())
        painter.rotate(-90.0)
        painter.drawText(QRectF(-plot.height() / 2.0, -12.0, plot.height(), 20.0), Qt.AlignmentFlag.AlignCenter, "Residue similarity (%)")
        painter.restore()
        for index, row in enumerate(self._points):
            x, y = px(row["center_rmsd"]), py(float(row["residue_similarity"]) * 100.0)
            color = QColor("#2563EB") if bool(row.get("locked")) else QColor("#F97316")
            painter.setPen(QPen(QColor("#1F2937"), 1.0))
            painter.setBrush(color)
            if bool(row.get("pipeline_match")):
                painter.drawRect(QRectF(x - 5.0, y - 5.0, 10.0, 10.0))
            else:
                painter.drawEllipse(QRectF(x - 5.5, y - 5.5, 11.0, 11.0))
            label = f"{row.get('dataset_label', 'target')} / C{int(row.get('target_cluster_id', 0))}"
            align_left = x < plot.center().x()
            label_rect = QRectF(x + (8.0 if align_left else -168.0), y - 18.0, 160.0, 18.0)
            painter.setPen(QColor("#374151"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft if align_left else Qt.AlignmentFlag.AlignRight, label)
            self._point_hits.append((QRectF(x - 12.0, y - 12.0, 24.0, 24.0), row))

    def mouseMoveEvent(self, event):
        for rect, row in self._point_hits:
            if rect.contains(event.position()):
                QToolTip.showText(event.globalPosition().toPoint(), f"{row.get('dataset_label')} · Cluster {row.get('target_cluster_id')}\nResidue similarity {float(row.get('residue_similarity', 0.0)) * 100:.1f}%\nCenter RMSD {float(row.get('center_rmsd', 0.0)):.3f} Å", self)
                break
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            for rect, row in self._point_hits:
                if rect.contains(event.position()):
                    self.dataset_selected.emit(str(row.get("dataset_key") or ""))
                    event.accept()
                    return
        super().mousePressEvent(event)


class CompositionEvidenceView(QWidget):
    """Signed residue-frequency changes for the selected path region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict = {}
        self.setMinimumHeight(360)

    def set_payload(self, payload: dict | None) -> None:
        self._payload = dict(payload or {})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rows = [dict(row) for row in self._payload.get("rows", []) or []]
        if not rows:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                str(self._payload.get("note") or "Select R1/R2/... to inspect residue composition replacement."),
            )
            return
        # Reserve independent columns for residue labels and negative values;
        # otherwise long sequence labels collide with values near the maximum
        # depleted bar in publication-sized captures.
        left, top, right, bottom = 220.0, 44.0, 82.0, 56.0
        plot = QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))
        row_h = plot.height() / max(1, len(rows))
        center_x = plot.center().x()
        half_w = plot.width() / 2.0
        max_abs = max(0.05, max(abs(float(row.get("delta", 0.0))) for row in rows))
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(left, 5.0, plot.width(), 24.0), Qt.AlignmentFlag.AlignCenter, "Selected-region residue replacement (target − baseline frequency)")
        painter.setPen(QPen(QColor("#9CA3AF"), 1.0))
        painter.drawLine(int(center_x), int(plot.top()), int(center_x), int(plot.bottom()))
        for index, row in enumerate(rows):
            y = plot.top() + index * row_h
            delta = float(row.get("delta", 0.0))
            width = abs(delta) / max_abs * max(1.0, half_w - 8.0)
            rect = QRectF(center_x - width if delta < 0 else center_x, y + row_h * 0.18, width, row_h * 0.64)
            color = QColor("#2563EB") if delta < 0 else QColor("#DC2626")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setPen(QColor("#374151"))
            painter.drawText(QRectF(4.0, y, left - 68.0, row_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(row.get("label") or row.get("residue_id") or "Residue"))
            value_rect = QRectF(rect.left() - 58.0 if delta < 0 else rect.right() + 5.0, y, 54.0, row_h)
            painter.setPen(color)
            value_alignment = (
                Qt.AlignmentFlag.AlignRight if delta < 0 else Qt.AlignmentFlag.AlignLeft
            ) | Qt.AlignmentFlag.AlignVCenter
            painter.drawText(value_rect, value_alignment, f"{delta:+.2f}")
        painter.setPen(QColor("#6B7280"))
        painter.drawText(QRectF(left, self.height() - 42.0, plot.width(), 18.0), Qt.AlignmentFlag.AlignCenter, "blue = depleted from baseline · red = enriched in target")
        painter.drawText(QRectF(8.0, self.height() - 23.0, self.width() - 16.0, 18.0), Qt.AlignmentFlag.AlignCenter, str(self._payload.get("note") or ""))


class BottleneckEvidenceView(QWidget):
    """Population bottleneck position and radius with position IQR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict = {}
        self.setMinimumHeight(360)

    def set_payload(self, payload: dict | None) -> None:
        self._payload = dict(payload or {})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        ref = dict(self._payload.get("reference") or {})
        target = dict(self._payload.get("target") or {})
        if not ref or not target:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Bottleneck population summaries are unavailable for this relation.")
            return
        left, top, right, bottom = 72.0, 38.0, 26.0, 64.0
        plot = QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))
        radius_values = [float(row.get("radius", 0.0) or 0.0) for row in (ref, target)]
        y_min = max(0.0, min(radius_values) - 0.25)
        y_max = max(radius_values) + 0.25

        def px(value):
            return plot.left() + max(0.0, min(1.0, float(value))) * plot.width()

        def py(value):
            return plot.bottom() - (float(value) - y_min) / max(1e-9, y_max - y_min) * plot.height()

        painter.setPen(QPen(QColor("#D1D5DB"), 1.0))
        painter.setBrush(QColor("#F9FAFB"))
        painter.drawRect(plot)
        for index in range(6):
            fraction = index / 5.0
            x = px(fraction)
            painter.setPen(QPen(QColor("#E5E7EB"), 0.8))
            painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))
            painter.setPen(QColor("#6B7280"))
            painter.drawText(QRectF(x - 24.0, plot.bottom() + 4.0, 48.0, 18.0), Qt.AlignmentFlag.AlignCenter, f"{fraction * 100:.0f}")
        for index in range(5):
            radius = y_min + (y_max - y_min) * index / 4.0
            y = py(radius)
            painter.setPen(QPen(QColor("#E5E7EB"), 0.8))
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
            painter.setPen(QColor("#6B7280"))
            painter.drawText(QRectF(3.0, y - 9.0, left - 10.0, 18.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{radius:.2f}")
        for row, color_value, label in ((ref, "#2563EB", str(self._payload.get("reference_label") or "Baseline")), (target, "#EA580C", str(self._payload.get("target_label") or "Target"))):
            fraction = float(row.get("fraction", 0.0))
            q25 = float(row.get("fraction_q25", fraction))
            q75 = float(row.get("fraction_q75", fraction))
            radius = float(row.get("radius", 0.0))
            x, y = px(fraction), py(radius)
            color = QColor(color_value)
            painter.setPen(QPen(color, 3.0))
            painter.drawLine(int(px(q25)), int(y), int(px(q75)), int(y))
            painter.drawLine(int(px(q25)), int(y - 6.0), int(px(q25)), int(y + 6.0))
            painter.drawLine(int(px(q75)), int(y - 6.0), int(px(q75)), int(y + 6.0))
            painter.setBrush(color)
            painter.drawEllipse(QRectF(x - 6.0, y - 6.0, 12.0, 12.0))
            if x > plot.center().x():
                label_rect = QRectF(x - 189.0, y - 27.0, 180.0, 20.0)
                label_alignment = Qt.AlignmentFlag.AlignRight
            else:
                label_rect = QRectF(x + 9.0, y - 27.0, 180.0, 20.0)
                label_alignment = Qt.AlignmentFlag.AlignLeft
            painter.drawText(label_rect, label_alignment, f"{label}: {fraction * 100:.1f}%, {radius:.2f} Å")
        painter.setPen(QColor("#374151"))
        painter.drawText(QRectF(plot.left(), self.height() - 30.0, plot.width(), 20.0), Qt.AlignmentFlag.AlignCenter, "Bottleneck position along physical arc length (%)")
        painter.save(); painter.translate(18.0, plot.center().y()); painter.rotate(-90.0)
        painter.drawText(QRectF(-plot.height() / 2.0, -12.0, plot.height(), 20.0), Qt.AlignmentFlag.AlignCenter, "Median radius (Å)")
        painter.restore()


class DistanceEvidenceView(QWidget):
    """Framewise mutation-to-residue C-alpha distance medians and IQRs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict = {}
        self.setMinimumHeight(390)

    def set_payload(self, payload: dict | None) -> None:
        self._payload = dict(payload or {})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rows = [dict(row) for row in self._payload.get("rows", []) or []]
        if not rows:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, str(self._payload.get("note") or "Frame-resolved C-alpha distances are unavailable."))
            return
        left, top, right, bottom = 126.0, 52.0, 26.0, 72.0
        plot = QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))
        endpoints = [float(stats[key]) for row in rows for stats in (dict(row.get("reference") or {}), dict(row.get("target") or {})) for key in ("q25", "q75") if stats.get(key) is not None]
        x_max = max(10.0, max(endpoints, default=10.0)) + 1.0
        row_h = plot.height() / max(1, len(rows))

        def px(value):
            return plot.left() + max(0.0, float(value)) / max(1e-9, x_max) * plot.width()

        painter.setPen(QColor("#111827"))
        anchor = str(self._payload.get("anchor_label") or "Mutation")
        painter.drawText(QRectF(left, 5.0, plot.width(), 22.0), Qt.AlignmentFlag.AlignCenter, f"{anchor} Cα distance to selected-region residues across MD frames")
        for index, row in enumerate(rows):
            y_center = plot.top() + (index + 0.5) * row_h
            painter.setPen(QPen(QColor("#E5E7EB"), 0.8))
            painter.drawLine(int(plot.left()), int(y_center), int(plot.right()), int(y_center))
            painter.setPen(QColor("#374151"))
            painter.drawText(QRectF(4.0, y_center - row_h / 2.0, left - 12.0, row_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(row.get("label") or row.get("residue_id") or "Residue"))
            for offset, side, color_value in ((-4.0, "reference", "#2563EB"), (4.0, "target", "#EA580C")):
                stats = dict(row.get(side) or {})
                if stats.get("median") is None:
                    continue
                median, q25, q75 = float(stats["median"]), float(stats.get("q25", stats["median"])), float(stats.get("q75", stats["median"]))
                y = y_center + offset
                color = QColor(color_value)
                painter.setPen(QPen(color, 2.2))
                painter.drawLine(int(px(q25)), int(y), int(px(q75)), int(y))
                painter.setBrush(color)
                painter.drawEllipse(QRectF(px(median) - 4.5, y - 4.5, 9.0, 9.0))
        for index in range(6):
            value = x_max * index / 5.0
            painter.setPen(QColor("#6B7280"))
            painter.drawText(QRectF(px(value) - 24.0, plot.bottom() + 5.0, 48.0, 18.0), Qt.AlignmentFlag.AlignCenter, f"{value:.1f}")
        painter.setPen(QColor("#374151"))
        painter.drawText(QRectF(plot.left(), self.height() - 43.0, plot.width(), 20.0), Qt.AlignmentFlag.AlignCenter, "Cα distance (Å) · point = median · line = IQR")
        painter.setPen(QColor("#6B7280"))
        painter.drawText(QRectF(8.0, self.height() - 22.0, self.width() - 16.0, 18.0), Qt.AlignmentFlag.AlignCenter, str(self._payload.get("note") or "blue = baseline · orange = target"))


class TemporalEvidenceStrip(QWidget):
    """Residue-frequency direction across sequential within-trajectory blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._blocks: list[dict] = []
        self._note = ""
        self.setMinimumHeight(340)

    def set_rows(
        self,
        rows: list[dict] | None,
        note: str = "",
        blocks: list[dict] | None = None,
    ) -> None:
        self._rows = [dict(row) for row in (rows or [])]
        self._blocks = [dict(block) for block in (blocks or [])]
        self._note = str(note or "")
        self.update()

    def _block_label(self, index: int) -> str:
        phases = ("Early", "Early–Mid", "Middle", "Mid–Late", "Late")
        block = self._blocks[index] if index < len(self._blocks) else {}
        phase = str(block.get("phase") or phases[min(index, len(phases) - 1)])
        reference = dict(block.get("reference") or {})
        target = dict(block.get("target") or {})
        if not reference or not target:
            return phase
        ref_range = (reference.get("frame_start"), reference.get("frame_end"))
        target_range = (target.get("frame_start"), target.get("frame_end"))
        ref_count = int(reference.get("path_count", 0) or 0)
        target_count = int(target.get("path_count", 0) or 0)
        if ref_range == target_range:
            count = str(ref_count) if ref_count == target_count else f"{ref_count}/{target_count}"
            return f"{phase} · frames {ref_range[0]}–{ref_range[1]} · n={count}"
        return (
            f"{phase} · Ref {ref_range[0]}–{ref_range[1]} n={ref_count}\n"
            f"Target {target_range[0]}–{target_range[1]} n={target_count}"
        )

    @staticmethod
    def _color(value: float) -> QColor:
        value = max(-1.0, min(1.0, float(value)))
        neutral = QColor("#F3F4F6")
        endpoint = QColor("#DC2626") if value >= 0 else QColor("#2563EB")
        ratio = abs(value)
        return QColor(
            round(neutral.red() + (endpoint.red() - neutral.red()) * ratio),
            round(neutral.green() + (endpoint.green() - neutral.green()) * ratio),
            round(neutral.blue() + (endpoint.blue() - neutral.blue()) * ratio),
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self._rows:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._note or "Select a residue-replacement region with frame-resolved paths")
            return
        block_count = max(len(row.get("values", []) or []) for row in self._rows)
        left, top, right, bottom = 116.0, 74.0, 18.0, 62.0
        width = max(1.0, self.width() - left - right)
        row_h = max(30.0, min(48.0, (self.height() - top - bottom) / max(1, len(self._rows))))
        cell_w = width / max(1, block_count)
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(left, 5.0, width, 22.0), Qt.AlignmentFlag.AlignCenter, "Target − baseline residue frequency in the selected path region")
        for block in range(block_count):
            painter.setPen(QColor("#4B5563"))
            painter.drawText(
                QRectF(left + block * cell_w + 2.0, 25.0, cell_w - 4.0, 45.0),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._block_label(block),
            )
        for row_index, row in enumerate(self._rows):
            y = top + row_index * row_h
            painter.setPen(QColor("#374151"))
            painter.drawText(QRectF(4.0, y, left - 10.0, row_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(row.get("label") or row.get("residue_id") or "Residue"))
            for block, value in enumerate(row.get("values", []) or []):
                rect = QRectF(left + block * cell_w + 2.0, y + 2.0, cell_w - 4.0, row_h - 4.0)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self._color(float(value)))
                painter.drawRoundedRect(rect, 3.0, 3.0)
                painter.setPen(QColor("#FFFFFF") if abs(float(value)) >= 0.48 else QColor("#374151"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{float(value):+.2f}")
        painter.setPen(QColor("#6B7280"))
        painter.drawText(QRectF(left, self.height() - 48.0, width, 18.0), Qt.AlignmentFlag.AlignCenter, "blue = decreased · red = increased · gray = little change")
        painter.drawText(QRectF(8.0, self.height() - 28.0, self.width() - 16.0, 20.0), Qt.AlignmentFlag.AlignCenter, self._note)


class RemoteEvidenceView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict = {}
        self._compact = False
        self.setMinimumHeight(300)

    def set_compact_mode(self, compact: bool = True) -> None:
        """Use a shallow association strip when embedded above the full-path map."""
        self._compact = bool(compact)
        if self._compact:
            self.setMinimumHeight(122)
            self.setMaximumHeight(132)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.setMinimumHeight(300)
            self.setMaximumHeight(16777215)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self.update()

    def set_payload(self, payload: dict | None) -> None:
        self._payload = dict(payload or {})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        evidence = list(self._payload.get("allosteric_evidence", []) or [])
        substitutions = list(self._payload.get("substitutions", []) or [])
        if not substitutions:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "No explicit amino-acid substitution is available for a distal-perturbation audit. Ensemble-state comparisons remain available in the other tabs.")
            return
        row = dict(evidence[0]) if evidence else {}
        mutation = str(row.get("mutation_label") or substitutions[0].get("label") or "Mutation")
        region = str(row.get("region_id") or self._payload.get("selected_region") or "Selected region")
        region_distance = row.get("region_distance", row.get("distance"))
        active_site_distance = row.get("active_site_distance")
        threshold = float(self._payload.get("remote_distance_threshold", 10.0) or 10.0)
        cards = (
            (mutation, "Perturbation", "#7C3AED"),
            (
                f"{float(active_site_distance):.1f} Å" if active_site_distance is not None else "distance unavailable",
                "Distance to active-site origin",
                "#2563EB",
            ),
            (
                f"{region} · Δregion {float(region_distance):.1f} Å"
                if region_distance is not None else region,
                "Path-local response",
                "#DC2626",
            ),
        )
        gap = 20.0 if self._compact else 36.0
        margin = 12.0 if self._compact else 18.0
        card_w = max(80.0, (self.width() - margin * 2.0 - gap * 2.0) / 3.0)
        top = 4.0 if self._compact else 72.0
        card_h = 62.0 if self._compact else 96.0
        rects = []
        for index, (value, title, color_value) in enumerate(cards):
            rect = QRectF(margin + index * (card_w + gap), top, card_w, card_h)
            rects.append(rect)
            color = QColor(color_value)
            fill = QColor(color_value); fill.setAlpha(24)
            painter.setPen(QPen(color, 1.2)); painter.setBrush(fill)
            painter.drawRoundedRect(rect, 7.0, 7.0)
            painter.setPen(color)
            title_bottom = -32.0 if self._compact else -48.0
            value_top = 26.0 if self._compact else 40.0
            painter.drawText(rect.adjusted(8.0, 6.0, -8.0, title_bottom), Qt.AlignmentFlag.AlignCenter, title)
            painter.setPen(QColor("#1F2937"))
            painter.drawText(rect.adjusted(8.0, value_top, -8.0, -6.0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, value)
        painter.setPen(QPen(QColor("#6B7280"), 1.5, Qt.PenStyle.DashLine))
        for left_rect, right_rect in zip(rects, rects[1:]):
            painter.drawLine(int(left_rect.right() + 4.0), int(left_rect.center().y()), int(right_rect.left() - 4.0), int(right_rect.center().y()))
        remote = bool(row.get("remote_candidate"))
        status = (
            f"Distal perturbation / path-response association candidate (active-site threshold ≥ {threshold:.1f} Å)"
            if remote
            else f"Active-site distal criterion not met (threshold ≥ {threshold:.1f} Å)"
        )
        painter.setPen(QColor("#166534") if remote else QColor("#6B7280"))
        status_top = 72.0 if self._compact else 190.0
        painter.drawText(QRectF(12.0, status_top, self.width() - 24.0, 22.0), Qt.AlignmentFlag.AlignCenter, status)
        painter.setPen(QColor("#6B7280"))
        note_top = 96.0 if self._compact else 224.0
        note_height = 24.0 if self._compact else 48.0
        note = (
            "Association only; dashed links are not physical transmission paths."
            if self._compact
            else "The active-site distance classifies perturbation remoteness; the region distance only localizes the response. Dashed lines are not physical transmission paths, and causality requires independent evidence."
        )
        painter.drawText(QRectF(20.0, note_top, self.width() - 40.0, note_height), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, note)


class EvidenceOverview(QWidget):
    """Question-oriented summaries that provide one-click access to details."""

    detail_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setChildrenCollapsible(False)

        self._path_host = QWidget(self._splitter)
        self._path_layout = QVBoxLayout(self._path_host)
        self._path_layout.setContentsMargins(0, 0, 0, 0)
        self._path_placeholder = QLabel(
            "Select a mapped cluster relation to initialize the full-path comparison."
        )
        self._path_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._path_placeholder.setWordWrap(True)
        self._path_placeholder.setStyleSheet(
            "color:#6B7280;font-size:12px;padding:20px;"
        )
        self._path_layout.addWidget(self._path_placeholder, 1)
        self._association_widget = None

        details_row = QWidget(self._path_host)
        details_layout = QHBoxLayout(details_row)
        details_layout.setContentsMargins(0, 2, 0, 0)
        details_layout.addStretch(1)
        self._details_button = QPushButton("Detailed Properties  →", details_row)
        self._details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._details_button.setToolTip(
            "Open tunnel profiles, statistics, and residue-combination regions."
        )
        self._details_button.setStyleSheet(
            "QPushButton{padding:5px 12px;border:1px solid #93C5FD;border-radius:6px;"
            "background:#EFF6FF;color:#1D4ED8;font-weight:600;}"
            "QPushButton:hover{background:#DBEAFE;border-color:#2563EB;}"
        )
        self._details_button.clicked.connect(
            lambda: self.detail_requested.emit("properties")
        )
        details_layout.addWidget(self._details_button)
        self._details_row = details_row
        self._path_layout.addWidget(details_row, 0)

        cards_host = QWidget(self._splitter)
        cards_layout = QVBoxLayout(cards_host)
        cards_layout.setContentsMargins(4, 0, 0, 0)
        cards_layout.setSpacing(8)
        self._cards: dict[str, QPushButton] = {}
        definitions = (
            ("composition", "1  Which residues changed?", "#EFF6FF", "#1D4ED8"),
            ("distance", "2  Is the perturbation spatially remote?", "#FAF5FF", "#7E22CE"),
            ("time", "3  Does the change persist over time?", "#FFF7ED", "#C2410C"),
        )
        for index, (key, title, background, color) in enumerate(definitions):
            button = QPushButton(f"{title}\n\nNo active comparison")
            button.setMinimumHeight(92)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton{{text-align:left;padding:16px;background:{background};"
                f"border:1px solid {color};border-radius:8px;color:#1F2937;font-size:13px;}}"
                f"QPushButton:hover{{border:2px solid {color};}}"
            )
            button.clicked.connect(lambda _checked=False, name=key: self.detail_requested.emit(name))
            cards_layout.addWidget(button, 1)
            self._cards[key] = button
        self._splitter.addWidget(self._path_host)
        self._splitter.addWidget(cards_host)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([620, 400])
        root.addWidget(self._splitter, 1)

    def set_path_region_widget(self, widget: QWidget | None) -> None:
        """Install the linked full-path replacement view in the left overview pane."""
        if widget is None:
            return
        if self._path_placeholder is not None:
            self._path_layout.removeWidget(self._path_placeholder)
            # Hide synchronously before scheduling destruction.  During
            # scripted/full-window captures the replacement widget can be
            # installed before Qt processes deferred deletes, otherwise the
            # obsolete empty-state text remains painted over valid results.
            self._path_placeholder.setVisible(False)
            self._path_placeholder.deleteLater()
            self._path_placeholder = None
        details_index = self._path_layout.indexOf(self._details_row)
        self._path_layout.insertWidget(max(0, details_index), widget, 1)

    def set_association_widget(self, widget: QWidget | None) -> None:
        """Place the compact association evidence directly above the path map."""
        if widget is None or widget is self._association_widget:
            return
        self._association_widget = widget
        title = QLabel("Association", self._path_host)
        title.setStyleSheet("font-weight:600;font-size:11px;color:#374151;")
        self._path_layout.insertWidget(0, title)
        self._path_layout.insertWidget(1, widget)

    def set_payload(self, payload: dict | None) -> None:
        payload = dict(payload or {})
        composition = dict(payload.get("composition") or {})
        composition_rows = [dict(row) for row in composition.get("rows", []) or []]
        if composition_rows:
            strongest = max(
                composition_rows,
                key=lambda row: abs(float(row.get("delta", 0.0) or 0.0)),
            )
            composition_text = (
                f"{len(composition_rows)} changed residue(s)\n"
                f"Largest change: {strongest.get('label') or strongest.get('residue_id') or 'Residue'} "
                f"{float(strongest.get('delta', 0.0) or 0.0):+.2f}"
            )
        else:
            composition_text = str(
                composition.get("note") or "No selected-region residue change is available."
            )
        self._cards["composition"].setText(
            "1  Which residues changed?\n\n" + composition_text + "\nOpen the composition view."
        )

        profile = dict(payload.get("residue_change_profile") or {})
        evidence = [dict(row) for row in profile.get("allosteric_evidence", []) or []]
        substitutions = list(profile.get("substitutions", []) or [])
        if evidence:
            strongest = evidence[0]
            distance = strongest.get("active_site_distance")
            label = "distal perturbation candidate" if bool(strongest.get("remote_candidate")) else "active-site distal criterion not met"
            remote_text = (
                f"Active-site distance {float(distance):.1f} Å\n{label}."
                if distance is not None else "Active-site distance is unavailable."
            )
        elif substitutions:
            remote_text = "Mutation detected; aligned distance is unavailable."
        else:
            remote_text = "No explicit substitution is available for a distance audit."
        self._cards["distance"].setText(
            "2  Is the perturbation spatially remote?\n\n" + remote_text + "\nOpen the spatial audit."
        )

        temporal = dict(payload.get("temporal") or {})
        rows = [dict(row) for row in temporal.get("rows", []) or []]
        block_count = max((len(row.get("values", []) or []) for row in rows), default=0)
        temporal_text = (
            f"{len(rows)} residue trace(s)\n{block_count} sequential block(s)."
            if rows else str(temporal.get("note") or "Temporal evidence is not available.")
        )
        self._cards["time"].setText(
            "3  Does the change persist over time?\n\n" + temporal_text + "\nOpen the temporal strip."
        )


class EvidencePanel(QWidget):
    """Coordinated detail-on-demand evidence view."""

    back_requested = Signal()
    observer_requested = Signal()
    marker_requested = Signal(object)
    dataset_selected = Signal(str)
    region_step_requested = Signal(int)
    properties_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("evidenceWorkspace")
        self.setMinimumHeight(320)
        self.setStyleSheet("#evidenceWorkspace{background:#ffffff;border-left:1px solid #e5e7eb;}")
        outer = QVBoxLayout(self)
        self._outer_layout = outer
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(7)

        header_widget = QWidget(self)
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Evidence")
        title.setStyleSheet("font-size:12px;font-weight:600;color:#1f2937;")
        header.addWidget(title)
        self._context = QLabel("No active comparison")
        self._context.setStyleSheet("font-size:11px;color:#475569;")
        header.addWidget(self._context, 1)
        self._observer_button = QPushButton("Inspect in Residue Observer")
        self._observer_button.clicked.connect(self.observer_requested.emit)
        self._observer_button.setEnabled(False)
        header.addWidget(self._observer_button)
        self._previous_region = QPushButton("Previous Region")
        self._previous_region.clicked.connect(lambda: self.region_step_requested.emit(-1))
        self._previous_region.setEnabled(False)
        header.addWidget(self._previous_region)
        self._next_region = QPushButton("Next Region")
        self._next_region.clicked.connect(lambda: self.region_step_requested.emit(1))
        self._next_region.setEnabled(False)
        header.addWidget(self._next_region)
        self._back_button = QPushButton("Back to Tunnel Properties")
        self._back_button.clicked.connect(self.back_requested.emit)
        self._back_button.setVisible(False)
        header.addWidget(self._back_button)
        header_widget.setVisible(False)
        outer.addWidget(header_widget)

        self._boundary = QLabel("Evidence is linked to the selected mapping and physical arc-length region. Trajectory blocks are descriptive subsamples, not independent replicas.")
        self._boundary.setWordWrap(True)
        self._boundary.setStyleSheet("font-size:11px;color:#1E3A8A;background:#EFF6FF;border:1px solid #BFDBFE;border-radius:5px;padding:6px;")
        self._boundary.setVisible(False)
        outer.addWidget(self._boundary)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self._tabs, 1)

        self._overview = EvidenceOverview()
        self._overview.detail_requested.connect(self.select_tab)
        self._tabs.addTab(self._wrap(self._overview), "Overview")
        self._composition = CompositionEvidenceView()
        self._tabs.addTab(self._wrap(self._composition), "Residues")
        # Bottleneck evidence remains integrated into the full-path map; keep
        # its renderer/data sink internal without exposing a duplicate tab.
        self._bottleneck = BottleneckEvidenceView(self)
        self._bottleneck.setVisible(False)
        self._distance = DistanceEvidenceView()
        self._tabs.addTab(self._wrap(self._distance), "Distance")
        self._temporal = TemporalEvidenceStrip()
        self._tabs.addTab(self._wrap(self._temporal), "Time")
        self._remote = RemoteEvidenceView()
        self._remote.set_compact_mode(True)
        self._overview.set_association_widget(self._remote)
        # Keep the marker implementation available for backward-compatible
        # programmatic use, but omit the redundant manual-entry tab from the UI.
        self._marker_page = self._build_marker_tab()
        self._marker_page.setVisible(False)

        self._charts_page = None
        self._shared_toolbar = None

    def set_path_region_widget(self, widget: QWidget | None) -> None:
        self._overview.set_path_region_widget(widget)

    def set_shared_toolbar(self, widget: QWidget | None) -> None:
        """Keep chart controls visible while switching analysis tabs."""
        if widget is None or widget is self._shared_toolbar:
            return
        if self._shared_toolbar is not None:
            self._outer_layout.removeWidget(self._shared_toolbar)
        self._shared_toolbar = widget
        tab_index = self._outer_layout.indexOf(self._tabs)
        self._outer_layout.insertWidget(max(0, tab_index), widget)

    def set_charts_page(
        self,
        widget: QWidget,
        *,
        visible: bool = True,
        select: bool = False,
    ) -> None:
        """Install the tunnel-property workspace as a detail-on-demand tab."""
        if widget is None:
            return
        previous = self._charts_page
        if previous is widget and self._tabs.indexOf(widget) >= 0:
            index = self._tabs.indexOf(widget)
            self._tabs.setTabVisible(index, bool(visible))
            if visible and select:
                self._tabs.setCurrentWidget(widget)
            elif not visible:
                self.select_tab("overview")
            return
        if previous is not None:
            previous_index = self._tabs.indexOf(previous)
            if previous_index >= 0:
                self._tabs.removeTab(previous_index)
        self._charts_page = widget
        overview_index = next(
            (
                index for index in range(self._tabs.count())
                if self._tabs.tabText(index).strip().lower() == "overview"
            ),
            0,
        )
        index = self._tabs.insertTab(overview_index + 1, widget, "Tunnel Properties")
        self._tabs.setTabVisible(index, bool(visible))
        if visible and select:
            self._tabs.setCurrentWidget(widget)
        else:
            self.select_tab("overview")

    def charts_page_visible(self) -> bool:
        """Return whether the Tunnel Properties entry is exposed to the user."""
        if self._charts_page is None:
            return False
        index = self._tabs.indexOf(self._charts_page)
        return bool(index >= 0 and self._tabs.isTabVisible(index))

    def tunnel_properties_visible(self) -> bool:
        return self.current_tab_name() == "tunnel properties"

    def _on_tab_changed(self, index: int) -> None:
        if index >= 0 and self._tabs.tabText(index).strip().lower() == "tunnel properties":
            self.properties_requested.emit()

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget, 1)
        return container

    def _build_marker_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        intro = QLabel("Define a residue or residue group as an independent functional marker. Showing a marker focuses the same residues in the current Observer datasets; it does not hard-code a protein-specific motif.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#475569;font-size:11px;")
        layout.addWidget(intro)
        form = QGridLayout()
        form.addWidget(QLabel("Marker name"), 0, 0)
        self._marker_name = QLineEdit()
        self._marker_name.setPlaceholderText("e.g. deep gate or DRY ionic lock")
        form.addWidget(self._marker_name, 0, 1)
        form.addWidget(QLabel("Residues (MD numbering)"), 1, 0)
        self._marker_residues = QLineEdit()
        self._marker_residues.setPlaceholderText("e.g. 127,310,487")
        self._marker_residues.returnPressed.connect(self._emit_marker)
        form.addWidget(self._marker_residues, 1, 1)
        self._show_marker = QPushButton("Show in Residue Observer")
        self._show_marker.clicked.connect(self._emit_marker)
        form.addWidget(self._show_marker, 2, 1, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(form)
        self._marker_status = QLabel("No marker selected.")
        self._marker_status.setWordWrap(True)
        self._marker_status.setStyleSheet("background:#F9FAFB;border:1px solid #E5E7EB;border-radius:5px;padding:7px;color:#374151;")
        layout.addWidget(self._marker_status)
        layout.addStretch(1)
        return widget

    def _emit_marker(self) -> None:
        residue_ids = []
        for token in self._marker_residues.text().replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                residue_id = int(token)
            except ValueError:
                self._marker_status.setText(f"Invalid residue identifier: {token}")
                return
            if residue_id > 0 and residue_id not in residue_ids:
                residue_ids.append(residue_id)
        if not residue_ids:
            self._marker_status.setText("Enter one or more MD residue identifiers.")
            return
        name = self._marker_name.text().strip() or "Functional marker"
        self._marker_status.setText(f"{name}: " + ", ".join(f"MD{value}" for value in residue_ids))
        self.marker_requested.emit({"name": name, "residue_ids": tuple(residue_ids)})

    def set_marker_result(self, text: str) -> None:
        self._marker_status.setText(str(text or "Marker evidence is unavailable."))

    def set_evidence(self, payload: dict | None) -> None:
        payload = dict(payload or {})
        reference = str(payload.get("reference_label") or payload.get("reference_key") or "baseline")
        target = str(payload.get("target_label") or payload.get("target_key") or "target")
        ref_cluster = payload.get("reference_cluster")
        target_cluster = payload.get("target_cluster")
        target_cluster_label = str(
            payload.get("target_cluster_label")
            or (f"C{target_cluster}" if target_cluster else "")
        )
        region = str(payload.get("selected_region") or "full path")
        if ref_cluster and target_cluster:
            self._context.setText(f"{reference} C{ref_cluster} → {target} {target_cluster_label} · {region}")
            self._observer_button.setEnabled(True)
            profile = dict(payload.get("residue_change_profile") or {})
            region_count = len(profile.get("hotspots", []) or [])
            self._previous_region.setEnabled(region_count > 1)
            self._next_region.setEnabled(region_count > 1)
        else:
            self._context.setText("No active comparison")
            self._observer_button.setEnabled(False)
            self._previous_region.setEnabled(False)
            self._next_region.setEnabled(False)
        self._overview.set_payload(payload)
        self._composition.set_payload(payload.get("composition"))
        # Bottleneck is already represented in the visible full-path map. Its
        # retired standalone renderer must not receive updates while hidden.
        if self._bottleneck.isVisible():
            self._bottleneck.set_payload(payload.get("bottleneck"))
        self._distance.set_payload(payload.get("distance"))
        self._remote.set_payload(payload.get("residue_change_profile"))
        temporal = dict(payload.get("temporal", {}) or {})
        self._temporal.set_rows(
            temporal.get("rows"),
            str(temporal.get("note") or ""),
            temporal.get("blocks"),
        )
        provenance = dict(payload.get("provenance") or {})
        if ref_cluster and target_cluster:
            ref_paths = int(provenance.get("reference_path_count", 0) or 0)
            target_paths = int(provenance.get("target_path_count", 0) or 0)
            provenance_target_label = str(
                provenance.get("target_cluster_label") or target_cluster_label
            )
            basis = str(provenance.get("position_basis") or "physical_arc_length").replace("_", " ")
            self._boundary.setText(
                f"Linked source: {reference} C{ref_cluster} ({ref_paths} paths) → "
                f"{target} {provenance_target_label} ({target_paths} paths) · {region} · {basis}. "
                "Composition, bottleneck, distance, and time panels use this same selection. "
                "Trajectory blocks are descriptive subsamples, not independent replicas."
            )
        else:
            self._boundary.setText("Evidence is linked to the selected mapping and physical arc-length region. Trajectory blocks are descriptive subsamples, not independent replicas.")

    def select_tab(self, name: str) -> None:
        normalized = str(name or "").strip().lower()
        labels = {
            "charts": "tunnel properties", "chart": "tunnel properties",
            "properties": "tunnel properties", "details": "tunnel properties",
            "tunnel properties": "tunnel properties",
            "overview": "overview",
            "composition": "residues", "residues": "residues",
            "bottleneck": "overview", "gate": "overview",
            "distance": "distance",
            "time": "time",
            "remote": "overview", "association": "overview",
        }
        target = labels.get(normalized)
        if target is None:
            return
        if target == "tunnel properties" and not self.charts_page_visible():
            target = "overview"
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index).strip().lower() == target:
                self._tabs.setCurrentIndex(index)
                return

    def current_tab_name(self) -> str:
        return self._tabs.tabText(self._tabs.currentIndex()).strip().lower()
