import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QSplitter,
    QWidget,
)

from TopoTunnel_UI.app.main_window import MainWindow
from TopoTunnel_UI.app.widgets.dataset_panel import DatasetPanel
from TopoTunnel_UI.app.widgets.evidence_panel import EvidencePanel


class _EmptyDatasetDb:
    def list_datasets(self):
        return []


class DatasetPanelLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_both_action_rows_remain_visible_at_supported_heights(self):
        panel = DatasetPanel(_EmptyDatasetDb())
        panel.show()
        try:
            footer_height = None
            for height in (700, 400, 300, 220, 180, 140, 110, 80, 60):
                panel.resize(900, height)
                self.app.processEvents()
                first_top = panel._add_btn.mapTo(
                    panel, panel._add_btn.rect().topLeft()
                ).y()
                second_bottom = panel._remove_btn.mapTo(
                    panel, panel._remove_btn.rect().bottomLeft()
                ).y()
                self.assertGreaterEqual(first_top, 0)
                self.assertLess(second_bottom, panel.height())
                if footer_height is None:
                    footer_height = panel._action_footer.height()
                self.assertEqual(panel._action_footer.height(), footer_height)
            self.assertFalse(panel._tree.isVisible())
            self.assertFalse(panel._header_label.isVisible())
            self.assertFalse(panel._summary_label.isVisible())
            self.assertFalse(panel._hint_label.isVisible())
        finally:
            panel.close()

    def test_main_controls_remain_in_one_row_at_initial_and_narrow_widths(self):
        controls = QWidget()
        layout = QGridLayout(controls)
        items = [QPushButton(f"Control {index}") for index in range(13)]
        owner = SimpleNamespace(
            _top_controls_layout=layout,
            _top_controls_widget=controls,
            _top_controls_items=items,
        )
        try:
            for width in (0, 600, 1200):
                controls.resize(width, 80)
                MainWindow._refresh_top_controls_layout(owner)
                for column, widget in enumerate(items):
                    layout_item = layout.itemAtPosition(0, column)
                    self.assertIsNotNone(layout_item)
                    self.assertIs(layout_item.widget(), widget)
                    self.assertIsNone(layout.itemAtPosition(1, column))
        finally:
            controls.close()

    def test_tunnel_properties_are_detail_on_demand_with_a_minimum_height(self):
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(QWidget())
        charts_page = QWidget()
        evidence_panel = EvidencePanel()
        evidence_panel.set_charts_page(charts_page)
        splitter.addWidget(evidence_panel)
        owner = SimpleNamespace(
            _bottom_charts_page=charts_page,
            _evidence_panel=evidence_panel,
            _main_splitter=splitter,
            _bottom_analysis_mode="charts",
            _bottom_chart_splitter_sizes=[560, 280],
        )
        try:
            self.assertEqual(evidence_panel.minimumHeight(), 320)
            self.assertEqual(evidence_panel._tabs.tabText(0), "Overview")
            self.assertEqual(evidence_panel._tabs.tabText(1), "Tunnel Properties")
            self.assertEqual(evidence_panel.current_tab_name(), "overview")
            self.assertTrue(evidence_panel._back_button.isHidden())

            MainWindow._set_bottom_analysis_mode(owner, "evidence")
            self.assertEqual(evidence_panel.current_tab_name(), "overview")

            MainWindow._set_bottom_analysis_mode(owner, "charts")
            self.assertIs(evidence_panel._tabs.currentWidget(), charts_page)
            self.assertEqual(evidence_panel.current_tab_name(), "tunnel properties")

            MainWindow._set_bottom_analysis_mode(owner, "evidence")
            evidence_panel._overview._details_button.click()
            self.app.processEvents()
            self.assertIs(evidence_panel._tabs.currentWidget(), charts_page)
            self.assertEqual(evidence_panel.current_tab_name(), "tunnel properties")
        finally:
            splitter.close()


if __name__ == "__main__":
    unittest.main()
