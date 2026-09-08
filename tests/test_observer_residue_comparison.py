import sqlite3
import unittest
from types import SimpleNamespace

from TopoTunnel_UI.app.main_window import MainWindow
from TopoTunnel_UI.app.widgets.protein_viewer_panel import _normalized_path_focus_indexes
from TopoTunnel_UI.vis.tunnel_viewer import TunnelViewer3D


class _Actor:
    def __init__(self):
        self.visible = True

    def SetVisibility(self, value):
        self.visible = bool(value)


class _FocusViewer:
    _DATASET_PREFIX = TunnelViewer3D._DATASET_PREFIX
    set_dataset_visible = TunnelViewer3D.set_dataset_visible

    def __init__(self):
        self._plotter = SimpleNamespace(actors={
            "dataset_paths": _Actor(),
            "dataset_compare_hotspot_0": _Actor(),
        })

    def _request_render(self):
        pass


class _TextTarget:
    def __init__(self):
        self.value = ""
        self.visible = False

    def setText(self, value):
        self.value = str(value)

    def setToolTip(self, _value):
        pass

    def setVisible(self, value):
        self.visible = bool(value)


class _Panel:
    def __init__(self):
        self.specs = {}

    def residue_name_map(self, _residue_ids):
        return {}

    def set_highlight_label_specs(self, specs):
        self.specs = dict(specs)


def _database(point_residues, names):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE paths(id INTEGER, cluster_id INTEGER)")
    connection.execute(
        "CREATE TABLE path_points("
        "path_id INTEGER, seq INTEGER, res_1 INTEGER, res_2 INTEGER, "
        "res_3 INTEGER, res_4 INTEGER)"
    )
    connection.execute("INSERT INTO paths VALUES(1, 7)")
    for seq, residues in enumerate(point_residues):
        values = list(residues) + [0] * (4 - len(residues))
        connection.execute(
            "INSERT INTO path_points VALUES(?,?,?,?,?,?)",
            (1, seq, *values),
        )
    connection.commit()
    database = SimpleNamespace(conn=connection, _residue_name_map=dict(names))
    database.get_residue_positions = lambda: [
        {
            "residue_id": int(rid),
            "x": float(index * 2),
            "y": 0.0,
            "z": 0.0,
        }
        for index, rid in enumerate(sorted(names))
    ]
    return database


class _ObserverOwner:
    _OBSERVER_RESIDUE_COLORS = MainWindow._OBSERVER_RESIDUE_COLORS
    _observer_active_path_membership = MainWindow._observer_active_path_membership
    _dataset_compare_observer_change_events = MainWindow._dataset_compare_observer_change_events
    _observer_residue_names_for_slot = MainWindow._observer_residue_names_for_slot
    _apply_observer_residue_comparison_visuals = MainWindow._apply_observer_residue_comparison_visuals

    def __init__(self):
        self.db = SimpleNamespace(_datasets_by_key={
            "wt": SimpleNamespace(db=_database([(145, 212)], {145: "ALA", 212: "VAL", 271: "LEU"})),
            "close": SimpleNamespace(db=_database([(145, 271)], {145: "ALA", 212: "VAL", 271: "LEU"})),
        })
        self._dataset_compare_active = {
            "reference_key": "wt",
            "reference_cluster": 7,
            "target_key": "close",
            "target_cluster": 7,
        }
        self._protein_observer_debug_label = _TextTarget()
        self.slots = [
            {"dataset_key": key, "panel": _Panel(), "residue_summary": _TextTarget()}
            for key in ("wt", "close")
        ]

    def _protein_observer_slots(self):
        return self.slots


class ObserverResidueComparisonTests(unittest.TestCase):
    def test_focus_indexes_use_equal_normalized_length_for_different_paths(self):
        focused, peaks = _normalized_path_focus_indexes(
            [1] * 5 + [2] * 9,
            0.45,
            0.55,
            0.50,
        )
        self.assertEqual(peaks.tolist(), [2, 9])
        self.assertEqual(focused.tolist(), [2, 9])

    def test_focus_keeps_comparison_annotations_visible(self):
        viewer = _FocusViewer()
        viewer.set_dataset_visible(False)
        self.assertFalse(viewer._plotter.actors["dataset_paths"].visible)
        self.assertTrue(viewer._plotter.actors["dataset_compare_hotspot_0"].visible)

    def test_pair_membership_detects_path_replacement(self):
        owner = _ObserverOwner()
        membership = owner._observer_active_path_membership((145, 212))
        self.assertEqual(membership["wt"]["group_count"], 1)
        self.assertEqual(membership["close"]["group_count"], 0)

    def test_path_contact_change_uses_delta_and_stable_position_color(self):
        owner = _ObserverOwner()
        membership = owner._observer_active_path_membership((145, 212))
        owner._apply_observer_residue_comparison_visuals(
            (145, 212),
            path_membership_by_dataset=membership,
        )
        reference, target = owner.slots
        self.assertIn("Δ 212 VAL", reference["panel"].specs[212]["text"])
        self.assertEqual(
            reference["panel"].specs[212]["shape_color"],
            target["panel"].specs[212]["shape_color"],
        )
        self.assertIn("Group on path 1/1", reference["residue_summary"].value)
        self.assertIn("Group off path 0/1", target["residue_summary"].value)
        self.assertIn("path contact changed", owner._protein_observer_debug_label.value)

    def test_detected_events_expose_single_and_pair_rewiring(self):
        owner = _ObserverOwner()
        events = owner._dataset_compare_observer_change_events("wt", 7, "close", 7)
        single_labels = [
            row["display_label"] for row in events
            if row.get("selection_type") == "single"
        ]
        pair_labels = [
            row["display_label"] for row in events
            if row.get("selection_type") == "pair"
        ]
        self.assertTrue(any("212 VAL" in label and "271 LEU" in label for label in single_labels))
        self.assertTrue(any("145–212 ALA-VAL" in label and "145–271 ALA-LEU" in label for label in pair_labels))
        rewired = next(row for row in events if "212 VAL" in row["display_label"] and "271 LEU" in row["display_label"])
        self.assertIn("Path contact was rewired", rewired["impact_text"])
        pair_event = next(row for row in events if "145–212" in row["display_label"] and "145–271" in row["display_label"])
        self.assertIn("Pair separation changes", pair_event["impact_text"])


if __name__ == "__main__":
    unittest.main()
