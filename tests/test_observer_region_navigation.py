import unittest

from TopoTunnel_UI.app.main_window import MainWindow


class ObserverRegionNavigationTests(unittest.TestCase):
    def _owner(self, observer_active: bool):
        owner = MainWindow.__new__(MainWindow)
        owner._protein_observer_page_active = lambda: observer_active
        owner.calls = []
        owner._on_dataset_compare_motif_selected = (
            lambda row, **kwargs: owner.calls.append(("motif", dict(row), dict(kwargs)))
        )
        owner._set_bottom_analysis_mode = (
            lambda mode: owner.calls.append(("bottom", mode))
        )
        owner._refresh_evidence_panel = (
            lambda row=None, **kwargs: owner.calls.append(
                ("evidence", dict(row or {}), dict(kwargs))
            )
        )
        owner._set_main_workspace = (
            lambda workspace: owner.calls.append(("workspace", workspace))
        )
        return owner

    def test_region_click_preserves_active_observer_workspace(self):
        owner = self._owner(True)

        MainWindow._on_dataset_compare_region_selected(owner, {"region_id": "R3"})

        self.assertIn(("bottom", "evidence"), owner.calls)
        self.assertFalse(any(call[0] == "workspace" for call in owner.calls))
        motif = next(call for call in owner.calls if call[0] == "motif")
        self.assertEqual(motif[2]["open_observer"], True)
        self.assertEqual(motif[2]["refresh_evidence"], False)

    def test_region_click_opens_lower_evidence_without_replacing_main_workspace(self):
        owner = self._owner(False)

        MainWindow._on_dataset_compare_region_selected(owner, {"region_id": "R2"})

        self.assertIn(("bottom", "evidence"), owner.calls)
        self.assertFalse(any(call[0] == "workspace" for call in owner.calls))


if __name__ == "__main__":
    unittest.main()
