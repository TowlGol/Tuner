import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from TopoTunnel_UI.app.widgets.dataset_compare_panel import DatasetComparePanel


class DatasetCompareRegionSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _profile():
        hotspots = [
            {
                "region_id": "R1",
                "start_fraction": 0.75,
                "end_fraction": 0.90,
                "peak_fraction": 0.84,
                "max_score": 0.70,
                "lost_sequence_ids": (10,),
                "gained_sequence_ids": (20,),
                "reference_residue_ids": (9,),
                "target_residue_ids": (19,),
            },
            {
                "region_id": "R3",
                "start_fraction": 0.38,
                "end_fraction": 0.50,
                "peak_fraction": 0.43,
                "max_score": 0.63,
                "lost_sequence_ids": (30,),
                "gained_sequence_ids": (40,),
                "reference_residue_ids": (29,),
                "target_residue_ids": (39,),
            },
        ]
        return {
            "bins": [{"fraction": 0.5, "score": 0.5}],
            "hotspots": hotspots,
            "substitutions": [{"label": "S117 ARG→ALA"}],
            "allosteric_evidence": [
                {
                    **hotspots[0],
                    "mutation_label": "S117 ARG→ALA",
                    "distance": 6.0,
                    "region_distance": 6.0,
                    "active_site_distance": 8.0,
                    "replacement_score": 0.70,
                    "remote_candidate": False,
                },
                {
                    **hotspots[1],
                    "mutation_label": "S117 ARG→ALA",
                    "distance": 4.0,
                    "region_distance": 4.0,
                    "active_site_distance": 13.0,
                    "replacement_score": 0.63,
                    "remote_candidate": True,
                },
            ],
            "remote_distance_threshold": 10.0,
        }

    def test_selecting_region_updates_track_strip_and_summary(self):
        panel = DatasetComparePanel()
        panel.show()
        try:
            panel.set_result(
                summary="comparison",
                changes=[],
                residue_change_profile=self._profile(),
                reference_dataset_key="wt",
                target_dataset_key="mutant",
            )
            self.app.processEvents()
            self.assertEqual(panel._replacement_track.selected_hotspot()["region_id"], "R1")
            self.assertEqual(panel._allosteric_strip.selected_region_id(), "R1")
            self.assertIn("8.0 Å from the active-site origin", panel._allosteric_summary.text())
            self.assertIn("linked R1 response is 6.0 Å", panel._allosteric_summary.text())
            self.assertIn("active-site distal criterion not met", panel._allosteric_summary.text())

            selected = panel.select_hotspot("R3", emit=False)
            self.app.processEvents()

            self.assertEqual(selected["region_id"], "R3")
            self.assertEqual(panel._replacement_track.selected_hotspot()["region_id"], "R3")
            self.assertEqual(panel._allosteric_strip.selected_region_id(), "R3")
            self.assertEqual(panel._allosteric_strip._selected_hotspot()["region_id"], "R3")
            self.assertIn("13.0 Å from the active-site origin", panel._allosteric_summary.text())
            self.assertIn("linked R3 response is 4.0 Å", panel._allosteric_summary.text())
            self.assertIn("distal perturbation / path-response association candidate", panel._allosteric_summary.text())
            self.assertNotIn("linked R1 response", panel._allosteric_summary.text())
        finally:
            panel.close()

    def test_region_step_updates_remote_strip_selection(self):
        panel = DatasetComparePanel()
        try:
            panel.set_result(
                summary="comparison",
                changes=[],
                residue_change_profile=self._profile(),
            )
            selected = panel.step_hotspot(1, emit=False)
            self.assertEqual(selected["region_id"], "R3")
            self.assertEqual(panel._allosteric_strip.selected_region_id(), "R3")
            self.assertIn("13.0 Å from the active-site origin", panel._allosteric_summary.text())
            self.assertIn("linked R3 response is 4.0 Å", panel._allosteric_summary.text())
        finally:
            panel.close()

    def test_retired_detail_tables_are_not_populated_while_hidden(self):
        panel = DatasetComparePanel()
        try:
            panel.set_result(
                summary="comparison",
                changes=[{
                    "combination_size": 2,
                    "residue_label": "1;2",
                    "status": "replacement",
                }],
                geometry_regions=[{
                    "start_fraction": 0.1,
                    "end_fraction": 0.2,
                }],
                motif_transitions=[{
                    "start_fraction": 0.1,
                    "end_fraction": 0.2,
                    "reference_residue_ids": (1, 2),
                    "target_residue_ids": (1, 3),
                }],
                residue_change_profile=self._profile(),
            )
            self.assertEqual(panel._table.rowCount(), 0)
            self.assertEqual(panel._geometry_table.rowCount(), 0)
            self.assertEqual(panel._motif_table.rowCount(), 0)
            self.assertFalse(panel._table.updatesEnabled())
            self.assertFalse(panel._geometry_table.updatesEnabled())
            self.assertFalse(panel._motif_table.updatesEnabled())
        finally:
            panel.close()

    def test_source_dataset_defaults_to_first_non_source_target(self):
        panel = DatasetComparePanel()
        try:
            panel.set_source_datasets(
                [
                    {"key": "wt", "prefix": "WT"},
                    {"key": "mut_a", "prefix": "R117A"},
                    {"key": "mut_b", "prefix": "R117K"},
                ],
                selected_key="wt",
            )

            self.assertEqual(panel.source_dataset_key(), "wt")
            self.assertEqual(panel.target_dataset_keys(), ["mut_a"])
            self.assertEqual(panel._target_dataset_combo.currentText(), "R117A")
            self.assertEqual(panel._target_dataset_combo.count(), 2)
        finally:
            panel.close()

    def test_changing_source_rebuilds_single_target_choice_without_the_source(self):
        panel = DatasetComparePanel()
        try:
            panel.set_source_datasets(
                [
                    {"key": "wt", "prefix": "WT"},
                    {"key": "mut_a", "prefix": "R117A"},
                    {"key": "mut_b", "prefix": "R117K"},
                ],
                selected_key="wt",
            )
            panel._source_dataset_combo.setCurrentIndex(1)

            self.assertEqual(panel.source_dataset_key(), "mut_a")
            self.assertEqual(panel.target_dataset_keys(), ["wt"])
            self.assertEqual(panel._target_dataset_combo.count(), 2)
            self.assertNotIn("mut_a", panel.target_dataset_keys())
        finally:
            panel.close()

    @staticmethod
    def _mapping_row(reference_cluster, target_key, target_label, target_cluster, rmsd, similarity):
        option = {
            "target_cluster_id": target_cluster,
            "center_rmsd": rmsd,
            "residue_similarity": similarity,
        }
        return {
            "reference_key": "wt",
            "reference_cluster_id": reference_cluster,
            "target_dataset": target_key,
            "target_label": target_label,
            "target_cluster_id": target_cluster,
            "target_cluster_ids": (target_cluster,),
            "target_options": [option],
            "center_rmsd": rmsd,
            "residue_similarity": similarity,
            "locked": False,
        }

    def test_mapping_table_uses_one_row_per_source_target_mapping(self):
        panel = DatasetComparePanel()
        try:
            panel.set_comparison_rows([
                self._mapping_row(7, "mut_a", "R117A", 6, 1.25, 0.72),
                self._mapping_row(7, "mut_b", "R117K", 12, 1.80, 0.64),
                self._mapping_row(8, "mut_a", "R117A", 4, 2.10, 0.58),
            ])

            self.assertEqual(panel._candidate_table.columnCount(), 5)
            self.assertEqual(panel._candidate_table.rowCount(), 3)
            self.assertEqual(panel._candidate_table.item(0, 0).text(), "C7")
            self.assertEqual(panel._candidate_table.item(0, 1).text(), "R117A")
            self.assertEqual(panel._candidate_table.item(0, 2).text(), "C6")
            self.assertIn("C6 1.250", panel._candidate_table.item(0, 3).text())
            self.assertIn("C6 72.0%", panel._candidate_table.item(0, 4).text())
            self.assertEqual(panel._candidate_table.item(1, 0).text(), "C7")
            self.assertEqual(panel._candidate_table.item(1, 1).text(), "R117K")
            self.assertEqual(panel._candidate_table.item(1, 2).text(), "C12")
            headers = [
                panel._candidate_table.horizontalHeaderItem(column).text()
                for column in range(panel._candidate_table.columnCount())
            ]
            self.assertNotIn("Local Rank", headers)
            self.assertNotIn("Status", headers)
        finally:
            panel.close()

    def test_table_click_refreshes_comparison_without_programmatic_default_refresh(self):
        panel = DatasetComparePanel()
        emitted = []
        panel.comparison_selected.connect(lambda payload: emitted.append(dict(payload)))
        try:
            panel.set_comparison_rows([
                self._mapping_row(7, "mut_a", "R117A", 6, 1.25, 0.72),
                self._mapping_row(7, "mut_b", "R117K", 12, 1.80, 0.64),
            ])
            self.app.processEvents()
            self.assertEqual(emitted, [])

            panel._on_candidate_cell_clicked(1, 1)
            self.assertEqual(len(emitted), 1)
            self.assertEqual(emitted[0]["target_dataset"], "mut_b")
            self.assertEqual(emitted[0]["target_cluster_ids"], (12,))
        finally:
            panel.close()


if __name__ == "__main__":
    unittest.main()
