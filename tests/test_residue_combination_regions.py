import unittest

from TopoTunnel_UI.vis.charts import (
    _build_residue_combination_regions,
    _combination_region_distance,
)


def _profile(prefix: str, residue_base: int, *, changed: bool = False) -> dict:
    local_left = [10, 10, 30, 30]
    local_right = [20, 20, 40, 40]
    if changed:
        local_left[2:] = [50, 50]
        local_right[2:] = [60, 60]
    return {
        "datasetPrefix": prefix,
        "resSeq": [0.0, 1.0, 2.0, 3.0],
        "radius": [2.0, 2.0, 2.0, 2.0],
        "res_1": [residue_base + value for value in local_left],
        "res_2": [residue_base + value for value in local_right],
        "res_3": [0, 0, 0, 0],
        "res_4": [0, 0, 0, 0],
        "local_res_1": local_left,
        "local_res_2": local_right,
        "local_res_3": [0, 0, 0, 0],
        "local_res_4": [0, 0, 0, 0],
    }


class ResidueCombinationRegionTests(unittest.TestCase):
    def test_local_ids_prevent_dataset_offsets_from_appearing_as_change(self):
        result = _build_residue_combination_regions(
            [
                _profile("WT", 1_000_000),
                _profile("MUT", 2_000_000),
            ],
            2,
            bin_count=8,
        )
        left, right = result["datasets"]
        scores = [
            _combination_region_distance(a["counts"], b["counts"])
            for a, b in zip(left["bins"], right["bins"])
        ]
        self.assertEqual(scores, [0.0] * 8)

    def test_changed_half_is_reported_as_complete_distribution_change(self):
        result = _build_residue_combination_regions(
            [
                _profile("WT", 1_000_000),
                _profile("MUT", 2_000_000, changed=True),
            ],
            2,
            bin_count=8,
        )
        left, right = result["datasets"]
        scores = [
            _combination_region_distance(a["counts"], b["counts"])
            for a, b in zip(left["bins"], right["bins"])
        ]
        self.assertEqual(scores[:4], [0.0] * 4)
        self.assertEqual(scores[4:], [1.0] * 4)

    def test_profile_sampling_limit_is_reported(self):
        profiles = [_profile("WT", 1_000_000) for _ in range(6)]
        result = _build_residue_combination_regions(
            profiles,
            2,
            bin_count=8,
            profile_limit=3,
        )
        dataset = result["datasets"][0]
        self.assertEqual(dataset["profile_count"], 6)
        self.assertEqual(dataset["sampled_profile_count"], 3)
        self.assertTrue(all(bin_data["counts"] for bin_data in dataset["bins"]))

    def test_default_uses_every_selected_profile(self):
        profiles = [_profile("WT", 1_000_000) for _ in range(6)]
        result = _build_residue_combination_regions(
            profiles,
            2,
            bin_count=8,
        )
        dataset = result["datasets"][0]
        self.assertEqual(dataset["profile_count"], 6)
        self.assertEqual(dataset["sampled_profile_count"], 6)


if __name__ == "__main__":
    unittest.main()
