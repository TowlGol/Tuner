import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from TopoTunnel_UI.app.main_window import MainWindow
from TopoTunnel_UI.vis.tunnel_viewer import (
    DEFAULT_PATH_COORDINATE_MODE,
    HAS_VTK,
    TunnelViewer3D,
)


class _ProfileDb:
    def __init__(self):
        self.calls = 0

    def get_path_profiles(self, path_ids):
        self.calls += 1
        return [{"pathIndex": int(path_id)} for path_id in path_ids]


class _InventoryDb:
    def __init__(self):
        self.keys = []
        self.paths = 0

    def list_datasets(self):
        return [{"key": key} for key in self.keys]

    def path_count(self):
        return self.paths


class RenderCachingTests(unittest.TestCase):
    def test_original_paths_are_the_default_coordinate_mode(self):
        self.assertEqual(DEFAULT_PATH_COORDINATE_MODE, "original")

    def test_3d_load_signature_changes_when_datasets_are_imported(self):
        window = MainWindow.__new__(MainWindow)
        window.db = _InventoryDb()
        window.state = SimpleNamespace(frame_min=None, frame_max=None)
        window._path_coordinate_mode = DEFAULT_PATH_COORDINATE_MODE

        empty_signature = window._current_3d_load_signature()
        window.db.keys = ["wt"]
        window.db.paths = 448
        imported_signature = window._current_3d_load_signature()
        window.db.keys = ["wt", "r117a"]
        second_dataset_signature = window._current_3d_load_signature()

        self.assertNotEqual(empty_signature, imported_signature)
        self.assertNotEqual(imported_signature, second_dataset_signature)
        self.assertEqual(imported_signature[-1], "original")

    def test_chart_profile_cache_is_order_independent_and_bounded(self):
        window = MainWindow.__new__(MainWindow)
        window.db = _ProfileDb()
        window._chart_profile_cache = OrderedDict()
        window._chart_profile_cache_limit = 2
        window._chart_profile_cache_max_paths = 4

        first, first_hit = window._cached_path_profiles([3, 0, 3])
        second, second_hit = window._cached_path_profiles([0, 3])

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(first, second)
        self.assertEqual([row["pathIndex"] for row in first], [0, 3])
        self.assertEqual(window.db.calls, 1)

        window._cached_path_profiles([4])
        window._cached_path_profiles([5])
        self.assertEqual(len(window._chart_profile_cache), 2)

        oversized = list(range(10, 16))
        window._cached_path_profiles(oversized)
        calls_after_first = window.db.calls
        window._cached_path_profiles(oversized)
        self.assertEqual(window.db.calls, calls_after_first + 1)

    def test_same_tunnel_property_population_is_rendered_only_once(self):
        class _Status:
            def setText(self, text):
                self.text = str(text)

        window = MainWindow.__new__(MainWindow)
        window.db = _ProfileDb()
        window._chart_profiles = []
        window._chart_status_label = _Status()
        window._tunnel_properties_render_signature = None
        window._tunnel_properties_visible = lambda: True
        render_calls = []

        def cached(path_ids):
            return ([{"pathIndex": int(path_id)} for path_id in path_ids], False)

        def render(profiles):
            render_calls.append(len(profiles))
            window._chart_profiles = list(profiles)
            window._tunnel_properties_render_signature = None

        window._cached_path_profiles = cached
        window._set_chart_profiles = render

        path_ids = [10_000_000 + value for value in range(1, 25)]
        MainWindow._sync_tunnel_properties_for_paths(window, path_ids)
        MainWindow._sync_tunnel_properties_for_paths(window, reversed(path_ids))

        self.assertEqual(render_calls, [24])
        self.assertIn("already rendered", window._chart_status_label.text)

    def test_opening_tunnel_properties_only_requests_manual_sync(self):
        class _Status:
            def setText(self, text):
                self.text = str(text)

        window = MainWindow.__new__(MainWindow)
        window.db = _ProfileDb()
        window._chart_profiles = []
        window._chart_status_label = _Status()
        window._tunnel_properties_render_signature = None
        window._active_tunnel_property_path_ids = lambda: (10_000_001, 10_000_002)

        MainWindow._on_tunnel_properties_requested(window)

        self.assertEqual(window.db.calls, 0)
        self.assertIn("click Sync Charts", window._chart_status_label.text)

    @unittest.skipUnless(HAS_VTK, "PyVista/VTK is not installed")
    def test_effective_mesh_cache_reuses_polydata(self):
        viewer = TunnelViewer3D.__new__(TunnelViewer3D)
        viewer._bg_data_dict = {
            1: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            2: np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
        }
        viewer._effective_mesh_cache = OrderedDict()
        viewer._effective_mesh_cache_points = 0
        viewer._effective_mesh_cache_max_points = 100
        viewer._effective_mesh_cache_limit = 4

        mesh_a, paths_a, points_a, hit_a = viewer._effective_mesh_for_paths([2, 1])
        mesh_b, paths_b, points_b, hit_b = viewer._effective_mesh_for_paths([1, 2])

        self.assertFalse(hit_a)
        self.assertTrue(hit_b)
        self.assertIs(mesh_a, mesh_b)
        self.assertEqual((paths_a, points_a), (2, 5))
        self.assertEqual((paths_b, points_b), (2, 5))


if __name__ == "__main__":
    unittest.main()
