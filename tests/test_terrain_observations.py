"""Analytic observation evidence tests, never controller/physical skill tests."""
import unittest
import numpy as np

from punktnull.terrain.observations import (
    ObservationCoverage, PatchConfig, RaySamples, classify_local_patch,
    coverage_from_rays,
)


def surface(z, xlim=(.25, 2.25), ylim=(-.35, .35), noise=0.):
    x = np.arange(xlim[0]+.0125, xlim[1], .025)
    y = np.arange(ylim[0]+.02, ylim[1], .04)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = z(xx, yy) if callable(z) else np.full_like(xx, z)
    if noise:
        zz = zz+np.random.default_rng(12).normal(0, noise, zz.shape)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def observed_free(config=None):
    """Synthetic sensor-evidence fixture; no scene data enters the classifier."""
    c = config or PatchConfig()
    bounds = np.array([c.x_bounds_m, c.y_bounds_m, c.z_bounds_m])
    shape = np.ceil((bounds[:, 1]-bounds[:, 0])/c.coverage_voxel_m).astype(int)
    return ObservationCoverage(bounds[:, 0], c.coverage_voxel_m,
                               np.zeros(tuple(shape), dtype=np.int8))


def stair_observations(sign=1, with_faces=True):
    edges = [.75, 1.05, 1.35]
    zbase, rise = -.82, .15*sign
    points = [surface(lambda x, y: zbase+rise*sum(x >= edge for edge in edges))]
    if with_faces:
        for i, edge in enumerate(edges):
            yy, zz = np.meshgrid(np.arange(-.33, .35, .04),
                                 np.linspace(zbase+rise*i, zbase+rise*(i+1), 13))
            points.append(np.column_stack((np.full(yy.size, edge), yy.ravel(), zz.ravel())))
    return np.concatenate(points)


def classify_supported(points, **kwargs):
    """Synthetic current load-bearing support datum, independent of point fit."""
    kwargs.setdefault('support_height_body_m', -.82)
    kwargs.setdefault('support_source', 'proprioceptive_contact')
    return classify_local_patch(points, **kwargs)


class TerrainObservationsTest(unittest.TestCase):
    def assert_unadmitted(self, result):
        self.assertIs(result['skill_validated'], False)
        if result['status'] != 'candidate':
            self.assertIsNone(result['candidate'])

    def test_floor_fit_uses_measured_datum_and_slope(self):
        points = surface(lambda x, y: -.83+.035*x+.015*y, noise=.001)
        result = classify_supported(points, support_height_body_m=-.83, coverage=observed_free())
        self.assertEqual((result['status'], result['candidate']), ('candidate', 'floor'), result)
        fit = result['metrics']['floor_plane']
        np.testing.assert_allclose(fit['z_equals_ax_plus_by_plus_c'], [.035, .015, -.83], atol=.001)
        self.assertLess(fit['rms_m'], .002)
        self.assert_unadmitted(result)

    def test_absence_of_overhead_returns_is_unknown_without_valid_coverage(self):
        result = classify_supported(surface(-.82))
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reasons'], ['free_clearance_unobserved'])
        self.assertEqual(result['metrics']['observed_geometry'], 'floor')
        self.assert_unadmitted(result)

    def test_low_passage_has_measured_floor_relative_clearance(self):
        points = np.concatenate([surface(-.82, noise=.001), surface(-.10, noise=.001)])
        result = classify_supported(points, coverage=observed_free())
        self.assertEqual((result['status'], result['candidate']), ('candidate', 'low_passage'), result)
        self.assertAlmostEqual(result['metrics']['overhead']['min_clearance_m'], .72, delta=.005)
        self.assert_unadmitted(result)
        # A visible underside alone says nothing about the unseen corridor.
        unsupported = classify_supported(points)
        self.assertEqual(unsupported['status'], 'unknown')

    def test_missing_or_occupied_voxel_cannot_count_as_free(self):
        points = surface(-.82)
        evidence = observed_free()
        for state, expected in [(-1, 'unknown'), (1, 'rejected')]:
            states = evidence.states.copy()
            # A voxel in the middle of the checked corridor.
            point = np.array([1., 0., 0.])
            idx = tuple(np.floor((point-evidence.origin_m)/evidence.voxel_m).astype(int))
            states[idx] = state
            result = classify_supported(points, coverage=ObservationCoverage(
                evidence.origin_m, evidence.voxel_m, states))
            self.assertEqual(result['status'], expected, result)
            self.assert_unadmitted(result)

    def test_ordered_stairs_measure_rise_tread_and_faces_both_directions(self):
        for sign, label in [(1, 'stairs_up'), (-1, 'stairs_down')]:
            result = classify_supported(stair_observations(sign), coverage=observed_free())
            self.assertEqual((result['status'], result['candidate']), ('candidate', label), result)
            metrics = result['metrics']
            self.assertEqual(len(metrics['risers']), 3)
            np.testing.assert_allclose([r['rise_m'] for r in metrics['risers']], [.15*sign]*3, atol=.005)
            np.testing.assert_allclose([t['depth_m'] for t in metrics['treads'][1:-1]], [.275, .275], atol=.03)
            self.assertTrue(all(r['observed_vertical_span_m'] > .07 for r in metrics['risers']))
            self.assert_unadmitted(result)

    def test_height_bands_without_riser_returns_do_not_establish_stairs(self):
        result = classify_supported(stair_observations(with_faces=False), coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        self.assertEqual(result['reasons'], ['riser_face_unobserved'])
        self.assert_unadmitted(result)

    def test_missing_tread_and_single_step_are_unknown(self):
        points = stair_observations()
        points = points[(points[:, 0] < .85) | (points[:, 0] > 1.0)]
        result = classify_supported(points, coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        single = surface(lambda x, y: -.82+.15*(x >= 1.05))
        result = classify_supported(single, coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        self.assertEqual(result['reasons'], ['insufficient_ordered_risers'])

    def test_obstacle_or_partial_roof_cannot_be_declared_clear(self):
        floor = surface(-.82)
        for obstruction in [np.array([[1.12, .1, -.43]]), surface(-.10, xlim=(.75, 1.25))]:
            result = classify_supported(np.concatenate([floor, obstruction]), coverage=observed_free())
            self.assertEqual(result['status'], 'rejected', result)
            self.assert_unadmitted(result)

    def test_noisy_sparse_and_nonfinite_observations_fail_closed(self):
        for points in [surface(-.82, noise=.08), surface(-.82)[::100], np.empty((0, 3)),
                       np.array([[1., 0., np.nan]])]:
            result = classify_supported(points, coverage=observed_free())
            self.assertNotEqual(result['status'], 'candidate', result)
            self.assert_unadmitted(result)

    def test_invalid_and_absent_rays_do_not_invent_free_coverage(self):
        rays = RaySamples([[.3, 0., 0.]], [[2., 0., 0.]], np.array([False]), np.array([False]))
        evidence = coverage_from_rays(rays)
        self.assertTrue(np.all(evidence.states == -1))
        result = classify_supported(surface(-.82), rays=rays)
        self.assertEqual(result['status'], 'unknown')
        self.assertGreater(result['metrics']['clearance_coverage']['unknown'], 0)

    def test_hit_endpoint_and_unseen_far_side_stay_occupied_or_unknown(self):
        c = PatchConfig()
        rays = RaySamples([[.3, 0., 0.], [.3, 0., 0.]],
                          [[1.25, 0., 0.], [2., 0., 0.]],
                          np.array([True, False]), np.array([True, False]))
        evidence = coverage_from_rays(rays, c)
        def at(x):
            idx = tuple(np.floor((np.array([x, 0., 0.])-evidence.origin_m)/evidence.voxel_m).astype(int))
            return evidence.states[idx]
        self.assertEqual(at(.8), 0)
        self.assertEqual(at(1.25), 1)
        self.assertEqual(at(1.8), -1)

    def test_explicit_valid_clear_segments_support_only_traversed_cells(self):
        rays = RaySamples([[.3, 0., 0.]], [[2., 0., 0.]], np.array([True]), np.array([False]))
        evidence = coverage_from_rays(rays)
        self.assertGreater(np.count_nonzero(evidence.states == 0), 0)
        self.assertGreater(np.count_nonzero(evidence.states == -1), 0)
        result = classify_supported(surface(-.82), coverage=evidence)
        self.assertEqual(result['status'], 'unknown')

    def test_missing_support_datum_or_source_is_unknown(self):
        # Even perfect-looking points and a synthetic fully observed free grid
        # cannot decide whether this horizontal surface is floor or ceiling.
        for height in (-.82, -.10):
            result = classify_local_patch(surface(height), coverage=observed_free())
            self.assertEqual(result['status'], 'unknown', result)
            self.assertEqual(result['reasons'], ['support_height_datum_unavailable'])
            self.assertNotIn('floor_plane', result['metrics'])
            self.assert_unadmitted(result)
        result = classify_local_patch(surface(-.82), support_height_body_m=-.82,
                                      coverage=observed_free())
        self.assertEqual(result['reasons'], ['support_height_source_unavailable'])

    def test_ceiling_only_cannot_be_anchored_as_floor(self):
        result = classify_supported(surface(-.10), coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        self.assertEqual(result['reasons'], ['observed_support_near_datum_unavailable'])
        self.assertNotIn('floor_plane', result['metrics'])
        self.assert_unadmitted(result)

    def test_support_datum_does_not_create_unobserved_floor(self):
        result = classify_supported(np.empty((0,3)), coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        self.assertEqual(result['reasons'], ['observed_support_near_datum_unavailable'])
        result = classify_supported(surface(-.55), coverage=observed_free())
        self.assertEqual(result['status'], 'unknown', result)
        self.assert_unadmitted(result)

    def test_observed_support_datum_matches_fitted_intercept_not_near_height(self):
        points = surface(lambda x,y:-.62+.10*x+.04*y)
        result = classify_local_patch(points, support_height_body_m=-.62,
            support_source='observed_support', coverage=observed_free())
        self.assertEqual((result['status'],result['candidate']),('candidate','floor'),result)
        np.testing.assert_allclose(result['metrics']['floor_plane']['z_equals_ax_plus_by_plus_c'], [.10,.04,-.62], atol=1e-8)
        # Points may pass the slope-expanded search band, but an inconsistent
        # fitted support intercept must still be rejected as an anchor.
        result = classify_supported(surface(-.76), coverage=observed_free())
        self.assertEqual(result['status'],'unknown',result)
        self.assertEqual(result['reasons'],['observed_support_datum_disagreement'])
        self.assert_unadmitted(result)

    def test_support_datum_contract_rejects_nonmeasurement_sources(self):
        for datum in (float('nan'), float('inf'), [-.82], True, '-.82'):
            with self.assertRaises(ValueError):
                classify_supported(surface(-.82),support_height_body_m=datum)
        with self.assertRaises(ValueError):
            classify_supported(surface(-.82),support_source='configured_scene_floor')
        with self.assertRaises(ValueError):
            PatchConfig(support_height_tolerance_m=.051)

    def test_invalid_contracts_are_errors(self):
        with self.assertRaises(ValueError):
            classify_supported(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            coverage_from_rays(RaySamples([[0, 0, 0]], [[1, 0, 0]], [1], [0]))
        with self.assertRaises(ValueError):
            coverage_from_rays(RaySamples([[0, 0, 0]], [[np.nan, 0, 0]], [True], [False]))
        with self.assertRaises(ValueError):
            PatchConfig(coverage_voxel_m=.001)


if __name__ == '__main__':
    unittest.main()
