"""Conservative, CPU-only local geometry measurements from registered observations.

All inputs are metres in a *gravity aligned*, yaw-heading body frame: x forward,
y left, z up. A tilted torso frame is not valid. The caller owns sensor validity,
deskew/registration, age checks and removal of self returns. No scene, robot pose,
map truth or skill/controller information is read here. A candidate is geometry,
never permission to execute a motion. A floor anchor additionally requires an
explicit observed-support or contact-proprioceptive height datum; scene/global
floor constants are not observations. The datum gates an observed plane fit; it
never creates floor points or free voxels.

Free coverage is evidence at a bounded voxel resolution, not a claim that every
point in continuous space is empty. Missing returns and unvisited cells stay
unknown. In particular, a sensor's dropped/invalid/missing ray must not be made
into a valid maximum-range ray. No temporal fusion is performed in this module.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PatchConfig:
    x_bounds_m: tuple = (.25, 2.25)
    y_bounds_m: tuple = (-.35, .35)
    z_bounds_m: tuple = (-1.5, 2.)
    profile_bin_m: float = .10
    coverage_voxel_m: float = .10
    anchor_length_m: float = .35
    support_height_tolerance_m: float = .03
    surface_tolerance_m: float = .025
    max_plane_rms_m: float = .018
    max_floor_slope: float = .18
    min_surface_points: int = 6
    lateral_bins: int = 3
    min_tread_depth_m: float = .18
    max_internal_tread_depth_m: float = .65
    min_rise_m: float = .07
    max_rise_m: float = .30
    min_risers: int = 2
    max_elevation_change_m: float = 1.
    overhead_search_height_m: float = 1.6
    # Geometric separation of a floor surface from another surface, NOT a
    # calibrated crawl/standing envelope or a declaration of robot suitability.
    surface_separation_m: float = .20

    def __post_init__(self):
        for bounds in (self.x_bounds_m, self.y_bounds_m, self.z_bounds_m):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
                raise ValueError('ROI bounds must be finite increasing pairs')
        numbers = [v for v in vars(self).values() if np.isscalar(v)]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError('Evidence thresholds must be positive and finite')
        if self.support_height_tolerance_m > .05:
            raise ValueError('Support height tolerance must be at most 0.05 m')
        if self.min_risers < 2 or self.lateral_bins < 3 or self.min_surface_points < 3:
            raise ValueError('Insufficient evidence thresholds')
        if self.min_rise_m >= self.max_rise_m or self.anchor_length_m < self.min_tread_depth_m:
            raise ValueError('Inconsistent geometry thresholds')
        spans = np.array([b[1]-b[0] for b in (self.x_bounds_m, self.y_bounds_m, self.z_bounds_m)])
        if np.prod(np.ceil(spans/self.coverage_voxel_m)) > 500_000:
            raise ValueError('Coverage grid exceeds bounded CPU budget')
        if spans[0]/self.profile_bin_m > 500 or self.lateral_bins > 32:
            raise ValueError('Profile exceeds bounded CPU budget')


@dataclass(frozen=True)
class ObservationCoverage:
    """Sensor-derived voxel evidence: -1 unknown, 0 traversed free, 1 occupied.

    origin_m is the lower corner; states axes are x/y/z. Unobserved voxels must
    be -1. Coverage must already be registered in the same frame as the points.
    This adapter never accepts a scene-derived occupancy grid as observations.
    """
    origin_m: object
    voxel_m: float
    states: object


@dataclass(frozen=True)
class RaySamples:
    """Actual registered ray segments, with explicit validity and endpoint type.

    origins_m/endpoints_m have shape (N,3), valid/hit shape (N,). A valid hit
    proves the open segment before the first return; its endpoint is occupied.
    A valid non-hit may be supplied ONLY when the sensor explicitly measured
    the finite segment as clear. Generic missing returns are valid=False.
    """
    origins_m: object
    endpoints_m: object
    valid: object
    hit: object


def _points(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(name+' must have shape (N,3)')
    return a


def _coverage_arrays(coverage):
    origin = np.asarray(coverage.origin_m, dtype=float)
    states = np.asarray(coverage.states)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Invalid coverage origin')
    if not np.isfinite(coverage.voxel_m) or coverage.voxel_m <= 0:
        raise ValueError('Invalid coverage resolution')
    if states.ndim != 3 or not states.size or states.size > 500_000 or not np.isin(states, [-1, 0, 1]).all():
        raise ValueError('Coverage needs a bounded nonempty unknown/free/occupied grid')
    return origin, float(coverage.voxel_m), states


def coverage_from_rays(rays, config=None):
    """Voxelize only explicitly valid finite segments, clipped to the local ROI.

    Occupied endpoints win over traversed free cells. Free evidence stops one
    voxel diagonal before a hit, preserving uncertainty beside the surface.
    This intentionally conservative discretization does not fill ray gaps.
    """
    c = config or PatchConfig()
    origins = _points(rays.origins_m, 'origins_m')
    ends = _points(rays.endpoints_m, 'endpoints_m')
    valid, hit = np.asarray(rays.valid), np.asarray(rays.hit)
    n = len(origins)
    if ends.shape != origins.shape or valid.shape != (n,) or hit.shape != (n,):
        raise ValueError('Ray channel shapes disagree')
    if valid.dtype != np.bool_ or hit.dtype != np.bool_ or n > 100_000:
        raise ValueError('Ray flags must be explicit booleans and batch bounded')
    if not np.isfinite(origins[valid]).all() or not np.isfinite(ends[valid]).all():
        raise ValueError('Valid rays must have finite origins and endpoints')
    bounds = np.asarray([c.x_bounds_m, c.y_bounds_m, c.z_bounds_m])
    lo, hi = bounds[:, 0], bounds[:, 1]
    shape = np.ceil((hi-lo)/c.coverage_voxel_m).astype(int)
    states = np.full(tuple(shape), -1, dtype=np.int8)
    for start, end, occupied in zip(origins[valid], ends[valid], hit[valid]):
        delta = end-start
        length = np.linalg.norm(delta)
        if length <= 1e-9:
            continue
        moving = np.abs(delta) > 1e-12
        if np.any((~moving) & ((start < lo) | (start >= hi))):
            continue
        a = np.full(3, -np.inf)
        b = np.full(3, np.inf)
        a[moving] = (lo[moving]-start[moving])/delta[moving]
        b[moving] = (hi[moving]-start[moving])/delta[moving]
        near = max(0., float(np.minimum(a, b).max()))
        far = min(1., float(np.maximum(a, b).min()))
        free_end = min(far, 1.-np.sqrt(3)*c.coverage_voxel_m/length if occupied else 1.)
        if free_end > near:
            count = int(np.ceil((free_end-near)*length/(c.coverage_voxel_m/3)))+1
            samples = start+np.linspace(near, free_end, count)[:, None]*delta
            indices = np.floor((samples-lo)/c.coverage_voxel_m).astype(int)
            indices = indices[((indices >= 0) & (indices < shape)).all(axis=1)]
            if len(indices):
                key = tuple(indices.T)
                states[key] = np.where(states[key] == 1, 1, 0)
        if occupied and ((end >= lo) & (end < hi)).all():
            states[tuple(np.floor((end-lo)/c.coverage_voxel_m).astype(int))] = 1
    return ObservationCoverage(lo, c.coverage_voxel_m, states)


def _lateral_support(points, c):
    bins = np.floor((points[:, 1]-c.y_bounds_m[0]) /
                    (c.y_bounds_m[1]-c.y_bounds_m[0])*c.lateral_bins).astype(int)
    return len(np.unique(np.clip(bins, 0, c.lateral_bins-1))) == c.lateral_bins


def _lowest_supported(points, heights, c, min_x_span=0.):
    if len(points) < c.min_surface_points:
        return None
    # Search measured clusters bottom-up. A few isolated low returns cannot
    # establish a support surface; count and lateral spread are both required.
    bins = np.unique(np.floor(heights/c.surface_tolerance_m).astype(int))
    for level in bins:
        middle = (level+.5)*c.surface_tolerance_m
        selected = np.abs(heights-middle) <= c.surface_tolerance_m
        p = points[selected]
        if (len(p) >= c.min_surface_points and _lateral_support(p, c) and
                np.ptp(p[:, 0]) >= min_x_span):
            median = float(np.median(heights[selected]))
            selected = np.abs(heights-median) <= c.surface_tolerance_m
            return median, selected
    return None


def _plane(points, c):
    a = np.column_stack((points[:, :2], np.ones(len(points))))
    if len(points) < c.min_surface_points or np.linalg.matrix_rank(a) < 3:
        return None
    coeff = np.linalg.lstsq(a, points[:, 2], rcond=None)[0]
    residual = points[:, 2]-a@coeff
    rms = float(np.sqrt(np.mean(residual**2)))
    if rms > c.max_plane_rms_m or not _lateral_support(points, c):
        return None
    return coeff, rms


def classify_local_patch(points_body_m, *, support_height_body_m=None,
                         support_source=None, rays=None, coverage=None, config=None):
    """Return status/candidate/metrics/reasons without any skill-admission claim.

    status is candidate, unknown (insufficient observation), or rejected
    (measured incompatible/noisy/obstructed geometry). Candidate is floor,
    low_passage, stairs_up or stairs_down, and is None unless status=candidate.
    Floor elevation is fitted from observed near-patch points, and its height
    extrapolated to body x=y=0 must agree with support_height_body_m. The datum
    must come from an independently observed support surface or currently
    established load-bearing contact proprioception, indicated by support_source
    ('observed_support' or 'proprioceptive_contact'). The caller owns freshness,
    uncertainty, frame registration and support-contact validity. A stale or
    ambiguous contact datum must be withheld. A configured scene/global floor is
    not an allowed source. Missing datum/source returns unknown. Body z=0 is
    never assumed to be floor, and the datum does not supply missing points.
    All reported extents are bounded by this ROI; no skill admission occurs.
    """
    c = config or PatchConfig()
    points = _points(points_body_m, 'points_body_m')
    if rays is not None and coverage is not None:
        raise ValueError('Supply rays or coverage, not both')
    if coverage is not None:
        _coverage_arrays(coverage)
    if rays is not None:
        coverage = coverage_from_rays(rays, c)
    metrics = {'input_points': len(points), 'frame': 'gravity_aligned_body_heading',
               'roi_m': [list(c.x_bounds_m), list(c.y_bounds_m), list(c.z_bounds_m)],
               'coverage_kind': 'none' if coverage is None else 'observed_voxels'}

    def result(status, reason, candidate=None):
        return dict(status=status, candidate=candidate, metrics=metrics,
                    reasons=[reason], skill_validated=False)

    if not np.isfinite(points).all():
        return result('rejected', 'nonfinite_points')
    if support_height_body_m is None:
        return result('unknown', 'support_height_datum_unavailable')
    datum = np.asarray(support_height_body_m)
    if datum.shape != () or datum.dtype.kind not in 'fiu' or not np.isfinite(datum):
        raise ValueError('Support height must be one finite measured scalar in metres')
    datum = float(datum)
    if support_source is None:
        return result('unknown', 'support_height_source_unavailable')
    if support_source not in ('observed_support', 'proprioceptive_contact'):
        raise ValueError('Support source must be observed_support or proprioceptive_contact')
    metrics['support_datum'] = {'height_body_m':datum, 'source':support_source,
                                'maximum_fit_difference_m':c.support_height_tolerance_m,
                                'caller_responsible_for_current_valid_support':True}
    bounds = np.asarray([c.x_bounds_m, c.y_bounds_m, c.z_bounds_m])
    points = points[((points >= bounds[:, 0]) & (points < bounds[:, 1])).all(axis=1)]
    metrics['roi_points'] = len(points)
    anchor = points[points[:, 0] < c.x_bounds_m[0]+c.anchor_length_m]
    # The datum is a plane-height intercept at the body origin, not a claim
    # that every observed near point has identical height. Account for the
    # bounded allowable slope, then check the fitted intercept independently.
    height_band = c.support_height_tolerance_m+c.max_floor_slope*np.linalg.norm(anchor[:,:2],axis=1)
    anchor = anchor[np.abs(anchor[:,2]-datum) <= height_band]
    cluster = _lowest_supported(anchor, anchor[:, 2], c)
    if cluster is None:
        return result('unknown', 'observed_support_near_datum_unavailable')
    support = anchor[cluster[1]]
    if np.ptp(support[:, 0]) < c.min_tread_depth_m:
        return result('unknown', 'near_floor_extent_unobserved')
    fitted = _plane(support, c)
    if fitted is None:
        return result('rejected', 'noisy_near_floor')
    plane, rms = fitted
    if np.linalg.norm(plane[:2]) > c.max_floor_slope:
        return result('rejected', 'floor_slope_outside_profile_model')
    metrics['support_datum']['observed_fit_difference_m'] = float(abs(plane[2]-datum))
    if abs(plane[2]-datum) > c.support_height_tolerance_m:
        return result('unknown', 'observed_support_datum_disagreement')
    elevation = points[:, 2]-(points[:, :2]@plane[:2]+plane[2])
    metrics['floor_plane'] = {'z_equals_ax_plus_by_plus_c': plane.tolist(),
                              'rms_m': rms, 'points': len(support)}
    edges = np.arange(c.x_bounds_m[0], c.x_bounds_m[1]+c.profile_bin_m*.5, c.profile_bin_m)
    edges[-1] = c.x_bounds_m[1]
    profiles, surface_indices = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((points[:, 0] >= left) & (points[:, 0] < right) &
                             (np.abs(elevation) <= c.max_elevation_change_m))
        found = _lowest_supported(points[idx], elevation[idx], c, (right-left)*.4)
        if found is None:
            return result('unknown', 'incomplete_floor_or_tread_profile')
        profiles.append(found[0])
        surface_indices.append(idx[found[1]])
    heights = np.array(profiles)
    segments = [[0]]
    for i in range(1, len(heights)):
        if abs(heights[i]-np.median(heights[segments[-1]])) <= 2*c.surface_tolerance_m:
            segments[-1].append(i)
        else:
            segments.append([i])
    treads = []
    for segment in segments:
        idx = np.concatenate([surface_indices[i] for i in segment])
        p = points[idx]
        fitted_tread = _plane(p, c)
        if fitted_tread is None:
            return result('rejected', 'noisy_support_surface')
        if np.linalg.norm(fitted_tread[0][:2]-plane[:2]) > c.max_floor_slope:
            return result('rejected', 'nonparallel_support_surface')
        depth = float(np.ptp(p[:, 0]))
        if depth < c.min_tread_depth_m:
            return result('unknown', 'insufficient_tread_depth_support')
        treads.append(dict(x_start_m=float(p[:, 0].min()), x_end_m=float(p[:, 0].max()),
                           depth_m=depth, elevation_m=float(np.median(elevation[idx])),
                           points=len(p), rms_m=fitted_tread[1]))
    metrics['treads'] = treads
    candidate = 'floor'
    risers = []
    if len(treads) > 1:
        if len(treads)-1 < c.min_risers:
            return result('unknown', 'insufficient_ordered_risers')
        changes = np.diff([t['elevation_m'] for t in treads])
        if (not (np.all(changes > 0) or np.all(changes < 0)) or
                np.any(np.abs(changes) < c.min_rise_m) or np.any(np.abs(changes) > c.max_rise_m)):
            return result('rejected', 'inconsistent_or_out_of_bounds_step_heights')
        for tread in treads[1:-1]:
            if tread['depth_m'] > c.max_internal_tread_depth_m:
                return result('rejected', 'internal_tread_outside_profile_model')
        for i, rise in enumerate(changes):
            edge = float(edges[segments[i+1][0]])
            low, high = sorted((treads[i]['elevation_m'], treads[i+1]['elevation_m']))
            face = points[(abs(points[:, 0]-edge) <= c.surface_tolerance_m*2) &
                          (elevation > low+c.surface_tolerance_m) &
                          (elevation < high-c.surface_tolerance_m)]
            if (len(face) < c.min_surface_points or not _lateral_support(face, c) or
                    np.ptp(face[:, 2]) < abs(rise)*.45):
                return result('unknown', 'riser_face_unobserved')
            risers.append(dict(x_m=edge, rise_m=float(rise), points=len(face),
                               observed_vertical_span_m=float(np.ptp(face[:, 2]))))
        candidate = 'stairs_up' if changes[0] > 0 else 'stairs_down'
    metrics['risers'] = risers

    # Detect a second supported surface over the local support profile.
    columns = np.clip(np.searchsorted(edges, points[:, 0], side='right')-1, 0, len(heights)-1)
    clearance = elevation-heights[columns]
    overhead_ids = []
    overhead_heights = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        idx = np.flatnonzero((points[:, 0] >= left) & (points[:, 0] < right) &
                             (clearance >= c.surface_separation_m) &
                             (clearance <= c.overhead_search_height_m))
        found = _lowest_supported(points[idx], clearance[idx], c, (right-left)*.4)
        if found is not None:
            overhead_heights.append(found[0])
            overhead_ids.extend(idx[found[1]])
    upper = c.overhead_search_height_m
    if overhead_ids:
        if len(overhead_heights) != len(heights):
            return result('rejected', 'partial_overhead_or_obstruction')
        roof = _plane(points[overhead_ids], c)
        if roof is None:
            return result('rejected', 'noisy_overhead_surface')
        if candidate != 'floor':
            return result('unknown', 'combined_stairs_and_overhead_requires_separate_model')
        upper = float(min(overhead_heights))
        metrics['overhead'] = dict(min_clearance_m=upper,
                                   max_clearance_m=float(max(overhead_heights)),
                                   plane=roof[0].tolist(), rms_m=roof[1], points=len(overhead_ids))
        candidate = 'low_passage'
    # Isolated obstacles may not have enough support to form a roof. They still
    # block the sampled corridor; never silently discard them as empty space.
    obstruction = (clearance > c.surface_tolerance_m*2) & (clearance < upper-c.surface_tolerance_m*2)
    for r in risers:
        obstruction &= abs(points[:, 0]-r['x_m']) > c.surface_tolerance_m*2
    if np.any(obstruction):
        metrics['obstruction_points'] = int(obstruction.sum())
        return result('rejected', 'occupied_clearance_volume')
    metrics['observed_geometry'] = candidate
    if coverage is None:
        return result('unknown', 'free_clearance_unobserved')
    origin, voxel, states = _coverage_arrays(coverage)
    if voxel > c.coverage_voxel_m+1e-12:
        return result('unknown', 'coverage_resolution_too_coarse')
    queries = []
    ys = np.arange(c.y_bounds_m[0]+voxel/2, c.y_bounds_m[1], voxel)
    # Exclude the measured surfaces themselves by a voxel diagonal; the gap is
    # reported, not described as observed free. All remaining grid samples must
    # be explicitly free; one absent sample is enough to withhold a candidate.
    margin = np.sqrt(3)*voxel
    dz = np.arange(margin, upper-margin+1e-9, voxel)
    if not len(dz) or not len(ys):
        return result('unknown', 'clearance_below_observation_resolution')
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        for x in np.arange(left+voxel/2, right, voxel):
            for y in ys:
                floor = plane[0]*x+plane[1]*y+plane[2]+heights[i]
                queries.extend((x, y, floor+z) for z in dz)
    q = np.asarray(queries)
    indices = np.floor((q-origin)/voxel).astype(int)
    inside = ((indices >= 0) & (indices < np.array(states.shape))).all(axis=1)
    observed = np.full(len(indices), -1, dtype=np.int8)
    observed[inside] = states[tuple(indices[inside].T)]
    metrics['clearance_coverage'] = dict(samples=len(q), free=int((observed == 0).sum()),
                                        unknown=int((observed == -1).sum()),
                                        occupied=int((observed == 1).sum()), voxel_m=voxel,
                                        surface_exclusion_margin_m=margin,
                                        checked_height_m=upper)
    if np.any(observed == 1):
        return result('rejected', 'occupied_clearance_voxels')
    if np.any(observed == -1):
        return result('unknown', 'incomplete_free_clearance_coverage')
    return result('candidate', 'measured_geometry_with_sampled_free_clearance', candidate)
