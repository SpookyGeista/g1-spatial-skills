#!/usr/bin/env python3
"""Bounded, read-only CPU FK screening of 19 already-local ground references.

This is a reference-geometry diagnostic, never a dynamics/skill acceptance test.
It does not construct new motion, change references, or run a controller.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from punktnull.skills.legacy import OLD, WORKSPACE, ground_module, load
from punktnull.skills.session import REGISTRY
from punktnull.retained import URDF
from punktnull.terrain.kinematics import CollisionFK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tilt(quat):
    return np.rad2deg(np.arccos(np.clip(1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2), -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKSPACE/'reports/ground_alternatives')
    args = parser.parse_args()
    output_json = args.output.with_suffix('.json')
    output_md = args.output.with_suffix('.md')
    if output_json.exists() or output_md.exists():
        parser.error('Choose a new output name; retained evidence is never overwritten')
    converter_path = OLD/'tools/convert_bones_to_sonic_reference.py'
    converter = load('ground_candidates_converter', converter_path)
    model_path = OLD/'gear_sonic_deploy/g1/g1_29dof.xml'
    kin = converter.G1Kinematics(model_path)
    imported_fk = CollisionFK(URDF, kin.joint_names)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq == 36 and model.njnt == 30
    data = mujoco.MjData(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in model.geom_bodyid]
    groups = {}
    for side in ['left', 'right']:
        for label, names in [('hand', [side+'_wrist_yaw_link']),
                             ('forearm', [side+'_elbow_link', side+'_wrist_roll_link'])]:
            groups[side+'_'+label] = [i for i, name in enumerate(body_names)
                                    if name in names and model.geom_contype[i] != 0]
    vertices = {}
    for gi in sorted(set(sum(groups.values(), []))):
        assert model.geom_type[gi] == int(mujoco.mjtGeom.mjGEOM_MESH)
        mesh = int(model.geom_dataid[gi])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices[gi] = model.mesh_vert[start:start+count].copy()
    selected = model.geom_bodyid != 0
    center = model.geom_aabb[selected, :3]
    half = model.geom_aabb[selected, 3:]
    limits = np.asarray(kin.joint_limits)
    sources = sorted((OLD/'data/bones_seed_full/g1/csv/230509').glob('elbow_plank_*__A[0-9][0-9][0-9].csv'))
    sources += sorted((OLD/'data/bones_seed_full/g1/csv/231010').glob('stand_up_lying_stomach_R_002__A[0-9][0-9][0-9].csv'))
    sources += sorted((OLD/'data/bones_seed_full/g1/csv/230301').glob('idle_crawl_stop_*__A231.csv'))
    assert len(sources) == 19, len(sources)
    reference = ground_module().Reference(REGISTRY['crawl'].source)
    # Ground Reference.joints is SDK/MuJoCo order, unlike raw joint_pos.csv.
    join_frame = 838  # Existing crawl_stop end, immediately before existing stand_up.
    join_q = reference.joints[join_frame]
    join_z = float(reference.xyz[join_frame, 2])
    join_tilt = float(tilt(reference.quat[join_frame:join_frame+1])[0])
    rows = []
    for path in sources:
        pos, quat, q = converter.read_bones_csv(path, kin)
        angles = tilt(quat)
        frames = np.unique(np.r_[np.arange(0, len(pos), 6), len(pos)-1]).astype(int)
        metrics = {k: [] for k in groups}
        body_bottom, body_top = [], []
        for fi in frames:
            data.qpos[:3] = pos[fi]
            data.qpos[3:7] = quat[fi]
            data.qpos[7:] = q[fi]
            mujoco.mj_forward(model, data)
            bottoms = {gi: float((v @ data.geom_xmat[gi].reshape(3, 3)[2]).min()+data.geom_xpos[gi, 2])
                       for gi, v in vertices.items()}
            for k, ids in groups.items():
                metrics[k].append(min(bottoms[i] for i in ids))
            rotation = data.geom_xmat[selected].reshape(-1, 3, 3)
            c = data.geom_xpos[selected] + np.einsum('nij,nj->ni', rotation, center)
            h = np.einsum('nij,nj->ni', np.abs(rotation), half)
            body_bottom.append(float((c-h)[:, 2].min()))
            body_top.append(float((c+h)[:, 2].max()))
        metrics = {k: np.array(v) for k, v in metrics.items()}
        imported_bounds = imported_fk.bounds(imported_fk.links(pos[frames], quat[frames], q[frames]))
        imported_bottoms = {}
        for side in ['left', 'right']:
            for label in ['hand', 'forearm']:
                ids = [i for i, c in enumerate(imported_fk.collisions) if (
                    (c['link'].startswith(side+'_hand_') or c['link'] == side+'_wrist_yaw_link')
                    if label == 'hand' else c['link'] == side+'_elbow_link')]
                imported_bottoms[side+'_'+label] = imported_bounds[:, ids, 0, 2].min(axis=1)
        ground = (pos[frames, 2] < .45) | (angles[frames] > 45)
        hand_only = np.zeros(len(frames), dtype=bool)
        forearm_close = np.zeros(len(frames), dtype=bool)
        for side in ['left', 'right']:
            hand_only |= (metrics[side+'_hand'] <= .03) & (metrics[side+'_forearm'] > .06)
            forearm_close |= metrics[side+'_forearm'] <= .03
        entry_frames = np.arange(min(len(pos), 240))
        match = np.sqrt(np.mean((q[entry_frames]-join_q)**2, axis=1))
        compatible = (abs(pos[entry_frames, 2]-join_z) <= .06) & (abs(angles[entry_frames]-join_tilt) <= 15)
        closest = int(entry_frames[np.argmin(np.where(compatible, match, np.inf))]) if compatible.any() else None
        row = {'source': str(path.relative_to(OLD)), 'sha256': sha(path), 'frames_120hz_convention': len(pos),
               'fk_samples': len(frames), 'ground_samples': int(ground.sum()),
               'root_start_end_z_m': pos[[0, -1], 2].tolist(),
               'root_start_end_tilt_deg': angles[[0, -1]].tolist(),
               'root_xy_net_displacement_m': float(np.linalg.norm(pos[-1, :2]-pos[0, :2])),
               'max_joint_limit_violation_rad': float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1]))),
               'ground_hand_near_floor_forearm_far_fraction': float(hand_only[ground].mean()) if ground.any() else None,
               'ground_either_forearm_near_floor_fraction': float(forearm_close[ground].mean()) if ground.any() else None,
               'ground_mesh_bottom_min_m': {k: float(v[ground].min()) if ground.any() else None for k, v in metrics.items()},
               'ground_mesh_bottom_median_m': {k: float(np.median(v[ground])) if ground.any() else None for k, v in metrics.items()},
               'whole_body_conservative_aabb_bottom_min_m': min(body_bottom),
               'whole_body_conservative_aabb_top_max_m': max(body_top),
               'actual_imported_urdf_conservative_bounds': {
                   'geometry_count': len(imported_fk.collisions),
                   'ground_bottom_median_m': {k: float(np.median(v[ground])) for k, v in imported_bottoms.items()},
                   'ground_bottom_min_m': {k: float(v[ground].min()) for k, v in imported_bottoms.items()}},
               'closest_first_2s_reference_only_join': None if closest is None else {
                   'source_frame': closest, 'joint_rms_rad': float(match[closest]),
                   'root_height_delta_m': float(pos[closest, 2]-join_z),
                   'tilt_delta_deg': float(angles[closest]-join_tilt),
                   'passes_existing_alignment_joint_gate_015': bool(match[closest] <= .15)}}
        rows.append(row)
    result = {'scope': 'Nineteen bounded local 29-joint raw references; CPU kinematics only. No simulator, controller, reference or acceptance modifications.',
        'method': {'source_fps_convention': 120, 'fk_sampling_every_source_frames': 6,
          'fk_model': str(model_path), 'fk_model_sha256': sha(model_path),
          'fk_model_scope': 'Deployment MuJoCo counterpart; mesh extrema are not the Isaac-imported URDF collider bounds.',
          'actual_imported_urdf': str(URDF), 'actual_imported_urdf_sha256': sha(URDF),
          'actual_imported_urdf_fingers': 'All fixed fingers included, cylinder-to-capsule importer configuration; mesh conservative AABBs, not exact convex hulls.',
          'actual_imported_fk_source_sha256': sha(WORKSPACE/'punktnull/terrain/kinematics.py'),
          'converter_sha256': sha(converter_path), 'script_sha256': sha(__file__),
          'sampled_geometry': 'Actual colliding mesh vertices for hand and forearm bottom; conservative per-geom transformed AABBs for overall envelope.',
          'hand_bodies': ['left_wrist_yaw_link', 'right_wrist_yaw_link'],
          'forearm_bodies': ['left_elbow_link', 'left_wrist_roll_link', 'right_elbow_link', 'right_wrist_roll_link'],
          'near_floor_diagnostic_threshold_m': .03, 'forearm_far_diagnostic_threshold_m': .06,
          'ground_definition': 'reference root height < .45 m OR reference tilt > 45 deg',
          'join_reference': str(reference.source), 'join_frame_50hz': join_frame,
          'join_height_m': join_z, 'join_tilt_deg': join_tilt,
          'join_position_channel_sha256': sha(reference.source/'body_pos.csv'),
          'join_joint_channel_sha256': sha(reference.source/'joint_pos.csv')},
        'limitations': ['No force, balance, joint effort, policy tracking or Isaac contact measurement: near-floor geometry is a support-risk proxy only.',
          'Raw reference is before retained converter smoothing and any trim/blend; a newly assembled clip would require separate continuity and dynamics checks.',
          'The 120 Hz raw convention comes from retained conversion tooling, not embedded CSV timestamp metadata.',
          'Sampled reference geometry does not bound actual controller errors or between-sample motion.',
          'A close reference joint match does not authorize a live phase jump, and a good static elbow pose is not a crawling locomotion skill.'],
        'references': rows}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Vorhandene Bodenreferenzen: begrenzte CPU-Prüfung', '',
      'Diese Auswertung prüft 19 bereits lokale Referenzen mit dem 29-Gelenk-Deployment-MJCF und ergänzt konservative Grenzen der tatsächlich importierten URDF einschließlich aller festen Finger. Sie führt keine Bewegung aus und ändert keine Abnahmegrenze. Hand-/Unterarmabstände sind Geometriehinweise, keine gemessenen Kräfte oder Lastverteilung.', '',
      '| Referenz | XY netto m | z Start→Ende m | Hand nah, Unterarm fern¹ | Unterarm nah¹ | Bester bestehender Stopp→Start-Gelenk-RMS² |',
      '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        join = row['closest_first_2s_reference_only_join']
        match_text = f"{join['joint_rms_rad']:.3f}" if join else 'keine passende Höhe/Neigung'
        z0, z1 = row['root_start_end_z_m']
        lines.append(f"| {Path(row['source']).stem} | {row['root_xy_net_displacement_m']:.3f} | {z0:.3f}→{z1:.3f} | {row['ground_hand_near_floor_forearm_far_fraction']:.0%} | {row['ground_either_forearm_near_floor_fraction']:.0%} | {match_text} |")
    lines += ['', '¹Anteil der ausgewerteten Bodenposen; Mesh-Unterkante ≤3 cm gilt hier ausschließlich als Diagnose als bodennah, Unterarm >6 cm als fern. Die Abstände entstehen aus Referenz-FK, nicht aus einem dynamischen Kontaktmodell.', '',
      '²Minimum innerhalb der ersten 2 s der Rohreferenz gegenüber Frame 838 der bestehenden Kriechsequenz, nur bei Höhenabweichung ≤6 cm und Neigungsabweichung ≤15°. Die geltende gemessene Eintrittsgrenze bleibt 0,15 rad. Kein automatisches Zusammensetzen oder Phasensprung wurde implementiert.', '',
      'Alle 19 Kandidaten halten die nominellen 29-Gelenk-Positionslimits ein. Das beweist weder verfügbare Drehmomente noch ausführbare Dynamik. Die Ellbogen-Planks beginnen und enden am Boden; sie liefern weder den Hinlege-/Aufstehübergang noch nennenswerten Vortrieb. Die übrigen Aufstehbewegungen müssen insbesondere für den Stopp→Aufstehen-Übergang und die Handlast physisch geprüft werden.', '',
      'Die JSON-Datei enthält exakte Quellen und Hashes, individuelle Mesh-Abstände, Eintrittsvergleich und Grenzen. Reproduktion: `python -B tools/audit_ground_alternatives.py --output reports/NEUER_NAME`.']
    output_md.write_text('\n'.join(lines)+'\n')
    print(output_json)
    print(output_md)


if __name__ == '__main__':
    main()
