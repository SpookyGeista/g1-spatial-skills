#!/usr/bin/env python3
"""Explicit simulation-only recorded-POSE review within a complete mission.

Reuses the retained capture validator, PoseGateway and operator-command channel.
It launches no simulator/controller and drives no robot motors directly. A quiet
saved body capture tests native source switching; it is not a live human or a
useful physical recovery. --check opens no sockets. Execution requires an
already running mission with the explicit operator-review injection in its
manifest. Evaluation reads retained evidence only and never changes its outcome.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from punktnull.retained import EXPLORATION, load, prepare
prepare()
retained = load('_punktnull_retained_review_harness', EXPLORATION/'replay_handover.py')
from tools.evaluate_g1_lio import trace_audit

LABEL = 'Recorded quiet POSE during explicitly injected review; no live human or useful recovery'
ROOMS = {'west', 'east', 'north'}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_manifest(manifest):
    if (manifest.get('test') != 'explore' or manifest.get('sim_only') is not True
            or manifest.get('real_actuators_enabled') is not False
            or manifest.get('injection_kind') != 'explicit_test_operator_review'
            or manifest.get('injected_review_after_room') not in ROOMS):
        raise ValueError('Explicit simulation-only exploration operator-review injection required')
    if manifest.get('live_human_tested') is not False:
        raise ValueError('Recorded input must not be described as a live human test')


def validate_run_dir(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'runs') or path == ROOT/'runs':
        raise ValueError('Use a dedicated existing run under this workspace runs/')
    return path


def one_event(events, event, **fields):
    found = [e for e in events if e.get('event') == event and all(e.get(k) == v for k, v in fields.items())]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one {event} {fields}; found {len(found)}')
    return found[0]


def native_rows(run):
    tables = []
    for field, key in [('motion_name', 'motion_name'), ('encoder_mode', 'encoder_mode_0'),
                       ('motion_playing', 'playing_0')]:
        table = {}
        with (run/'controller'/(field+'.csv')).open() as handle:
            for row in csv.DictReader(handle):
                index = int(row['index']); stamp = float(row['time_monotonic_ms'])/1000
                if index in table or not np.isfinite(stamp):
                    raise ValueError('Duplicate/nonfinite native feedback')
                table[index] = (stamp, row[key].strip('"'))
        tables.append(table)
    ids = sorted(set.intersection(*(set(t) for t in tables)))
    result = []
    for index in ids:
        values = [t[index] for t in tables]
        if np.ptp([v[0] for v in values]) > .002:
            raise ValueError('Native feedback tables not time-aligned')
        result.append((index, values[0][0], values[0][1], int(float(values[1][1])), int(float(values[2][1]))))
    if len(result) < 5 or np.any(np.diff([r[1] for r in result]) <= 0):
        raise ValueError('Missing or nonmonotonic native feedback')
    return result


def native_audit(rows, ready, pending, operator, returned, resumed, end):
    phases = {}
    for label, a, b, name, mode in [
            ('before', ready, pending, 'planner_motion', 0),
            ('operator', operator+.10, returned, 'streamed', 2),
            ('resumed', resumed, end, 'planner_motion', 0)]:
        selected = [r for r in rows if a <= r[1] <= b]
        phases[label] = {'samples': len(selected), 'expected_motion': name, 'expected_encoder_mode': mode,
            'pass': bool(len(selected) >= 5 and all(r[2:] == (name, mode, 1) for r in selected)
                         and np.all(np.diff([r[0] for r in selected]) == 1))}
    active = [r for r in rows if ready <= r[1] <= end]
    # RETURN_CHECK includes the deliberate native transition back to planner.
    unexpected = [r for r in active if (r[2] == 'streamed' or r[3] == 2)
                  and not pending-.05 <= r[1] <= resumed]
    nonsteady = [r for r in active if r[3] not in (0, 2)
                 or (r[2], r[3]) not in (('planner_motion', 0), ('streamed', 2))]
    # The retained ZMQ endpoint deliberately pauses a copy of the outgoing
    # reference, named temporary_motion, while its planner buffer is disabled
    # and the first POSE chunk is validated. This is not another command owner.
    # Admit only one directly bracketed, <=100 ms / <=5-sample block wholly in
    # TAKEOVER_PENDING. No temporary-motion exception applies in steady phases
    # or RETURN_CHECK, and even the temporary reference must remain paused.
    held = [r for r in nonsteady if r[2:] == ('temporary_motion', 0, 0)]
    hold = {'samples': len(held), 'maximum_samples': 5, 'maximum_duration_s': .10,
            'owner_phase': 'TAKEOVER_PENDING', 'accepted': False,
            'basis': 'Retained zmq_endpoint_interface.hpp safety reset copies current_motion under mutex, pauses playback, disables planner; first valid stream replaces it under mutex'}
    accepted_ids = set()
    if held:
        first = next(i for i, r in enumerate(rows) if r[0] == held[0][0])
        last = next(i for i, r in enumerate(rows) if r[0] == held[-1][0])
        before = rows[first-1] if first > 0 else None
        following = rows[last+1] if last+1 < len(rows) else None
        contiguous = len(held) == last-first+1 and bool(np.all(np.diff([r[0] for r in held]) == 1))
        duration = following[1]-held[0][1] if following else None
        bracketed = (before is not None and following is not None
            and before[0]+1 == held[0][0] and held[-1][0]+1 == following[0]
            and before[2:] == ('planner_motion', 0, 1)
            and following[2:] == ('streamed', 2, 1)
            and pending <= held[0][1] <= held[-1][1] < following[1] <= operator)
        accepted = bool(contiguous and bracketed and len(held) <= 5 and duration is not None and 0 < duration <= .10+1e-9)
        hold.update(accepted=accepted, first_native_index=held[0][0], last_native_index=held[-1][0],
                    first_monotonic=held[0][1], last_monotonic=held[-1][1], duration_until_stream_s=duration,
                    immediately_bracketed_by_planner_then_stream=bool(bracketed))
        if accepted: accepted_ids = {r[0] for r in held}
    invalid_modes = [r for r in nonsteady if r[0] not in accepted_ids]
    return {'phases': phases, 'unexpected_streamed_samples': len(unexpected),
            'nonsteady_source_mode_samples': len(nonsteady), 'bounded_native_transition_hold': hold,
            'invalid_source_mode_samples': len(invalid_modes),
            'pass': bool(all(p['pass'] for p in phases.values()) and not unexpected and not invalid_modes)}


def room_scan_audit(manifest, events, data, columns, metadata):
    declared = manifest.get('completed_room_scans', [])
    if len(declared) != 3 or {r['id'] for r in declared} != ROOMS:
        return {'pass': False, 'reason': 'Three distinct completed room scans are required', 'rooms': []}
    starts = [e for e in events if e.get('event') == 'scan_started']
    completed = [e for e in events if e.get('event') == 'scan_completed']
    bounds = {r['id']: np.asarray(r['bounds_xy'], float) for r in metadata['scene']['rooms']}
    wall = data[:, columns['wall_monotonic']]
    reports = []
    for room in declared:
        number = int(room['scan_number'])
        if number < 1 or number > len(completed):
            raise ValueError('Declared room scan has no completed scan event')
        finish = completed[number-1]
        prior = [e for e in starts if e['monotonic'] < finish['monotonic']]
        if not prior: raise ValueError('Completed room scan has no preceding start')
        start = prior[-1]
        masked = (wall >= start['monotonic']) & (wall <= finish['monotonic'])
        rows = data[masked]
        if len(rows) < 100: raise ValueError('Insufficient physical room scan samples')
        xy = rows[:, [columns['root_0'], columns['root_1']]]
        requested = np.asarray(room['observation_xy_m'], float)
        measured_start_distance = float(np.linalg.norm(xy[0]-requested))
        box = bounds[room['id']]
        inside = np.all((xy >= box[0]) & (xy <= box[1]), axis=1)
        q = rows[:, [columns['root_'+str(i)] for i in range(3, 7)]]
        yaw = np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2)))
        measured_turn = float(yaw[-1]-yaw[0])
        # Runtime uses estimated orientation. This independent check retains
        # 0.20 rad tolerance for the saved full-rate physical scan interval.
        passed = (bool(inside.all()) and measured_start_distance <= .45
                  and measured_turn >= 2*np.pi-.20
                  and finish.get('sensor_frames', 0) >= 20)
        reports.append({'id': room['id'], 'completed_scan_number': number,
            'actual_root_start_xy_m': xy[0].tolist(), 'actual_observation_distance_m': measured_start_distance,
            'actual_full_rate_yaw_radians': measured_turn, 'physical_room_inside_fraction': float(inside.mean()),
            'sensor_frames_reported': finish.get('sensor_frames'), 'samples': len(rows), 'pass': bool(passed)})
    return {'pass': all(r['pass'] for r in reports), 'rooms': reports,
            'meaning': 'Actual saved full-rate pelvis positions and yaw during the three recorded scans; no complete camera coverage or general exploration claim'}


def audit(run):
    run = Path(run)
    manifest = read_json(run/'manifest.json'); validate_manifest(manifest)
    events = manifest['events']; metadata = read_json(run/'sim/metadata.json')
    review = one_event(events, 'navigation_limit', status='OPERATOR_REVIEW_REQUIRED')
    pending_event = one_event(events, 'transition', state='TAKEOVER_PENDING')
    operator_event = one_event(events, 'transition', state='OPERATOR_CONTROL')
    returned_event = one_event(events, 'transition', state='RETURN_CHECK')
    resumed_event = one_event(events, 'autonomy_resumed')
    ended_event = one_event(events, 'mission_result')
    ready, pending, operator, returned, resumed, end = [float(e['monotonic']) for e in
        [review, pending_event, operator_event, returned_event, resumed_event, ended_event]]
    if not ready < pending < operator < returned < resumed < end:
        raise ValueError('Review/takeover/release/resume/mission sequence is not ordered')
    native = native_audit(native_rows(run), ready, pending, operator, returned, resumed, end)
    with np.load(run/'sim/trace.npz', allow_pickle=False) as saved:
        data = saved['data']; names = [str(n) for n in saved['columns']]
    columns = {n: i for i, n in enumerate(names)}
    physics, physics_checks = trace_audit(run, manifest, events, metadata)
    scans = room_scan_audit(manifest, events, data, columns, metadata)
    wall = data[:, columns['wall_monotonic']]
    xyz = data[:, [columns['root_'+str(i)] for i in range(3)]]
    active = (wall >= ready) & (wall <= end)
    after = (wall >= resumed) & (wall <= end)
    if after.sum() < 2: raise ValueError('No measured post-resume continuation')
    post = xyz[after, :2]
    continuation = float(np.linalg.norm(post-post[0], axis=1).max())
    max_step = float(np.linalg.norm(np.diff(xyz[active], axis=0), axis=1).max())
    release_window = (wall >= returned-.60) & (wall <= returned)
    release_rows = data[release_window]
    speed = np.linalg.norm(release_rows[:, [columns['root_'+str(i)] for i in range(7, 10)]], axis=1)
    max_dq = np.max(abs(release_rows[:, [columns['dq_'+str(i)] for i in range(29)]]), axis=1)
    release_stable = ((release_rows[:, columns['root_2']] >= .70)
        & (release_rows[:, columns['root_2']] <= .90) & (release_rows[:, columns['tilt_deg']] <= 12)
        & (release_rows[:, columns['support']] == 0) & (speed <= .13) & (max_dq <= 2.))
    source = read_json(run/'mission_review_replay/source.json')
    target = manifest.get('target_observation') or {}
    checks = {'explicit_simulation_review_only': True,
        'final_outcome_is_found_and_returned': manifest.get('outcome') == 'FOUND_AND_RETURNED',
        'visual_target_report_retained': target.get('target_id') == 7 and len(set(target.get('confirming_frames', []))) >= 3,
        'final_native_source_sequence_exclusive': native['pass'],
        'scene_reset_not_requested': resumed_event.get('scene_reset') is False,
        'quiet_recorded_source_declared': source.get('source') == 'recorded_pose_replay' and source.get('live_human_acceptance') is False,
        'release_after_observed_stability': len(release_rows) >= 100 and bool(release_stable.all()),
        'measured_post_resume_motion_at_least_020m': continuation >= .20,
        'no_large_base_jump_during_continuation': max_step < .10,
        'three_actual_completed_room_scans': scans['pass'],
        'trace_clock_and_support_preserved': physics_checks['full_rate_monotonic_physics'] and physics_checks['support_removed_no_reattach'],
        'actual_upright_return_and_settle': physics_checks['bounded_upright_physics'] and physics_checks['physical_return_in_exit_zone'] and physics_checks['terminal_stable_full_window'],
        'cleanup_complete_without_error': manifest.get('cleanup_errors') == [] and metadata.get('error') in (None, ''),
        'no_live_human_claim': manifest.get('live_human_tested') is False}
    paths = [run/'manifest.json', run/'sim/trace.npz', run/'sim/metadata.json', run/'mission_review_replay/source.json',
             *[run/'controller'/(f+'.csv') for f in ('motion_name', 'encoder_mode', 'motion_playing')]]
    return {'native_replay_review_pass': bool(all(checks.values())), 'live_human_acceptance': False,
        'useful_operator_recovery_demonstrated': False, 'evidence_kind': LABEL,
        'checks': checks, 'native': native, 'room_scans': scans, 'physics': physics,
        'post_resume_max_displacement_from_release_m': continuation, 'max_full_rate_base_step_m': max_step,
        'release_stability_samples': len(release_rows),
        'does_not_prove': ['live human intervention', 'useful obstacle recovery', 'real G1 operation',
                          'FAST-LIO accuracy or the full separate mission acceptance contract'],
        'source_sha256': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_revision': '2: source-verified bounded paused temporary_motion during TAKEOVER_PENDING; added after original run010 audit, original summary preserved'}


class ReviewHarness:
    def __init__(self, run, gateway, provenance):
        self.run = validate_run_dir(run)
        manifest = read_json(self.run/'manifest.json'); validate_manifest(manifest)
        if manifest.get('outcome') != 'RUNNING': raise ValueError('Mission must already be running')
        self.output = self.run/'mission_review_replay'; self.output.mkdir()
        self.gateway = gateway; self.provenance = dict(provenance, diagnostic_kind=LABEL)
        self.sent_takeover = False
        retained.runtime.atomic_json(self.output/'source.json', self.provenance)
        (self.output/'harness.py').write_bytes(Path(__file__).read_bytes())

    def record(self, event, **details):
        value = {'event': event, 'monotonic': time.monotonic(), **details}
        with (self.output/'events.jsonl').open('a') as out: out.write(json.dumps(value, allow_nan=False)+'\n')
        print('RECORDED_MISSION_REVIEW '+event, flush=True)

    def wait(self, predicate, seconds, check_source=False):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            manifest = read_json(self.run/'manifest.json')
            validate_manifest(manifest)
            if manifest.get('outcome') == 'FAILED': raise RuntimeError('Mission failed: '+str(manifest.get('failure')))
            if predicate(manifest): return manifest
            if check_source:
                status = self.gateway.status()
                if status['last_error']: raise RuntimeError('Recorded source failed: '+status['last_error'])
            time.sleep(.025)
        raise TimeoutError('Bounded recorded mission review wait expired')

    def snapshot(self):
        state = read_json(self.run/'sim_state.json')
        error = retained.mission_predicates.state_error(state, time.monotonic())
        if error: raise RuntimeError(error)
        stable = retained.mission_predicates.stable(state)
        self.record('measured_state', sim_time=state['sim_time'], base=state['base'],
                    tilt_deg=state['tilt_deg'], stable=stable, support=state.get('support'))
        return state, stable

    def execute(self):
        result = {'native_replay_review_pass': False, 'live_human_acceptance': False,
                  'useful_operator_recovery_demonstrated': False, 'evidence_kind': LABEL}
        started = False
        try:
            self.wait(lambda m: any(e.get('event') == 'navigation_limit' and e.get('status') == 'OPERATOR_REVIEW_REQUIRED'
                                   for e in m['events']), 240)
            self.snapshot(); self.gateway.start(); started = True; self.gateway.set_enabled(True)
            time.sleep(.25)  # Existing supervisor's SUB joins while planner remains owner.
            retained.runtime.post_command(self.run, 'takeover'); self.sent_takeover = True
            self.record('takeover_command', source='automated recorded review test')
            self.wait(lambda m: any(e.get('event') == 'transition' and e.get('state') == 'OPERATOR_CONTROL'
                                   for e in m['events']), 12, check_source=True)
            self.record('native_operator_acknowledged', input='quiet saved POSE, no live human')
            stable_since = None; deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                state, stable = self.snapshot()
                stable = stable and state.get('stable_full_rate_sim_seconds', 0) >= .70
                stable_since = (stable_since or time.monotonic()) if stable else None
                if stable_since is not None and time.monotonic()-stable_since >= 2: break
                if self.gateway.status()['last_error']: raise RuntimeError('Recorded POSE source failed')
                time.sleep(.10)
            else: raise TimeoutError('No two-second observed stable operator window')
            retained.runtime.post_command(self.run, 'release'); self.record('release_command', source='explicit automated replay command')
            self.wait(lambda m: any(e.get('event') == 'autonomy_resumed' for e in m['events']), 15, check_source=True)
            self.snapshot(); self.record('native_planner_resumed'); self.gateway.stop(); started = False
            self.wait(lambda m: m.get('outcome') == 'FOUND_AND_RETURNED', 300)
            self.wait(lambda m: 'cleanup_errors' in m and ('native_map_saved' in m or 'map_save_error' in m)
                      and (self.run/'sim/trace.npz').is_file() and (self.run/'sim/metadata.json').is_file(), 45)
            result = audit(self.run)
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            self.record('review_diagnostic_failed', reason=result['error'])
            if self.sent_takeover:
                retained.runtime.post_command(self.run, 'abort')
        finally:
            if started: self.gateway.stop()
            result['recorded_source'] = self.provenance; result['gateway_final'] = self.gateway.status()
            retained.runtime.atomic_json(self.output/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['native_replay_review_pass'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-name', default='summary_reaudit.json')
    args = parser.parse_args()
    if args.evaluate:
        if args.run_dir is None: parser.error('--evaluate requires --run-dir')
        run = validate_run_dir(args.run_dir)
        if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.json'):
            parser.error('Output name must be one JSON filename')
        output = run/'mission_review_replay'/args.output_name
        if output.exists(): parser.error('Refusing to overwrite prior evidence')
        try: result = audit(run)
        except Exception as exc: result = {'native_replay_review_pass': False, 'live_human_acceptance': False,
            'useful_operator_recovery_demonstrated': False, 'evidence_kind': LABEL, 'error': f'{type(exc).__name__}: {exc}'}
        result['evaluation_only'] = True
        with output.open('x') as handle: json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
        print(json.dumps(result, indent=2))
        return 0 if result['native_replay_review_pass'] else 2
    gateway, provenance = retained.capture_check()
    if args.check:
        print(json.dumps({**provenance, 'static_capture_validation_pass': True, 'sockets_opened': False,
                          'diagnostic_kind': LABEL}, indent=2)); return 0
    if args.run_dir is None: parser.error('--execute requires --run-dir')
    return ReviewHarness(args.run_dir, gateway, provenance).execute()


if __name__ == '__main__':
    raise SystemExit(main())
