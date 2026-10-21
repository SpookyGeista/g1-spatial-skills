"""Evidence-bound metadata for the separate isolated platform component.

No registration here authorizes automatic terrain admission in a mission.
Actual execution uses PlatformSession, never the crawl-only SkillSession.
"""
from .legacy import WORKSPACE

PLATFORM120_SPEC = {
    'name': 'platform120',
    'status': 'validated_bounded_sim_component',
    'source': str(WORKSPACE/'assets/platform_p12_reference_002'),
    'reference_digest': 'c735c50fe2c7027a3bb05f2186f4a3056f045c1bd98dcb4249619600e726391b',
    'permitted_backend': 'isolated_native_pose_v1',
    'session': 'punktnull.skills.platform.PlatformSession',
    'launcher': 'tools/platform_trial.py',
    'autonomous_mission_admission': False,
    'sim_only': True,
    'hardware_validated': False,
    'live_human_validated': False,
    'rise_m': .120,
    'width_m': 1.55,
    'landing_length_m': 2.23,
    'variation_bounds': {'start_y_m': [-.02,.02], 'start_yaw_deg': [-1.,1.]},
    'acceptance_contract': 'platform120_actual_v2 + native_reference_owned_phase_v2 + timing/uncut supplement',
    'accepted_runs': 3,
    'sample_size': 3,
    'evidence': [
        'runs/platform_002_followthrough/evaluation_v3_timing_recording.json',
        'runs/platform_003_yplus_yawplus/evaluation_timing_recording.json',
        'runs/platform_004_yminus_yawminus/evaluation_timing_recording.json',
    ],
    'summary': 'evidence/platform/three_run_summary.json',
    'scope': 'One configured12cm service-platform ascent with entire-foot support and native planner return; not general stairs or integrated building navigation',
}
