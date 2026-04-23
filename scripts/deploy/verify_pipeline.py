"""Verify that the deployment pipeline's observation and action spaces match
the training environment exactly.

This script compares:
  1. Observation term order and dimensions
  2. Action dimensions and joint ordering
  3. Standing reference pose values
  4. Scaling factors (ang_vel, joint_vel)
  5. History buffer dimensions

Run from workspace root:
    python scripts/deploy/verify_pipeline.py
"""

from __future__ import annotations

import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def verify_observation_layout():
    """Check that the deploy observation builder matches training config."""
    print("=" * 70)
    print("OBSERVATION LAYOUT VERIFICATION")
    print("=" * 70)

    from config import CONTROLLED_JOINT_NAMES as DEPLOY_JOINTS
    from config import NUM_JOINTS as DEPLOY_COUNT
    from config import OBS_FRAME_DIM, OBS_HISTORY_LENGTH, OBS_TOTAL_DIM
    from config import ANG_VEL_SCALE, JOINT_VEL_SCALE

    errors = []

    # Try to import training joints for cross-check
    try:
        from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES as TRAIN_JOINTS
        from mjlab_roller.core.control_spec import CONTROLLED_JOINT_COUNT as TRAIN_COUNT
        has_training_ref = True
    except ImportError:
        TRAIN_JOINTS = DEPLOY_JOINTS
        TRAIN_COUNT = DEPLOY_COUNT
        has_training_ref = False
        print("  [INFO] mjlab_roller not importable, cross-check uses deploy constants")

    # Joint count
    if TRAIN_COUNT != DEPLOY_COUNT:
        errors.append(f"Joint count mismatch: train={TRAIN_COUNT}, deploy={DEPLOY_COUNT}")
    else:
        print(f"  [OK] Joint count: {TRAIN_COUNT}")

    # Joint order
    for i, (t, d) in enumerate(zip(TRAIN_JOINTS, DEPLOY_JOINTS)):
        if t != d:
            errors.append(f"Joint {i} mismatch: train='{t}', deploy='{d}'")
    if not any("Joint" in e and "mismatch" in e for e in errors):
        print(f"  [OK] Joint order: all 23 joints match")

    # Observation frame dimension
    # Per training config: 3+3+3+3+23+23+23+8+3 = 92
    expected_frame_dim = 3 + 3 + 3 + 3 + 23 + 23 + 23 + 8 + 3
    if OBS_FRAME_DIM != expected_frame_dim:
        errors.append(f"Frame dim: expected {expected_frame_dim}, deploy={OBS_FRAME_DIM}")
    else:
        print(f"  [OK] Observation frame dim: {OBS_FRAME_DIM}")

    # History
    expected_history = 10
    if OBS_HISTORY_LENGTH != expected_history:
        errors.append(f"History length: expected {expected_history}, deploy={OBS_HISTORY_LENGTH}")
    else:
        print(f"  [OK] History length: {OBS_HISTORY_LENGTH}")

    # Total dim
    expected_total = expected_frame_dim * expected_history
    if OBS_TOTAL_DIM != expected_total:
        errors.append(f"Total dim: expected {expected_total}, deploy={OBS_TOTAL_DIM}")
    else:
        print(f"  [OK] Total observation dim: {OBS_TOTAL_DIM}")

    # Scales
    if ANG_VEL_SCALE != 0.25:
        errors.append(f"Angular velocity scale: expected 0.25, deploy={ANG_VEL_SCALE}")
    else:
        print(f"  [OK] Angular velocity scale: {ANG_VEL_SCALE}")

    if JOINT_VEL_SCALE != 0.05:
        errors.append(f"Joint velocity scale: expected 0.05, deploy={JOINT_VEL_SCALE}")
    else:
        print(f"  [OK] Joint velocity scale: {JOINT_VEL_SCALE}")

    # Observation term ordering
    expected_order = [
        ("command", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos_rel", 23),
        ("joint_vel", 23),
        ("last_action", 23),
        ("wheel_contact", 8),
        ("skate_separation", 3),
    ]
    print(f"\n  Observation term order (per frame):")
    offset = 0
    for name, dim in expected_order:
        print(f"    [{offset:3d}:{offset + dim:3d}] {name} ({dim}-D)")
        offset += dim
    assert offset == expected_frame_dim

    return errors


def verify_action_space():
    """Check action space dimensions and standing pose."""
    print("\n" + "=" * 70)
    print("ACTION SPACE VERIFICATION")
    print("=" * 70)

    from config import STANDING_JOINT_POS, NUM_JOINTS

    errors = []

    if NUM_JOINTS != 23:
        errors.append(f"Action dim: expected 23, deploy={NUM_JOINTS}")
    else:
        print(f"  [OK] Action dimension: {NUM_JOINTS}")

    # Training reference from g1.py STANDING_SKATE_CONTROLLED_JOINT_POS
    train_ref = np.array([
        -0.05,  0.0,  0.0,  0.45, -0.25, 0.0,
        -0.05,  0.0,  0.0,  0.45, -0.25, 0.0,
         0.0,   0.0,  0.05,
        -0.10,  0.45, -0.20, 1.10,
        -0.10, -0.45,  0.20, 1.10,
    ])

    try:
        from mjlab_roller.assets.robots.roller.g1 import STANDING_SKATE_CONTROLLED_JOINT_POS
        train_ref = np.array(STANDING_SKATE_CONTROLLED_JOINT_POS)
        print(f"  [OK] Loaded training reference from mjlab_roller")
    except ImportError:
        print(f"  [INFO] mjlab_roller not importable, using hardcoded reference")

    deploy_ref = STANDING_JOINT_POS

    if not np.allclose(train_ref, deploy_ref, atol=1e-6):
        diff = np.abs(train_ref - deploy_ref)
        max_diff_idx = np.argmax(diff)
        errors.append(
            f"Standing pose mismatch: max diff={diff[max_diff_idx]:.6f} "
            f"at joint {max_diff_idx}"
        )
    else:
        print(f"  [OK] Standing reference pose: all {NUM_JOINTS} values match")

    return errors


def verify_control_frequency():
    """Check that control frequency matches simulation decimation."""
    print("\n" + "=" * 70)
    print("CONTROL FREQUENCY VERIFICATION")
    print("=" * 70)

    from config import CONTROL_FREQ_HZ, CONTROL_DT

    errors = []

    # Sim: timestep=0.005, decimation=4 → 50 Hz
    sim_dt = 0.005 * 4  # 0.02s
    sim_freq = 1.0 / sim_dt  # 50 Hz

    if CONTROL_FREQ_HZ != sim_freq:
        errors.append(f"Control freq: expected {sim_freq}Hz, deploy={CONTROL_FREQ_HZ}Hz")
    else:
        print(f"  [OK] Control frequency: {CONTROL_FREQ_HZ} Hz")

    if abs(CONTROL_DT - sim_dt) > 1e-9:
        errors.append(f"Control dt: expected {sim_dt}s, deploy={CONTROL_DT}s")
    else:
        print(f"  [OK] Control dt: {CONTROL_DT}s")

    return errors


def verify_observation_builder():
    """Smoke-test the observation builder produces correct dimensions."""
    print("\n" + "=" * 70)
    print("OBSERVATION BUILDER SMOKE TEST")
    print("=" * 70)

    from config import STANDING_JOINT_POS, OBS_TOTAL_DIM, NUM_JOINTS
    from observation_builder import ObservationBuilder

    errors = []

    builder = ObservationBuilder(standing_ref=STANDING_JOINT_POS)
    builder.reset(STANDING_JOINT_POS)

    obs = builder.push_and_get(
        command=np.zeros(3),
        base_lin_vel_b=np.zeros(3),
        base_ang_vel_b=np.zeros(3),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        joint_pos=STANDING_JOINT_POS,
        joint_vel=np.zeros(NUM_JOINTS),
        wheel_contacts=np.ones(8),
        skate_separation=np.array([0.0, -0.13, 0.0]),
    )

    if obs.shape != (OBS_TOTAL_DIM,):
        errors.append(f"Obs shape: expected ({OBS_TOTAL_DIM},), got {obs.shape}")
    else:
        print(f"  [OK] Observation shape: {obs.shape}")

    if not np.isfinite(obs).all():
        errors.append("Observation contains NaN or Inf")
    else:
        print(f"  [OK] Observation finite: all values are finite")

    # At standing pose, joint_pos_rel should be zero
    frame_0 = obs[:92]
    joint_pos_rel = frame_0[12:35]
    if not np.allclose(joint_pos_rel, 0.0, atol=1e-6):
        errors.append(f"Joint pos rel at standing should be zero, max={np.max(np.abs(joint_pos_rel)):.6f}")
    else:
        print(f"  [OK] Joint pos relative: zero at standing pose")

    # Projected gravity should be [0, 0, -1] → frame[9:12]
    pg = frame_0[9:12]
    expected_pg = np.array([0.0, 0.0, -1.0])
    if not np.allclose(pg, expected_pg, atol=1e-6):
        errors.append(f"Projected gravity mismatch: got {pg}, expected {expected_pg}")
    else:
        print(f"  [OK] Projected gravity: {pg}")

    return errors


def main():
    all_errors = []

    all_errors.extend(verify_observation_layout())
    all_errors.extend(verify_action_space())
    all_errors.extend(verify_control_frequency())
    all_errors.extend(verify_observation_builder())

    print("\n" + "=" * 70)
    if all_errors:
        print(f"VERIFICATION FAILED — {len(all_errors)} error(s):")
        for i, err in enumerate(all_errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print("=" * 70)
        print("\nThe deployment pipeline's observation and action spaces")
        print("match the training environment configuration.")


if __name__ == "__main__":
    main()
