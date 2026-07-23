"""quaternion_imu_features_test — verify the orientation primitives (offline, pure numpy).

  1. impute_unit_quaternion reconstructs a masked component so |q|=1 and matches the known truth.
  2. quat_angular_velocity recovers a KNOWN constant-rate rotation (analytic ω).
  3. quat_angular_distance = |ω|·dt for that stream, and is ROTATION-INVARIANT under a global SO(3) rotation.
  4. rotate_frame leaves angular velocity/distance invariant (the augmentation guarantee) but changes the raw quats.
  5. agent contract returns done with the feature block.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import quaternion_imu_features as Q


def _axis_angle_quat(axis, ang):
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(ang / 2), [np.cos(ang / 2)]])


def _run():
    print("=== QUATERNION-IMU-FEATURES VERIFIER ===")
    checks = {}
    rng = np.random.default_rng(0)

    # constant-rate rotation about a fixed axis: ω magnitude = rate (rad/s)
    n = 200; dt = 0.1; rate = 0.3
    axis = np.array([0.2, -0.5, 0.83]); axis = axis / np.linalg.norm(axis)
    q = np.stack([_axis_angle_quat(axis, rate * dt * k) for k in range(n)])

    # 1. imputation
    qm = q.copy(); qm[5, 3] = np.nan
    qi = Q.impute_unit_quaternion(qm)
    checks["impute_unit_norm"] = abs(np.linalg.norm(qi[5]) - 1.0) < 1e-6
    checks["impute_matches_truth"] = np.min([np.linalg.norm(qi[5] - q[5]), np.linalg.norm(qi[5] + q[5])]) < 1e-3

    # 2. angular velocity magnitude ≈ rate
    av = Q.quat_angular_velocity(q, dt=dt)
    spd = np.linalg.norm(av[:-1], axis=1)
    checks["angvel_magnitude"] = abs(np.median(spd) - rate) < 1e-2
    # direction aligns with axis
    dir_ok = abs(abs(np.dot(av[10] / (np.linalg.norm(av[10]) + 1e-9), axis)) - 1.0) < 1e-2
    checks["angvel_direction"] = dir_ok
    print(f"  -> median |ω|={np.median(spd):.4f} (truth {rate})")

    # 3. angular distance = rate*dt, and invariant to a global rotation
    ad = Q.quat_angular_distance(q)
    checks["angdist_value"] = abs(np.median(ad[:-1]) - rate * dt) < 1e-3
    g = _axis_angle_quat(rng.standard_normal(3), 1.1)               # arbitrary global rotation
    q_rot = Q._normalize(Q.quat_mul(np.broadcast_to(g, q.shape), q))
    ad_rot = Q.quat_angular_distance(q_rot)
    checks["angdist_rotation_invariant"] = float(np.max(np.abs(ad - ad_rot))) < 1e-6
    print(f"  -> median ang_dist={np.median(ad[:-1]):.4f} (truth {rate*dt}); "
          f"max Δ under global rot={np.max(np.abs(ad - ad_rot)):.2e}")

    # 4. rotate_frame augmentation invariance
    accel = rng.standard_normal((n, 3))
    q2, a2 = Q.rotate_frame(q, accel, g)
    av2 = Q.quat_angular_velocity(q2, dt=dt)
    checks["augment_angvel_invariant"] = float(np.max(np.abs(np.linalg.norm(av2[:-1], axis=1) - spd))) < 1e-6
    checks["augment_changes_quats"] = float(np.mean(np.abs(q2 - q))) > 1e-2
    checks["augment_preserves_accel_norm"] = float(np.max(np.abs(np.linalg.norm(a2, axis=1) - np.linalg.norm(accel, axis=1)))) < 1e-6

    # 5. agent contract
    st, d, to, msg = Q.run({"spec": {"n": 100, "seed": 1}}, "t")
    checks["agent_done"] = st == "done" and d["n_features"] == 5 and "ang_dist" in d["feature_names"]
    print(f"  -> agent: {st} | {msg[:90]}")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== quaternion-imu-features: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
