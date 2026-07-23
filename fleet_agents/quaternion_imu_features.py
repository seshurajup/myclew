"""quaternion-imu-features — orientation-aware features + augmentation for quaternion/IMU sensor streams.

Distilled from the CMI-Detect-Behavior 1st-place solution (Devin | Ogurtsov | zyz). The existing
imu-feature-engineer covers ACCELEROMETER kinematics (magnitude / jerk / gravity-removal / spectral).
This pack adds the ORIENTATION half — everything derived from the rotation quaternion — which that
solution leaned on and which is reusable for any wearable / IMU / pose competition. Pure numpy,
quaternion convention [x, y, z, w] (scipy R.from_quat order):

  • impute_unit_quaternion : reconstruct ONE missing quaternion component from the |q|=1 constraint
    (w²+x²+y²+z²=1), choosing the sign for temporal continuity with the previous frame; fall back to
    identity when ≥2 are missing. Physically-grounded imputation instead of mean/ffill.
  • quat_angular_velocity  : ω_t = rotvec(q_t⁻¹ · q_{t+1}) / Δt — the body-frame angular velocity, a far
    stronger motion signal than raw quaternion channels.
  • quat_angular_distance  : geodesic angle of the frame-to-frame relative rotation (rotation-invariant).
  • rotate_frame           : SO(3) data augmentation — left-multiply every quaternion AND rotate every
    accel vector by a shared random rotation. Angular velocity/distance are INVARIANT under it, so it
    teaches the model orientation-invariance without corrupting the derived motion features.

The invariance is the guarantee that makes the augmentation safe and the test exact.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent

_EPS = 1e-9


# ---------------------------------------------------------------- quaternion algebra ([x,y,z,w])
def _normalize(q):
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_inv(q):                       # unit-quaternion inverse = conjugate
    q = np.asarray(q, np.float64)
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def quat_to_rotvec(q):
    """axis*angle (rotation vector) of a unit quaternion; angle in [0, π], canonicalized to w>=0."""
    q = _normalize(q)
    q = np.where(q[..., 3:4] < 0, -q, q)          # canonical hemisphere so angle is the short way
    v = q[..., :3]; w = np.clip(q[..., 3], -1.0, 1.0)
    vn = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(vn, w)
    scale = np.where(vn > _EPS, angle / np.maximum(vn, _EPS), 2.0)  # small-angle: rotvec≈2*v
    return v * scale[..., None]


def rotate_vec(q, v):
    """Rotate 3-vectors v by unit quaternions q: v' = q · (v,0) · q⁻¹."""
    q = _normalize(q)
    vq = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_inv(q))[..., :3]


# ---------------------------------------------------------------- CMI 1st-place primitives
def impute_unit_quaternion(rot):
    """(n,4) quaternions with NaNs → cleaned unit quaternions using the |q|=1 constraint."""
    rot = np.asarray(rot, np.float64).copy()
    out = rot.copy()
    for i in range(len(rot)):
        row = rot[i]; miss = np.isnan(row)
        k = int(miss.sum())
        if k == 0:
            nrm = np.linalg.norm(row)
            out[i] = row / nrm if nrm > 1e-8 else [0, 0, 0, 1.0]
        elif k == 1:
            j = int(np.where(miss)[0][0]); valid = row[~miss]
            ss = float(np.sum(valid ** 2))
            if ss <= 1.0:
                val = np.sqrt(max(0.0, 1.0 - ss))
                if i > 0 and not np.isnan(out[i - 1, j]) and out[i - 1, j] < 0:
                    val = -val                                  # sign for temporal continuity
                out[i, ~miss] = valid; out[i, j] = val
            else:
                out[i] = [0, 0, 0, 1.0]
        else:
            out[i] = [0, 0, 0, 1.0]
    return out


def _valid(q):
    return ~(np.all(np.isnan(q), axis=-1) | np.all(np.isclose(q, 0.0), axis=-1))


def quat_angular_velocity(rot, dt=0.1):
    """ω_t = rotvec(q_t⁻¹·q_{t+1})/dt, shape (n,3); last row 0. Invalid frames → 0."""
    q = np.asarray(rot, np.float64); n = len(q)
    out = np.zeros((n, 3))
    if n < 2:
        return out
    ok = _valid(q)
    rel = quat_mul(quat_inv(q[:-1]), q[1:])
    rv = quat_to_rotvec(rel) / dt
    pair = ok[:-1] & ok[1:]
    out[:-1][pair] = rv[pair]
    return out


def quat_angular_distance(rot):
    """Geodesic angle (rad) of each frame-to-frame relative rotation, shape (n,); last row 0."""
    q = np.asarray(rot, np.float64); n = len(q)
    out = np.zeros(n)
    if n < 2:
        return out
    ok = _valid(q)
    rel = quat_mul(quat_inv(q[:-1]), q[1:])
    ang = np.linalg.norm(quat_to_rotvec(rel), axis=-1)
    pair = ok[:-1] & ok[1:]
    out[:-1][pair] = ang[pair]
    return out


def rotate_frame(rot, accel, rot_quat):
    """SO(3) augmentation: apply a shared world rotation `rot_quat` ([x,y,z,w]) to the whole sequence —
    left-multiply the orientation quaternions and rotate the accel vectors. Returns (rot2, accel2)."""
    rq = _normalize(np.asarray(rot_quat, np.float64))
    rot2 = _normalize(quat_mul(np.broadcast_to(rq, np.asarray(rot).shape), rot))
    accel2 = rotate_vec(np.broadcast_to(rq, accel.shape[:-1] + (4,)), np.asarray(accel, np.float64))
    return rot2, accel2


def orientation_features(rot, accel=None, dt=0.1):
    """Bundle the orientation feature block: angular velocity (3) + angular distance (1) + rot-angle (1)."""
    q = impute_unit_quaternion(rot)
    av = quat_angular_velocity(q, dt=dt)
    ad = quat_angular_distance(q)
    ang = 2.0 * np.arccos(np.clip(np.abs(_normalize(q)[..., 3]), -1.0, 1.0))   # absolute tilt angle
    X = np.column_stack([av, ad, ang])
    names = ["ang_vel_x", "ang_vel_y", "ang_vel_z", "ang_dist", "orient_angle"]
    return np.nan_to_num(X).astype(np.float32), names


# ---------------------------------------------------------------- agent
class QuaternionIMUFeatures(BaseAgent):
    name = "quaternion-imu-features"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rot = s.get("rot"); accel = s.get("accel"); dt = float(s.get("dt", 0.1))
        if rot is None:                                   # self-demo on a synthetic smooth rotation stream
            rng = np.random.default_rng(int(s.get("seed", 0))); n = int(s.get("n", 128))
            axis = _normalize(rng.standard_normal(3)); t = np.linspace(0, 4 * np.pi, n)
            ang = t[:, None] * 0.3
            qv = np.concatenate([axis[None] * np.sin(ang / 2), np.cos(ang / 2)], axis=1)
            rot = qv; accel = rng.standard_normal((n, 3))
        rot = np.asarray(rot, np.float64); accel = np.asarray(accel, np.float64)
        X, names = orientation_features(rot, accel, dt=dt)
        msg = (f"quaternion-imu-features: built {X.shape[1]} orientation features {names} over {len(rot)} "
               f"frames (ang-vel/ang-dist from q_t⁻¹·q_t+1, |q|=1 imputation, SO(3) rot-augment available)")
        self.log(msg, kind="finding",
                 recommendation="add these to imu-feature-engineer's kinematics for wearable/IMU comps")
        return self.done({"n_features": X.shape[1], "feature_names": names,
                          "mean_angular_speed": float(np.mean(np.linalg.norm(X[:, :3], axis=1)))}, msg)


_AGENT = QuaternionIMUFeatures()


def run(q, worker):
    return _AGENT.run(q, worker)
