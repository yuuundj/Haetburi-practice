"""Feedback balance for the ankles, added on top of the open-loop gait targets.

Uses only what the real robot can measure: the torso IMU (pitch / roll angle and
rate) and the ankle servos' load (STS / Dynamixel servos report it).

  pitch, slow:  the ankle pitch servos' load says how far the centre of mass is
                ahead of (or behind) the pitch axis.  Integrating it shifts both
                ankle pitch targets until the load is gone: the body leans until
                its CoM sits over the pitch axis, and keeps re-finding it as the
                gait shifts weight.  Gain ki: rad/s per unit of load (load / stall).
  pitch, fast:  damps fore-aft rocking with the torso pitch rate (gain kd, s).
  roll:         damps side-to-side rocking with the torso roll rate (gain kdr, s),
                on top of the gait's own rocking.

Signs (checked on v15): a positive ankle pitch leans the body back; a positive
ankle roll tips the torso to negative roll.

Without ankle pitch joints (v17: hip pitch instead) the pitch part works the
hip pitch joints: their load is the upper body's CoM ahead of (or behind) the
hip axis, and the same integral leans the upper body until it is gone.  With
the feet flat, a positive hip pitch leans the torso back, like a positive ankle
pitch, so the same signs apply (checked on v17: the hip pitch load is positive
with the CoM ahead of the axis).

A gait turns it on with  g["fb"] = dict(ki=..., kd=..., kdr=..., max_deg=...).
"""
import mujoco
import numpy as np


class Balance:
    def __init__(self, m, fb):
        self.ki, self.kd, self.kdr = fb.get("ki", 0.0), fb.get("kd", 0.0), fb.get("kdr", 0.0)
        self.max = np.deg2rad(fb.get("max_deg", 12.0))
        self.tc = fb.get("rate_tc", 0.02)                      # rate low-pass, s (IMU filtering)
        act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.pitch_ids = [i for i in (act("ankle_pitch_l"), act("ankle_pitch_r")) if i >= 0]
        self.pitch_joint = "ankle_pitch"
        if not self.pitch_ids:                                 # v17: lean with the hips instead
            self.pitch_ids = [i for i in (act("hip_pitch_l"), act("hip_pitch_r")) if i >= 0]
            self.pitch_joint = "hip_pitch"
        self.stall = float(m.actuator_forcerange[:, 1].max())
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.bias, self.prev, self.rate = 0.0, None, np.zeros(2)

    @staticmethod
    def angles(d, body):
        R = d.xmat[body].reshape(3, 3)
        return np.array([np.arctan2(R[0, 2], R[2, 2]),          # pitch: + leaning forward
                         np.arctan2(R[2, 1], R[2, 2])])         # roll

    def update(self, d, out, dt):
        a = self.angles(d, self.torso)
        if self.prev is not None:
            raw = (a - self.prev) / dt
            self.rate += (raw - self.rate) * min(1.0, dt / self.tc)
        self.prev = a
        if self.pitch_ids:
            load = float(np.mean(d.actuator_force[self.pitch_ids])) / self.stall
            self.bias = float(np.clip(self.bias + self.ki * load * dt, -self.max, self.max))
            p = float(np.clip(self.bias + self.kd * self.rate[0], -self.max, self.max))
            for s in ("l", "r"):
                out[f"{self.pitch_joint}_{s}"] = out.get(f"{self.pitch_joint}_{s}", 0.0) + p
        if self.kdr:
            r = self.kdr * self.rate[1]
            for s in ("l", "r"):
                out[f"ankle_roll_{s}"] = out.get(f"ankle_roll_{s}", 0.0) + r
        return out


def make(m, g):
    return Balance(m, g["fb"]) if g.get("fb") else None
