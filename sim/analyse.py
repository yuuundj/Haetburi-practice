#!/usr/bin/env python3
"""Static check of the Haetburi model: does it stand, and how hard do the
XL430-W250-T servos have to work?   Headless, no graphics required.

    python3 sim/analyse.py [--variant v6_split_round]
"""
import os

import mujoco
import numpy as np

import variants

SCENE = None                      # set from --variant in main()
STALL = 1.4                     # XL430-W250-T stall torque at 11.1 V, N.m


def act(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def ramp_hold(m, d, targets, ramp=1.5, hold=2.0):
    """Drive the named actuators to their targets over `ramp` s, then hold."""
    ids = {act(m, k): v for k, v in targets.items()}
    n_ramp = int(ramp / m.opt.timestep)
    for i in range(n_ramp):
        a = (i + 1) / n_ramp
        for j, v in ids.items():
            d.ctrl[j] = a * v
        mujoco.mj_step(m, d)
    for _ in range(int(hold / m.opt.timestep)):
        mujoco.mj_step(m, d)


def torques(m, d):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): d.actuator_force[i]
            for i in range(m.nu)}


def foot_world_verts(m, d, foot="foot_l"):
    """Foot contact-shape vertices in the world frame: a mesh (stored by MuJoCo
    in its own principal-axis frame) or a box pad."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_c0")
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        s = np.array(np.meshgrid(*[(-1, 1)] * 3)).reshape(3, -1).T * m.geom_size[g]
        return s @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def sole_size(m, d, foot="foot_l"):
    """(sole length, sole width, overall length, overall width) in mm, neutral pose."""
    v = foot_world_verts(m, d, foot)
    sole = v[v[:, 2] < v[:, 2].min() + 1e-4]
    return (np.ptp(sole[:, 0]) * 1000, np.ptp(sole[:, 1]) * 1000,
            np.ptp(v[:, 0]) * 1000, np.ptp(v[:, 1]) * 1000)


def sweep(m, d, joints, label, com0, degs):
    print(f"\n  {label}: both joints commanded together, 1.5 s ramp then hold")
    print(f"    {'cmd deg':>8s} {'N.m worst':>10s} {'% stall':>8s} "
          f"{'CoM shift mm':>13s}  state")
    for deg in degs:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ramp_hold(m, d, {j: np.deg2rad(deg) for j in joints})
        t = torques(m, d)
        tq = max(abs(t[j]) for j in joints)
        com = d.subtree_com[1]
        up = com[2] > 0.75 * com0[2]
        shift = np.linalg.norm(com[:2] - com0[:2]) * 1000
        print(f"    {deg:8.0f} {tq:10.3f} {tq/STALL*100:7.1f}% {shift:13.1f}  "
              f"{'standing' if up else 'TIPPED OVER'}")


def motor_clearance(m, d):
    """Smallest gap between any two XL430 boxes in the neutral pose (mm).
    Negative means two motors occupy the same space and cannot be built."""
    mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "motor")
    boxes = [i for i in range(m.ngeom) if m.geom_matid[i] == mat]
    worst = (np.inf, None)
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            i, j = boxes[a], boxes[b]
            gap = np.abs(d.geom_xpos[i] - d.geom_xpos[j]) - m.geom_size[i] - m.geom_size[j]
            sep = gap.max()                        # separated along the best axis
            if sep < worst[0]:
                worst = (sep, (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]),
                               mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[j])))
    return len(boxes), worst[0] * 1000, worst[1]


def foot_load(m, d, foot):
    """Normal force (N) the floor pushes on one foot pad."""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_c0")
    f6, total = np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if gid in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            total += f6[0]
    return total


def single_support(m, d, com0):
    """Kneeless shuffle step, standing on the left foot.

    With no hip roll the pelvis and both legs are one rigid piece sideways, so
    a few degrees of ankle roll tips the whole robot onto one foot and lifts the
    other. The lifted leg is then swung with its hip while its ankle pitch keeps
    the pad parallel to the floor. The stance ankle carries the entire robot;
    that is the load that decides whether the robot can take steps."""
    weight = m.body_mass.sum() * 9.81
    print("\n  single support: lean onto the left foot, swing the right leg")
    # Which roll direction tips the body onto the left foot depends on the leg
    # geometry, so try both and keep the one that actually lifts the right foot.
    lift, sgn = None, 1.0
    rmax = np.degrees(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_roll_l")][1])
    for sg_ in (1.0, -1.0):
        for deg in np.arange(0.5, rmax + 0.01, 0.5):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            ramp_hold(m, d, {"ankle_roll_l": np.deg2rad(sg_ * deg),
                             "ankle_roll_r": np.deg2rad(sg_ * deg)}, ramp=1.5, hold=1.0)
            if foot_load(m, d, "foot_r") < 0.01 * weight and d.subtree_com[1][2] > 0.75 * com0[2]:
                lift, sgn = deg, sg_
                break
        if lift is not None:
            break
    if lift is None:
        print("    right foot never unloaded: with no hip roll the body cannot shift sideways")
        print("    on flat feet, and here ankle roll tips both feet onto their edges instead")
        print("    of tilting the body onto one foot (roll axis high above the sole)")
        return
    roll = min(lift + 3.0, rmax)                       # a little past lift-off
    tgt = {"ankle_roll_l": np.deg2rad(sgn * roll), "ankle_roll_r": np.deg2rad(sgn * roll)}
    mujoco.mj_resetDataKeyframe(m, d, 0)
    ramp_hold(m, d, tgt, ramp=1.5, hold=1.0)
    t = torques(m, d)
    clear = foot_world_verts(m, d, "foot_r")[:, 2].min() * 1000
    print(f"    right foot lifts at {sgn*lift:+.1f} deg; holding {sgn*roll:+.1f} deg the right pad "
          f"clears the floor by ~{clear:.0f} mm")
    print(f"    left foot carries {foot_load(m, d, 'foot_l')/weight*100:.0f} % of the weight; "
          f"left ankle roll {abs(t['ankle_roll_l']):.2f} N.m "
          f"({abs(t['ankle_roll_l'])/STALL*100:.0f} %), pitch {abs(t['ankle_pitch_l']):.2f} N.m "
          f"({abs(t['ankle_pitch_l'])/STALL*100:.0f} %)")

    print(f"    {'R hip':>7s} {'swing':>6s}  peak %:  {'L roll':>7s} {'L pitch':>8s} "
          f"{'L hip':>6s} {'R hip':>6s} {'waist':>6s}  state")
    for hip, dur in ((-20, 1.0), (20, 1.0), (-20, 0.3), (20, 0.3)):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ramp_hold(m, d, tgt, ramp=1.5, hold=0.5)
        swing = dict(tgt, hip_pitch_r=np.deg2rad(hip), ankle_pitch_r=np.deg2rad(-hip))
        ids = {act(m, k): v for k, v in swing.items()}
        moving = {act(m, "hip_pitch_r"), act(m, "ankle_pitch_r")}
        n = int(dur / m.opt.timestep)                  # swing, then hold 1 s
        peak = {}
        for i in range(n + int(1.0 / m.opt.timestep)):
            a = min(1.0, (i + 1) / n)
            for j, v in ids.items():
                d.ctrl[j] = a * v if j in moving else v
            mujoco.mj_step(m, d)
            for k, v in torques(m, d).items():
                peak[k] = max(peak.get(k, 0.0), abs(v))
        up = d.subtree_com[1][2] > 0.75 * com0[2] and foot_load(m, d, "foot_l") > 0.5 * weight
        pc = lambda k: f"{peak[k]/STALL*100:5.0f}"
        print(f"    {hip:+5d} deg {dur:4.1f} s         {pc('ankle_roll_l'):>7s} {pc('ankle_pitch_l'):>8s} "
              f"{pc('hip_pitch_l'):>6s} {pc('hip_pitch_r'):>6s} {pc('waist_yaw'):>6s}  "
              f"{'standing' if up else 'LOST BALANCE'}")


def main():
    global SCENE
    a = variants.parser(__doc__).parse_args()
    SCENE = variants.scene(a.variant)
    print(f"variant {a.variant}")
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    com0 = d.subtree_com[1].copy()

    print(f"model   {m.nu} actuated DoF, {m.nbody-1} links, {m.body_mass.sum():.3f} kg")
    print(f"CoM     x {com0[0]*1000:+.1f}  y {com0[1]*1000:+.1f}  z {com0[2]*1000:.1f}  mm")
    for b in ("pelvis", "torso", "leg_l", "ankle_l", "foot_l"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"        {b:8s} {m.body_mass[i]:6.3f} kg")

    n, gap, pair = motor_clearance(m, d)
    print(f"\n{n} XL430 boxes, tightest pair {pair[0]} / {pair[1]}: "
          f"{gap:+.1f} mm {'(OK)' if gap >= 0 else '(COLLIDE - not buildable)'}")

    ramp_hold(m, d, {}, ramp=0.001, hold=3.0)
    print(f"\nneutral pose held for 3 s: CoM sank {(com0[2]-d.subtree_com[1][2])*1000:.1f} mm, "
          f"residual speed {np.linalg.norm(d.cvel[1][3:])*1000:.1f} mm/s")
    print("  holding torque")
    for k, v in torques(m, d).items():
        print(f"    {k:15s} {v:+7.3f} N.m   {abs(v)/STALL*100:5.1f} % of stall")

    hi_p = np.degrees(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_pitch_l")][1])
    hi_r = np.degrees(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_roll_l")][1])
    sweep(m, d, ("ankle_pitch_l", "ankle_pitch_r"), "ankle pitch (lean fore/aft)", com0,
          np.arange(0, hi_p + 0.1, 1.0))
    sweep(m, d, ("ankle_roll_l", "ankle_roll_r"), "ankle roll (lean left/right)", com0,
          np.arange(0, hi_r + 0.1, 1.0))

    single_support(m, d, com0)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    fx, fy, fx_all, fy_all = sole_size(m, d)
    ank = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_l")]
    sole = foot_world_verts(m, d, "foot_l")
    sole = sole[sole[:, 2] < sole[:, 2].min() + 1e-4]
    heel = (com0[0] - sole[:, 0].min()) * 1000                  # CoM ahead of the heel edge
    toe = (sole[:, 0].max() - com0[0]) * 1000                   # CoM behind the toe edge
    margin = min(heel, toe)
    lever = (com0[2] - ank[2]) * 1000
    print(f"\nfoot {fx_all:.0f} x {fy_all:.0f} mm per foot, flat sole {fx:.0f} x {fy:.0f} mm; "
          f"ankle pitch axis {ank[2]*1000:.0f} mm above the sole")
    print(f"CoM is {lever:.0f} mm above the ankle axis; sole reaches {heel:.0f} mm behind and "
          f"{toe:.0f} mm ahead of it")
    tip = m.body_mass.sum() * 9.81 * margin / 1000
    print(f"static tip-over toward the {'heel' if heel < toe else 'toe'} at "
          f"{np.degrees(np.arcsin(min(1, margin/lever))):.1f} deg of ankle pitch")
    print(f"  holding that lean on two feet : {tip/2:.2f} N.m per ankle "
          f"({tip/2/STALL*100:.0f} % of stall)")
    print(f"  holding that lean on one foot : {tip:.2f} N.m "
          f"({tip/STALL*100:.0f} % of stall){'  <- exceeds the XL430' if tip > STALL else ''}")


if __name__ == "__main__":
    main()
