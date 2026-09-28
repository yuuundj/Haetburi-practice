"""
Haetburi biped - design parameters for every model variant.

Pick a variant with the environment variable HAETBURI_VARIANT (default: the
latest).  Each variant builds into robot/<variant>/ and never touches another.

Frames
------
Robot frame   :  +x forward (face direction), +y robot-left, +z up, z = 0 at the sole.
Source STL    :  -y_stl is the face direction, +z_stl up, origin at the sole.
Mapping       :  (x_w, y_w, z_w) = ( -y_stl, x_stl, z_stl ) * s + (0, 0, lift)

Lengths are millimetres unless a name ends in _M.
"""
import os as _os

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SRC_STL = _os.path.join(ROOT, "modelings", "haetburi_grip_20mm_300mm.stl")
SRC_HEIGHT = 300.0

# =============================================================== variants
DEFAULT_VARIANT = "v6_split_round"
VARIANTS = {
    "v1_700_exposed": dict(
        title="700 mm, ankle motors stacked under the legs (exposed)",
        LAYOUT="exposed", TOTAL_HEIGHT=700.0),
    "v2_500_exposed": dict(
        title="500 mm, ankle motors stacked under the legs (exposed)",
        LAYOUT="exposed", TOTAL_HEIGHT=500.0),
    "v4_ankle_in_foot": dict(
        title="both ankle motors inside a tall foot, all 7 motors hidden",
        LAYOUT="foot_both"),
    "v5_split_boxy": dict(
        title="roll motor in the leg, pitch motor in a boxy foot",
        LAYOUT="split", SPLIT_STYLE="boxy"),
    "v6_split_round": dict(
        title="roll motor in the leg, pitch motor in a round forward foot, shorts",
        LAYOUT="split", SPLIT_STYLE="round"),
    "v7_roll_only": dict(
        title="no ankle pitch: hip pitch + ankle roll, roll motor low in the foot",
        LAYOUT="roll_foot"),
    "v8_yaw_roll": dict(
        title="Otto layout: hip yaw + ankle roll, lower foot",
        LAYOUT="roll_foot", HIP_JOINT="yaw"),
}
VARIANT = _os.environ.get("HAETBURI_VARIANT", DEFAULT_VARIANT)
if VARIANT not in VARIANTS:
    raise KeyError(f"unknown HAETBURI_VARIANT {VARIANT!r}; choose from {list(VARIANTS)}")
OUT_DIR = _os.path.join(ROOT, "robot", VARIANT)
MESH_DIR = _os.path.join(OUT_DIR, "meshes")

# ============================================================ shared values
# Dynamixel XL430-W250-T.  ROBOTIS drawing (same case as XM430):
# 28.5 (W) x 46.5 (H) x 34 (D) mm, shaft along D, on the W centre line,
# 11.25 mm from one end of H.
MOTOR_W, MOTOR_H, MOTOR_D = 28.5, 46.5, 34.0
MOTOR_AXIS_FROM_END = 11.25
MOTOR_HORN = 2.5
MOTOR_MASS_KG = 0.0572
MOTOR_STALL_NM = 1.4        # at 11.1 V
MOTOR_NOLOAD_RADS = 5.97    # 57 rpm at 11.1 V
MOTOR_GEAR = 258.5
MOTOR_ROTOR_INERTIA = 1.0e-7

CUT_LEG_TOP = 20.5          # native z: legs merge into the belly just above this
CUT_WAIST = 50.0            # native z: pelvis / upper-body split
LEG_X_LIMIT = 25.0          # |x_stl| beyond this at leg height is the tail
NECK_NATIVE = 88.0
ARM_X_NATIVE = 36.0
HEAD_BANDS = 6

WALL_MM = 3.0               # printed wall used for shell mass
RIB_FACTOR = 1.25
RHO_PLA = 1.24e-3
BRACKET_MASS_KG = 0.025
ELECTRONICS_KG = 0.30       # battery + U2D2 + wiring
HEAD_EXTRA_KG = 0.05

JOINT_DAMPING = 0.10
JOINT_FRICTION = 0.04
ACT_KP = 30.0
ACT_KV = 1.2

# packing defaults for the in-shell layouts
FLOOR = 3.0                 # sole plate under the lowest motor
WALL = 2.5                  # printed wall around a motor
CLEAR = 1.5                 # minimum motor-to-motor gap through any sweep
FOOT_COST = {"length": 1.0, "width": 1.3, "height": 0.25}   # height cheapest by request
FIT_LEGACY = False
FOOT_PLAN_EXPONENT = FOOT_SIDE_EXPONENT = 4.0
FOOT_SOLE_CUT = 0.30
FOOT_HEEL_REACH = FOOT_TOE_REACH = None
LEG_FOOT_CLEAR = 2.0

# demo / preview motion amplitudes (degrees)
DEMO = dict(pitch_lean=7, roll_lean=8, yaw=35, bow_hip=16, bow_ankle=-13,
            diag_roll=7, diag_yaw=-30,
            pose_bow_hip=18, pose_bow_ankle=-12,     # held 3 s for the preview images
            pose_lift_roll=6.5, pose_lift_hip=15)

globals().update({k: v for k, v in VARIANTS[VARIANT].items() if k != "title"})
TITLE = VARIANTS[VARIANT]["title"]

# ======================================================== layout: exposed
if LAYOUT == "exposed":
    # Motor boxes are centred on their shafts (the simplification used at the time).
    LEG_HALF_SPACING_NATIVE = 14.35
    MOTOR_GAP_MIN = 4.0
    FOOT_GAP_MIN = 8.0
    FOOT_PAD_T, FOOT_PAD_X, FOOT_PAD_Y = 12.0, 110.0, 66.0
    LIM = {"waist_yaw": (-60.0, 60.0), "hip_pitch": (-45.0, 75.0),
           "ankle_pitch": (-45.0, 45.0), "ankle_roll": (-30.0, 30.0)}

    def _ankle_layout(total_h):
        """Roll motor flat (46.5 wide) if the ankles have room, else on end."""
        for upright in (False, True):
            roll_v = MOTOR_H if upright else MOTOR_D
            roll_y = MOTOR_D if upright else MOTOR_H
            stack = FOOT_PAD_T + roll_v + MOTOR_D
            y_hip = LEG_HALF_SPACING_NATIVE * (total_h - stack) / SRC_HEIGHT
            if 2 * (y_hip - roll_y / 2) >= MOTOR_GAP_MIN:
                return upright, roll_v, roll_y, stack, y_hip
        raise ValueError(f"no ankle layout fits a {total_h:.0f} mm robot")

    ROLL_UPRIGHT, ROLL_V, ROLL_Y, ANKLE_STACK, _Y_HIP = _ankle_layout(TOTAL_HEIGHT)
    Z_ANKLE_ROLL = FOOT_PAD_T + ROLL_V / 2.0
    Z_ANKLE_PITCH = FOOT_PAD_T + ROLL_V + MOTOR_D / 2.0
    FOOT_PAD_Y_OFFSET = max(8.0, FOOT_GAP_MIN / 2 + FOOT_PAD_Y / 2 - _Y_HIP)
    MESH_SCALE = (TOTAL_HEIGHT - ANKLE_STACK) / SRC_HEIGHT
    CHAR_HEIGHT = TOTAL_HEIGHT - ANKLE_STACK

# ====================================================== layout: foot_both
elif LAYOUT == "foot_both":
    # Both ankle motors in the foot: pitch fixed in the foot (standing, long end
    # down), roll riding on the pitch output (standing, long end up), 30 mm apart.
    CHAR_HEIGHT = 500.0
    MESH_SCALE = CHAR_HEIGHT / SRC_HEIGHT
    LEG_HALF_SPACING_NATIVE = 14.35
    LEG_STRAIGHT_Z = 10.0
    LEG_BOTTOM_NATIVE = 9.0
    ANKLE_LOWER, PITCH_H_DIR, ROLL_H_DIR = "pitch", "-z", "+z"
    PITCH_SHIFT = ROLL_SHIFT = 0.0
    ROLL_ABOVE_PITCH = 30.0
    LEG_ABOVE_ROLL_CASE = 2.0
    FOOT_PLAN_EXPONENT = FOOT_SIDE_EXPONENT = 4.0
    FOOT_SOLE_CUT = 0.30
    FIT_LEGACY = True      # reproduces the fitter as it was (it stopped at the first
                           # feasible height, see README); gives 103 x 43 x 120 mm
    ELECTRONICS_AT = "backpack"
    LIM = {"waist_yaw": (-60.0, 60.0), "hip_pitch": (-45.0, 75.0),
           "ankle_pitch": (-15.0, 15.0), "ankle_roll": (-12.0, 12.0)}

# ========================================================== layout: split
elif LAYOUT == "split":
    # One ankle motor in the foot (pitch, lying), one in the leg (roll, standing).
    CHAR_HEIGHT = 500.0
    MESH_SCALE = CHAR_HEIGHT / SRC_HEIGHT
    LEG_HALF_SPACING_NATIVE = 14.75   # leg centres at LEG_STRAIGHT_Z
    LEG_STRAIGHT_Z = 16.0
    LEG_EXPONENT = 6.0
    LIM = {"waist_yaw": (-60.0, 60.0), "hip_pitch": (-45.0, 45.0),
           "ankle_pitch": (-15.0, 15.0), "ankle_roll": (-12.0, 12.0)}
    if SPLIT_STYLE == "boxy":
        PITCH_H_DIR = "-x"               # long end to the heel: CoM sat 11 mm behind the legs
        FOOT_PLAN_EXPONENT = FOOT_SIDE_EXPONENT = 6.0
        FOOT_SOLE_CUT = 0.30
        ELECTRONICS_AT = "backpack"
        TRUNK_BLEND_TOP = None           # legs keep the original flare into the belly
        DEMO = dict(DEMO, pitch_lean=4, bow_hip=12, bow_ankle=-9, pose_bow_hip=12, pose_bow_ankle=-9)
    else:
        PITCH_H_DIR = "+x"               # long end to the toe: a forward-pointing foot
        FOOT_PLAN_EXPONENT = 2.5         # oval from above
        FOOT_SIDE_EXPONENT = 4.0         # domed
        FOOT_HEEL_REACH, FOOT_TOE_REACH = 30.0, 55.0
        FOOT_SOLE_CUT = 0.0              # sole on the widest line: no rocker
        ELECTRONICS_AT = "belly"
        ELECTRONICS_POS = (26.0, 0.0, 12.0)   # pelvis frame, from the hip axis
        TRUNK_BLEND_TOP = 26.0           # "shorts" rebuilt below this native height
        TAIL_CUT_X = 28.0
        DEMO = dict(DEMO, pitch_lean=5, bow_hip=12, bow_ankle=-9, pose_bow_hip=12, pose_bow_ankle=-9)
# ======================================================= layout: roll_foot
elif LAYOUT == "roll_foot":
    HIP_JOINT = globals().get("HIP_JOINT", "pitch")
    # Ankle pitch removed (5 motors).  The roll motor stands in the foot with its
    # long end up, so the roll axis sits low (~14 mm above the sole): with a low
    # axis, rolling tips the body onto one foot instead of rolling the feet onto
    # their edges (v6).  With no motor in the leg, the leg is the original one.
    CHAR_HEIGHT = 500.0
    MESH_SCALE = CHAR_HEIGHT / SRC_HEIGHT
    LEG_HALF_SPACING_NATIVE = 14.35
    LEG_STRAIGHT_Z = 10.0            # original leg, straight below this
    LEG_BOTTOM_NATIVE = 9.0          # same leg as the reviewed Doraemon-foot concept
    ROLL_H_DIR = "+z"                # long end up: shaft near the sole
    ROLL_SHIFT = 0.0
    LEG_ABOVE_ROLL_CASE = 2.0
    FOOT_PLAN_EXPONENT = 2.5
    FOOT_SIDE_EXPONENT = 4.0
    FOOT_HEEL_REACH, FOOT_TOE_REACH = 30.0, 55.0
    FOOT_SOLE_CUT = 0.0
    ELECTRONICS_AT = "belly"
    ELECTRONICS_POS = (26.0, 0.0, 12.0)
    LIM = {"waist_yaw": (-60.0, 60.0), "hip_pitch": (-45.0, 45.0), "hip_yaw": (-30.0, 30.0),
           "ankle_pitch": (0.0, 0.0), "ankle_roll": (-15.0, 15.0)}
    DEMO = dict(DEMO, pitch_lean=0, roll_lean=5, diag_roll=5, bow_hip=12, bow_ankle=0,  # tips at ~8 deg of roll
                pose_bow_hip=6, pose_bow_ankle=0, pose_lift_roll=5, pose_lift_hip=4)
    if HIP_JOINT == "yaw":
        # v8: squeeze the empty space out of the foot.  Same standing roll motor,
        # but the foot is fitted with height weighted most, a flatter top, and the
        # motor slid 12 mm toward the toe where the bun is tallest.
        # (v7: 95.6 x 43 x 93.5 mm; this: 85 x 43 x 57.5 mm)
        ROLL_SHIFT = 12.0
        FOOT_SIDE_EXPONENT = 6.0
        FOOT_COST = {"length": 1.0, "width": 1.3, "height": 2.0}
        DEMO = dict(DEMO, bow_hip=0, pose_bow_hip=0, pose_lift_hip=8)
        HIP_YAW_LIFT = 15.5         # hip yaw case bottom above the crotch (see layout.hip_yaw_case)
        ELECTRONICS_POS = (26.0, 0.0, 12.0 + 12.25)   # same belly spot as v7, pelvis frame at the crotch
else:
    raise ValueError(LAYOUT)
