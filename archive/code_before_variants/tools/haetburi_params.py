"""
Haetburi biped - design parameters (single source of truth).

Frames
------
Robot frame   :  +x forward (face direction), +y robot-left, +z up, z = 0 at the sole.
Source STL    :  -y_stl is the face direction, +z_stl up, origin at the sole.
Mapping       :  (x_w, y_w, z_w) = ( -y_stl, x_stl, z_stl ) * s + (0, 0, LIFT)

Lengths are millimetres unless a name ends in _M.
"""
import os as _os

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SRC_STL = _os.path.join(ROOT, "modelings", "haetburi_grip_20mm_300mm.stl")
OUT_DIR = _os.path.join(ROOT, "robot")
MESH_DIR = _os.path.join(OUT_DIR, "meshes")
SRC_HEIGHT = 300.0

# ------------------------------------------------------------------- scaling
# The character (head to the original sole) is built at this size.  The feet
# grow downward to hold the ankle motors, so the robot ends up taller than this.
CHAR_HEIGHT = float(_os.environ.get("HAETBURI_HEIGHT", 500.0))
MESH_SCALE = CHAR_HEIGHT / SRC_HEIGHT

# ------------------------------------------------------- Dynamixel XL430-W250-T
# ROBOTIS drawing (same case as XM430): 28.5 (W) x 46.5 (H) x 34 (D) mm.
# The output shaft runs along D, on the W centre line, 11.25 mm from one end of H.
MOTOR_W, MOTOR_H, MOTOR_D = 28.5, 46.5, 34.0
MOTOR_AXIS_FROM_END = 11.25
MOTOR_HORN = 2.5            # horn + screw heads beyond the D face
MOTOR_MASS_KG = 0.0572
MOTOR_STALL_NM = 1.4        # at 11.1 V
MOTOR_NOLOAD_RADS = 5.97    # 57 rpm at 11.1 V
MOTOR_GEAR = 258.5
MOTOR_ROTOR_INERTIA = 1.0e-7

# ------------------------------------------------------------ source landmarks
# Native (300 mm) heights in the source STL.
LEG_STRAIGHT_Z = 16.0       # below this the leg is a straight rounded-square prism
CUT_LEG_TOP = 20.5          # legs merge into the belly just above this
CUT_WAIST = 50.0            # pelvis / upper-body split
LEG_X_LIMIT = 25.0          # |x_stl| beyond this at leg height is the tail
LEG_HALF_SPACING_NATIVE = 14.75   # leg centres at LEG_STRAIGHT_Z
NECK_NATIVE = 88.0
ARM_X_NATIVE = 36.0
HEAD_BANDS = 6

# ------------------------------------------------------------------ ankle
# One ankle motor in the foot, one in the leg (see README for the search that
# ruled out putting both in the foot).
#   ankle pitch : motor fixed in the foot, lying on the sole, long end to the toe
#   ankle roll  : motor fixed in the leg, standing, long end up, so its shaft
#                 (and the roll axis) sits near the bottom of the leg
#   the ankle bracket between the two horns carries no motor
PITCH_H_DIR = "+x"
FLOOR = 3.0                 # sole plate under the pitch motor
WALL = 2.5                  # printed wall thickness
CLEAR = 1.5                 # minimum motor-to-motor gap through any joint sweep

# ------------------------------------------------------------------ shapes
# Superellipsoid feet: (|x/a|^p + |y/b|^p)^(q/p) + |z/c|^q <= 1.
#   p = roundness seen from above (2 = ellipse, higher = squarer)
#   q = roundness seen from the side
FOOT_PLAN_EXPONENT = 2.5    # oval from above, like Doraemon's feet
FOOT_SIDE_EXPONENT = 4.0    # domed, fairly upright sides so the motor fits low
FOOT_HEEL_REACH = 30.0      # the sole reaches at least this far behind the leg ...
FOOT_TOE_REACH = 55.0       # ... and this far in front of it: a forward-pointing foot
FOOT_SOLE_CUT = 0.0         # sole on the bun's widest line: a dome, not a rocker.
                            # Cutting lower (0.3) left the sides bulging past the
                            # sole, and ankle roll just rolled the feet onto them.
LEG_EXPONENT = 6.0          # straight leg section: rounded square that wraps the roll motor
LEG_FOOT_CLEAR = 2.0        # closest the swinging leg may come to its foot
# The lower belly is rebuilt as a smooth "shorts" shape from the leg tops up to
# this native height, so the square legs run into the body without a ledge.
TRUNK_BLEND_TOP = 26.0
TAIL_CUT_X = 28.0           # below the blend height, x_stl beyond this is the tail root
# cost of foot size when fitting the bun: height is deliberately the cheapest
FOOT_COST = {"length": 1.0, "width": 1.3, "height": 0.25}

# -------------------------------------------------------------- structure
WALL_MM = 3.0
RIB_FACTOR = 1.25
RHO_PLA = 1.24e-3
BRACKET_MASS_KG = 0.025
ELECTRONICS_KG = 0.30       # battery + U2D2 + wiring
# Carried low in the belly, in front of the hip motors (pelvis frame, mm from the
# hip axis).  In the backpack it put the centre of mass 11 mm behind the legs,
# which forced the feet to reach backward.
ELECTRONICS_POS = (26.0, 0.0, 12.0)
HEAD_EXTRA_KG = 0.05

# ---------------------------------------------------------------- limits
LIM = {                     # degrees
    "waist_yaw":   (-60.0, 60.0),
    "hip_pitch":   (-45.0, 45.0),   # the leg motor clears the hip motor over this range
    "ankle_pitch": (-15.0, 15.0),   # motor sweep inside the foot was sized for this
    "ankle_roll":  (-12.0, 12.0),
}

# ------------------------------------------------------------- simulation
JOINT_DAMPING = 0.10
JOINT_FRICTION = 0.04
ACT_KP = 30.0
ACT_KV = 1.2
