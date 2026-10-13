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
    "v9_yaw_roll_display": dict(
        title="v8 holding a 9 inch display in both hands",
        LAYOUT="roll_foot", HIP_JOINT="yaw", DISPLAY=True),
    "v10_split_display": dict(
        title="3 joints per leg (v5/v6 layout) with the display, round feet, <= 600 mm",
        LAYOUT="split", SPLIT_STYLE="round", DISPLAY=True,
        # applied after the layout block: the foot gets whatever height the
        # 600 mm budget leaves (see README, v10)
        after=dict(
            # Foot: round from above (3) and domed (5), height capped at 55 mm.  The
            # 600 mm budget would allow ~66 mm, but the roll axis sits in the leg just
            # above the foot, and above ~80 mm the feet roll onto their edges instead
            # of the body tipping onto one foot (v6).  55 mm keeps it at 77 mm.
            FOOT_MAX_H=55.0, FOOT_PLAN_EXPONENT=3.0, FOOT_SIDE_EXPONENT=5.0,
            FOOT_COST={"length": 1.0, "width": 1.3, "height": 0.25},
            # Weight: the display pulls the CoM forward.  With an ankle pitch joint the
            # CoM must sit over that axis, or the stance ankle holds the robot up all
            # the time (44 % of stall on one foot).  The pitch axis moves under the CoM;
            # the battery stays low in the belly (in the backpack no foot would lift).
            ELECTRONICS_AT="belly", PITCH_AXIS_X="auto")),
    "v11_newbase_display": dict(
        title="v9 layout on the new display character: its own feet hold the roll motors",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True),
    "v12_slipper_display": dict(
        title="v11 at 500 mm: roll motor out in the open at the ankle, toe cap as a slipper",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(
            # duck-robot style ankle: the roll motor lies on a sole plate under the
            # leg, in the open; the front of the STL foot is a toe cap on the plate
            FOOT_STYLE="slipper", MESH_SCALE=500.0 / 120.0, CHAR_HEIGHT=500.0,
            ANKLE_MOTOR=_os.environ.get("HAETBURI_ANKLE_MOTOR", "standing"),
                                    # standing: long side up (narrow, low axis, taller ankle);
                                    # lying: on its side (low ankle, but as wide as the foot)
            ANKLE_INSET=8.0,        # mm: standing motor slid inward from the foot centre (max ~8.9,
                                    # its inner face then meets the foot's inner edge); the roll
                                    # axis moves inboard, so less roll lifts the other foot
            SOLE_PLATE=3.0,         # mm
            HEEL_INSET=1.0,         # mm: motor back this far in from the heel
            SLIPPER_CUT_Y=-8.2,     # native: toe cap ends just in front of the leg (front at -7.7)
            LEG_CUT_Z=7.4,          # native: the leg is capped flat here (its open end sank into the foot)
            BRACKET_T=3.0,          # mm, ankle bracket plates
            # the 500 mm belly is shallow behind the hips: the hip cases point forward and
            # sit 15 mm up (98 % inside); the waist goes above them
            HIP_LONG_END="front", HIP_YAW_LIFT=15.0, CUT_WAIST=22.0,
            ELECTRONICS_POS=(22.0, 0.0, 30.0))),     # between the forward-pointing hip cases
    # v14: walking first, as small as possible.  v12's layout (ankle roll motor out in
    # the open on a sole plate, toe cap as a slipper) with smaller servos; the waist is
    # fixed (walking does not use it, and its motor stacked above the hip motors set the
    # minimum size).  Everything shrinks with the character: the display keeps its
    # proportion to the body (210.5 mm at 500 mm), electronics, walls and brackets too
    # (SCALE_EXTRAS).  The size is the smallest at which both hip yaw motors still sit
    # wholly inside the belly.
    "v14_xc330": dict(title="v12 layout, Dynamixel XC330-T288 (12 V), fixed waist, 300 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=300.0, hip_lift=26.0))),
    "v14_xl330": dict(title="v12 layout, Dynamixel XL330-M288 (5 V), fixed waist, 300 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XL330-M288-T", char=300.0, hip_lift=26.0))),
    "v14_sts3032": dict(title="v12 layout, Feetech STS3032 (6 V), fixed waist, 228 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=228.0, hip_lift=12.0))),
    "v14_scs0009": dict(title="v12 layout, Feetech SCS0009 (6 V micro), fixed waist, 222 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="SCS0009", char=222.0, hip_lift=13.0))),
    # v14 *_hidden: the same servos, every motor inside the character (v11's foot: the
    # roll motor lies in the STL foot, flat sole).  The foot sets the size now: the
    # smallest scale at which the walled motor fits it (walls 1.5 mm, floor 2 mm).
    "v14_xc330_hidden": dict(title="XC330-T288, all motors inside (roll motor in the STL foot), 516 mm",
                             LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                             after=dict(SMALL=dict(servo="XC330-T288-T", char=516.0, hip_lift=20.0, hidden=True))),
    "v14_xl330_hidden": dict(title="XL330-M288, all motors inside (roll motor in the STL foot), 516 mm",
                             LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                             after=dict(SMALL=dict(servo="XL330-M288-T", char=516.0, hip_lift=20.0, hidden=True))),
    "v14_sts3032_hidden": dict(title="STS3032, all motors inside (roll motor in the STL foot), 390 mm",
                               LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                               after=dict(SMALL=dict(servo="STS3032", char=390.0, hip_lift=12.0, hidden=True))),
    "v14_scs0009_hidden": dict(title="SCS0009, all motors inside (roll motor in the STL foot), 384 mm",
                               LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                               after=dict(SMALL=dict(servo="SCS0009", char=384.0, hip_lift=12.0, hidden=True))),
    # v15: v14 *_hidden with the ankle pitch joint back (6 motors, all inside): v13's
    # stacked ankle - roll motor lying in the STL foot, pitch motor standing in the leg
    # above it, the leg lengthened by its own section.  The size is the larger of the
    # foot's and the leg section's minimum (the pitch shaft runs across the leg, and the
    # Feetech servos' shaft is along their longest side).
    "v15_sts3032": dict(title="STS3032, ankle pitch + roll stacked, all 6 motors inside, 456 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=456.0, hip_lift=12.0, stack=True, elec="real", shape=True))),
    "v15_scs0009": dict(title="SCS0009, ankle pitch + roll stacked, all 6 motors inside, 432 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="SCS0009", char=432.0, hip_lift=12.0, stack=True, elec="real", shape=True))),
    "v15_xc330": dict(title="XC330-T288, ankle pitch + roll stacked, all 6 motors inside, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=516.0, hip_lift=20.0, stack=True, elec="real", shape=True))),
    "v15_xl330": dict(title="XL330-M288, ankle pitch + roll stacked, all 6 motors inside, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XL330-M288-T", char=516.0, hip_lift=20.0, stack=True, elec="real", shape=True))),
    # v16: v15's two ankle joints with the STL legs left exactly as they are: both ankle
    # motors inside the STL foot (roll standing, pitch lying on its output, newbase.fit_two),
    # the character scaled until they fit.  IMU feedback / RL is assumed (the controller
    # finds the lean that puts the CoM over the pitch axis), so the size is not grown to put
    # the axis under the CoM; the axis just goes as near it as the foot allows.
    "v16_sts3032": dict(title="STS3032, ankle pitch + roll both in the STL foot, legs unchanged, 522 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=522.0, hip_lift=12.0, two=True, elec="real"))),
    "v16_scs0009": dict(title="SCS0009, ankle pitch + roll both in the STL foot, legs unchanged, 510 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="SCS0009", char=510.0, hip_lift=12.0, two=True, elec="real"))),
    "v16_xc330": dict(title="XC330-T288, ankle pitch + roll both in the STL foot, legs unchanged, 642 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=642.0, hip_lift=20.0, two=True, elec="real"))),
    "v16_xl330": dict(title="XL330-M288, ankle pitch + roll both in the STL foot, legs unchanged, 642 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XL330-M288-T", char=642.0, hip_lift=20.0, two=True, elec="real"))),
    "v16_xm430": dict(title="XM430-W350 (12 V), ankle pitch + roll both in the STL foot, legs unchanged, 852 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XM430-W350-T", char=852.0, hip_lift=20.0, two=True, elec="real"))),
    # v17: the ankle pitch joint traded for a hip pitch joint.  v14 hidden (roll motor in the
    # STL foot, legs and feet the STL's), plus a pitch motor standing in the belly under each
    # hip yaw motor (newbase.fit_hip_pitch), its axis as near the upper body's CoM as fits.
    "v17_sts3032": dict(title="STS3032, hip yaw + hip pitch in the belly, ankle roll in the foot, legs unchanged, 402 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=402.0, hip_lift=12.0, hidden=True,
                                              hip_pitch=True, elec="real"))),
    "v17_scs0009": dict(title="SCS0009, hip yaw + hip pitch in the belly, ankle roll in the foot, legs unchanged, 384 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="SCS0009", char=384.0, hip_lift=12.0, hidden=True,
                                              hip_pitch=True, elec="real"))),
    "v17_xc330": dict(title="XC330-T288, hip yaw + hip pitch in the belly, ankle roll in the foot, legs unchanged, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=516.0, hip_lift=20.0, hidden=True,
                                            hip_pitch=True, elec="real"))),
    "v17_xl330": dict(title="XL330-M288, hip yaw + hip pitch in the belly, ankle roll in the foot, legs unchanged, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XL330-M288-T", char=516.0, hip_lift=20.0, hidden=True,
                                            hip_pitch=True, elec="real"))),
    # v18: v17 with the legs cut off whole at the crotch, so the whole leg swings; the lower
    # belly is carved around the legs' sweep instead (newbase.hip_carve)
    "v18_sts3032": dict(title="STS3032, v17 with whole legs: lower belly carved around the leg sweep, 402 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=402.0, hip_lift=12.0, hidden=True,
                                              hip_pitch=True, carve=True, elec="real"))),
    "v18_scs0009": dict(title="SCS0009, v17 with whole legs: lower belly carved around the leg sweep, 384 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="SCS0009", char=384.0, hip_lift=12.0, hidden=True,
                                              hip_pitch=True, carve=True, elec="real"))),
    "v18_xc330": dict(title="XC330-T288, v17 with whole legs: lower belly carved around the leg sweep, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=516.0, hip_lift=20.0, hidden=True,
                                            hip_pitch=True, carve=True, elec="real"))),
    "v18_xl330": dict(title="XL330-M288, v17 with whole legs: lower belly carved around the leg sweep, 516 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XL330-M288-T", char=516.0, hip_lift=20.0, hidden=True,
                                            hip_pitch=True, carve=True, elec="real"))),
    # v19: after images-modeling/PartsAssemblyV2 - the torso ends in a flat bottom with a socket
    # per leg; the legs, carried up to it, end in a hidden dome about the hip pivot (newbase.hip_socket)
    # *_w4: the STS3032 models again with the printed walls they will really have (4 mm
    # body / legs / feet / arms / backpack, 2 mm head / tail, motors 4 mm inside the skin),
    # each at the smallest size its motors fit with those walls
    "v14_sts3032_w4": dict(title="v14_sts3032 with 4 mm walls (2 mm head / tail), 270 mm (the belly needs it for walled hip motors)",
                           LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                           after=dict(SMALL=dict(servo="STS3032", char=270.0, hip_lift=16.0, w4=True))),
    "v14_sts3032_hidden_w4": dict(title="v14_sts3032_hidden with 4 mm walls (2 mm head / tail), 462 mm",
                                  LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                                  after=dict(SMALL=dict(servo="STS3032", char=462.0, hip_lift=12.0, hidden=True,
                                                        w4=True))),
    "v15_sts3032_w4": dict(title="v15_sts3032 with 4 mm walls (2 mm head / tail), 480 mm",
                           LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                           after=dict(SMALL=dict(servo="STS3032", char=480.0, hip_lift=12.0, stack=True, elec="real",
                                                 shape=True, w4=True))),
    "v16_sts3032_w4": dict(title="v16_sts3032 with 4 mm walls (2 mm head / tail), 594 mm",
                           LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                           after=dict(SMALL=dict(servo="STS3032", char=594.0, hip_lift=12.0, two=True, elec="real",
                                                 w4=True))),
    "v17_sts3032_w4": dict(title="v17_sts3032 with 4 mm walls (2 mm head / tail), 476 mm",
                           LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                           after=dict(SMALL=dict(servo="STS3032", char=476.0, hip_lift=12.0, hidden=True,
                                                 hip_pitch=True, elec="real", w4=True))),
    "v19_sts3032": dict(title="STS3032, after PartsAssemblyV2: flat torso bottom with leg sockets, legs carried up into it, 476 mm, 4 mm walls",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(SMALL=dict(servo="STS3032", char=476.0, hip_lift=12.0, hidden=True,
                              hip_pitch=True, socket=True, elec="real", w4=True))),
    "v19_scs0009": dict(title="SCS0009, after PartsAssemblyV2: flat torso bottom with leg sockets, legs carried up into it, 458 mm, 4 mm walls",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(SMALL=dict(servo="SCS0009", char=458.0, hip_lift=12.0, hidden=True,
                              hip_pitch=True, socket=True, elec="real", w4=True))),
    "v19_xc330": dict(title="XC330-T288-T, after PartsAssemblyV2: flat torso bottom with leg sockets, legs carried up into it, 588 mm, 4 mm walls",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(SMALL=dict(servo="XC330-T288-T", char=588.0, hip_lift=20.0, hidden=True,
                              hip_pitch=True, socket=True, elec="real", w4=True))),
    "v19_xl330": dict(title="XL330-M288-T, after PartsAssemblyV2: flat torso bottom with leg sockets, legs carried up into it, 588 mm, 4 mm walls",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(SMALL=dict(servo="XL330-M288-T", char=588.0, hip_lift=20.0, hidden=True,
                              hip_pitch=True, socket=True, elec="real", w4=True))),
    # v20: hip pitch + ankle pitch + ankle roll (no hip yaw): v19's sockets at the hip, v15's
    # stacked ankle (roll motor in the foot, pitch motor in the leg), 4 mm walls
    "v20_sts3032": dict(title="STS3032, hip pitch + ankle pitch + ankle roll, sockets, 4 mm walls, 480 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=480.0, hip_lift=12.0, stack=True, shape=True,
                                              v20=True, elec="placed", w4=True))),
    "v20_xc330": dict(title="XC330-T288, hip pitch + ankle pitch + ankle roll, sockets, 4 mm walls, 588 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=588.0, hip_lift=20.0, stack=True, shape=True,
                                            v20=True, elec="placed", w4=True))),
    # v21: v20 with a straight leg (one section top to bottom) and the lower torso re-lofted
    # over the legs so they come out from under it (newbase.fit_torso_bottom)
    "v21_sts3032": dict(title="STS3032, v20 with straight legs and the lower torso fitted to them, 480 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=480.0, hip_lift=12.0, stack=True, shape=True,
                                              v20=True, v21=True, elec="placed", w4=True))),
    "v21_xc330": dict(title="XC330-T288, v20 with straight legs and the lower torso fitted to them, 588 mm",
                      LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                      after=dict(SMALL=dict(servo="XC330-T288-T", char=588.0, hip_lift=20.0, stack=True, shape=True,
                                            v20=True, v21=True, elec="placed", w4=True))),
    "v21_sts3032_lite": dict(title="v21_sts3032 with a lighter display panel (0.25 kg with its mount, was 0.58)",
                             LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                             after=dict(SMALL=dict(servo="STS3032", char=480.0, hip_lift=12.0, stack=True, shape=True,
                                                   v20=True, v21=True, elec="placed", w4=True, display_kg=0.20))),
    # v22: v21 with a real 7-inch portable monitor (PeakDo class: 152 g, 175.5 x 98.5 x 9.4 mm)
    "v22_sts3032": dict(title="STS3032, v21 holding a real 7-inch portable monitor (152 g), 480 mm",
                        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                        after=dict(SMALL=dict(servo="STS3032", char=480.0, hip_lift=12.0, stack=True, shape=True,
                                              v20=True, v21=True, elec="placed", w4=True, display="7in"))),
    "v14_sts3032_hidden_7in": dict(title="v14_sts3032_hidden_w4 holding a real 7-inch portable monitor (152 g), "
                                         "electronics placed, 462 mm",
                                   LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
                                   after=dict(SMALL=dict(servo="STS3032", char=462.0, hip_lift=12.0, hidden=True,
                                                         w4=True, display="7in", elec="placed"))),
    "v13_ankle_pitch_display": dict(
        title="v11 + the ankle pitch joint back: pitch motor in the foot, roll motor stacked in the leg",
        LAYOUT="newbase", HIP_JOINT="yaw", DISPLAY=True,
        after=dict(
            # Stacked ankle (v5/v6/v10 style): one ankle motor in the STL foot, the other in
            # the leg above it; the leg is lengthened by its own section just enough.
            # Roll in the foot (lying, v11; fits both flat-soled feet from x5.95) keeps the
            # roll axis low (~17 mm); pitch lies in the leg, axis under the CoM.
            # (Pitch in the foot / roll standing in the leg fit from x5.6, but the roll axis
            # sat 72 mm up and rolling tipped the feet onto their edges: no foot lifted.
            # Both motors in the foot needed x8.0, ~1 m, and the XL430 could not lift a foot.)
            FOOT_STYLE="stacked", STACK_UP="pitch", MESH_SCALE=5.95, CHAR_HEIGHT=120.0 * 5.95,
            FOOT_MOTOR_THETA=90.0,
            PITCH_AXIS_AHEAD_NATIVE=1.26,   # CoM ahead of the leg centre (9.1 mm at x7.2)
            LEG_CUT_Z=8.6,                  # native: the STL leg is used from here up (below it
                                            # sinks into the foot and its surface has holes)
            LEG_JOIN_OVERLAP=0.4,           # native: the lengthened part reaches this far up inside it
            LEG_JOIN_INSET=0.03,            # native: ... and is this much thinner, so it stays hidden
            LEG_TOP_CLEAR=1.0,              # mm between the roll case top and the crotch
            LINERS=True,                    # line the STL's holes in the shell colour
            # hip cases: long end forward, 24 mm up -> wholly inside (pointing back they
            # poked 1-4 % out through the back of the thighs); the waist goes above them
            HIP_LONG_END="front", HIP_YAW_LIFT=24.0, CUT_WAIST=20.8,
            ELECTRONICS_POS=(22.0, 0.0, 30.0),      # between the forward-pointing hip cases
            LIM={"waist_yaw": (-60.0, 60.0), "hip_pitch": (0.0, 0.0), "hip_yaw": (-30.0, 30.0),
                 "ankle_pitch": (-15.0, 15.0), "ankle_roll": (-15.0, 15.0)},
            DEMO=dict(pitch_lean=5, roll_lean=5, yaw=35, bow_hip=0, bow_ankle=0, diag_roll=5,
                      diag_yaw=-30, pose_bow_hip=0, pose_bow_ankle=0, pose_lift_roll=5,
                      pose_lift_hip=8))),
}
VARIANT = _os.environ.get("HAETBURI_VARIANT", DEFAULT_VARIANT)
if VARIANT not in VARIANTS:
    raise KeyError(f"unknown HAETBURI_VARIANT {VARIANT!r}; choose from {list(VARIANTS)}")
OUT_DIR = _os.environ.get("HAETBURI_OUT") or _os.path.join(ROOT, "robot", VARIANT)   # override: experiments
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
FOOT_MAX_H = None                           # optional cap on the foot height
PITCH_AXIS_X = 0.0          # ankle pitch axis ahead of the leg centre (mm), or "auto":
                            # put it right under the robot's centre of mass
LEG_FOOT_CLEAR = 2.0

# 9 inch display held in both hands (images-modeling/Haetburi_display.png).
# Waveshare 9inch 2560x1600 Monitor: 210.5 x 142.3 x 11.5 mm outline, 192 x 120 mm
# active area, listed weight 0.621 kg (used as is: it may include packaging,
# which errs on the heavy side for a load check).
DISPLAY = False
DISPLAY_OUTLINE = (11.5, 210.5, 142.3)      # thickness (x), width (y), height (z)
DISPLAY_ACTIVE = (192.0, 120.0)
DISPLAY_MASS_KG = 0.621
DISPLAY_MOUNT_KG = 0.030                    # clips / brackets into the fists
DISPLAY_CLEAR = 3.0                         # gap to the body and head (fists excepted)

# Servo used for every joint (SERVOS below); WAIST False fixes the waist (no motor)
SERVO = "XL430-W250-T"
WAIST = True
SCALE_EXTRAS = False        # True: display, electronics, walls, brackets shrink with the character
LEG_FRONT_EXT_MAX = 0.0     # stacked ankle: how far (native) the added leg part may grow forward

# demo / preview motion amplitudes (degrees)
DEMO = dict(pitch_lean=7, roll_lean=8, yaw=35, bow_hip=16, bow_ankle=-13,
            diag_roll=7, diag_yaw=-30,
            pose_bow_hip=18, pose_bow_ankle=-12,     # held 3 s for the preview images
            pose_lift_roll=6.5, pose_lift_hip=15)

globals().update({k: v for k, v in VARIANTS[VARIANT].items() if k not in ("title", "after")})
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
# ========================================================= layout: newbase
elif LAYOUT == "newbase":
    # v9's joints (waist yaw, hip yaw, ankle roll) on the new character
    # (modelings/haetburi_display_parts.stl, built by tools/make_display_model.py:
    # tail on, fists on the display).  Its feet keep their own shape; the whole
    # character is scaled until a walled XL430 fits inside them (tools/newbase.py).
    # Native units: the STL's 120-tall frame, not the old 300 one.
    NATIVE_HEIGHT = 120.0
    FOOT_FLAT_Z = 1.5           # native: below this the bun becomes straight sides on a
    FOOT_FILLET = 0.4           # flat sole (rounded edge), as big as the outline at that height
    FOOT_MOTOR_THETA = 90.0     # roll motor lying on its side, long end outward:
                                # needs the smallest scale (standing up: x6.97)
    MESH_SCALE = 5.95           # mm per native unit; the motor fits both flat-soled feet from x5.95
                                # (the round-bottomed STL foot needed x6.32; a motor fixed in the
                                # leg and reaching down into the foot needs x5.88, x5.74 if it
                                # also reaches 4 native up into the belly: not worth it)
    CHAR_HEIGHT = NATIVE_HEIGHT * MESH_SCALE
    CUT_LEG_TOP = 9.8           # the two legs merge into the belly at z 9.5..10
    CUT_WAIST = 16.0            # above the hip cases (top at 15.5), below the backpack (21.6)
    LEG_X_LIMIT = 12.0
    LEG_HALF_SPACING_NATIVE = 5.66      # leg centres at x -5.74 / +5.58 (only used by shared helpers)
    NECK_NATIVE = 43.0          # head starts at 42.3
    ARM_X_NATIVE = 14.5         # the body is +-14.2 wide; outboard of this is arm
    HIP_YAW_LIFT = WALL         # hip case just above the crotch: the belly is deep enough
    ELECTRONICS_AT = "belly"
    ELECTRONICS_POS = (35.0, 0.0, 18.0)     # pelvis frame: in front of the hip cases
    LIM = {"waist_yaw": (-60.0, 60.0), "hip_pitch": (0.0, 0.0), "hip_yaw": (-30.0, 30.0),
           "ankle_pitch": (0.0, 0.0), "ankle_roll": (-15.0, 15.0)}
    DEMO = dict(DEMO, pitch_lean=0, roll_lean=5, diag_roll=5, bow_hip=0, bow_ankle=0,
                pose_bow_hip=0, pose_bow_ankle=0, pose_lift_roll=5, pose_lift_hip=8)
else:
    raise ValueError(LAYOUT)

# per-variant values that must win over the layout block
globals().update(VARIANTS[VARIANT].get("after", {}))

# ------------------------------------------------------------------ servos
# W x H x D (mm, shaft along D, AX from one end of H), mass, stall torque (N.m) and
# no-load speed (rad/s) at the voltage named; gear ratio for the reflected rotor inertia.
# AX of the 330 cases and the STS3032 is estimated (drawings not checked).
SERVOS = {
    "XL430-W250-T": dict(W=28.5, H=46.5, D=34.0, AX=11.25, HORN=2.5, MASS=0.0572,
                         STALL=1.4, NOLOAD=5.97, GEAR=258.5),             # 11.1 V
    "XC330-T288-T": dict(W=20.0, H=34.0, D=26.0, AX=8.2, HORN=2.0, MASS=0.023,
                         STALL=1.0, NOLOAD=7.43, GEAR=288.35),            # 12 V, 71 rpm
    "XL330-M288-T": dict(W=20.0, H=34.0, D=26.0, AX=8.2, HORN=2.0, MASS=0.018,
                         STALL=0.52, NOLOAD=10.79, GEAR=288.35),          # 5 V, 103 rpm
    "STS3032":      dict(W=12.0, H=23.0, D=27.5, AX=6.0, HORN=2.0, MASS=0.025,
                         STALL=0.44, NOLOAD=11.6, GEAR=300.0),            # 6 V, 4.5 kg.cm, 0.09 s/60 deg
    "XM430-W350-T": dict(W=28.5, H=46.5, D=34.0, AX=11.25, HORN=2.5, MASS=0.082,
                         STALL=4.1, NOLOAD=4.82, GEAR=353.5),             # 12 V, 46 rpm (XL430 case)
    "SCS0009":      dict(W=12.1, H=23.2, D=25.25, AX=6.0, HORN=2.0, MASS=0.0132,
                         STALL=0.226, NOLOAD=10.5, GEAR=416.0),           # 6 V, 2.3 kg.cm, 0.1 s/60 deg
}
if "SMALL" in globals():
    # v14: v12's slipper layout, re-sized around a small servo
    _sm = SMALL
    SERVO, WAIST, SCALE_EXTRAS = _sm["servo"], False, True
    if _sm.get("two"):             # both ankle motors inside the STL foot, legs untouched
        FOOT_STYLE, PITCH_LONG_END, LINERS = "inside2", "toe", True     # toe: the axis reaches the CoM
        WALL, FLOOR, CLEAR = 1.5, 2.0, 1.0
        PITCH_AXIS_AHEAD_NATIVE = "auto"
        LIM = dict(LIM, ankle_pitch=(-15.0, 15.0), ankle_roll=(-12.0, 12.0))
        DEMO = dict(DEMO, pitch_lean=5)
    elif _sm.get("stack"):         # v13's stacked ankle: roll in the foot, pitch in the leg
        FOOT_STYLE, STACK_UP, FOOT_MOTOR_THETA, LINERS = "stacked", "pitch", 90.0, True
        WALL, FLOOR, CLEAR = 1.5, 2.0, 1.0
        PITCH_AXIS_AHEAD_NATIVE, LEG_CUT_Z = 1.26, 8.6
        LEG_JOIN_OVERLAP, LEG_JOIN_INSET, LEG_TOP_CLEAR = 0.4, 0.03, 1.0
        LIM = dict(LIM, ankle_pitch=(-15.0, 15.0), ankle_roll=(-15.0, 15.0))
        DEMO = dict(DEMO, pitch_lean=5)
        if _sm.get("shape"):       # pitch axis under the measured CoM; the added part of the
            PITCH_AXIS_AHEAD_NATIVE, LEG_FRONT_EXT_MAX = "auto", 3.0     # leg may reach forward
    elif _sm.get("hidden"):        # roll motor inside the STL foot (v11), holes lined
        FOOT_STYLE, FOOT_MOTOR_THETA, LINERS = "inside", 90.0, True
        WALL, FLOOR, CLEAR = 1.5, 2.0, 1.0
        if _sm.get("hip_pitch"):   # v17: hip pitch motor in the belly under the hip yaw motor
            HIP_PITCH, HIP_PITCH_AHEAD = True, "auto"
            HIP_SEAM, HIP_SEAM_GAP = "sphere", 1.0     # leg / belly seam: sphere about the hip axes, 1 mm gap
            HIP_PITCH_TARGET = 20.0                    # the sphere is made just big enough for +-20 deg
            if _sm.get("carve"):   # v18: whole legs, belly carved around their sweep
                HIP_SEAM, HIP_CARVE_PITCH, HIP_CARVE_YAW = "carve", (-30.0, 25.0), 20.0
            if _sm.get("socket"):  # v19: flat torso bottom with leg sockets (PartsAssemblyV2)
                HIP_SEAM, HIP_CARVE_PITCH, HIP_CARVE_YAW = "socket", (-30.0, 25.0), 20.0
            LIM = dict(LIM, hip_pitch=(-45.0, 45.0))   # cut back to the collision-free range by the build
            DEMO = dict(DEMO, bow_hip=10, pose_bow_hip=8)
    else:                          # roll motor out in the open, toe cap (v12)
        FOOT_STYLE, ANKLE_MOTOR, ANKLE_INSET = "slipper", "standing", "max"
    if _sm.get("v20"):
        # v20: hip pitch (no hip yaw) in the belly, stacked ankle (pitch in the leg, roll in
        # the foot), torso / leg sockets as v19: the ankle pitch keeps the sole level while
        # the hip pitch swings the leg
        HIP_PITCH, HIP_YAW, HIP_PITCH_AHEAD = True, False, "auto"
        HIP_SEAM, HIP_SEAM_GAP, HIP_CARVE_PITCH, HIP_CARVE_YAW = "socket", 1.0, (-30.0, 25.0), 0.0
        LIM = dict(LIM, hip_pitch=(-45.0, 45.0), hip_yaw=(0.0, 0.0))
        DEMO = dict(DEMO, bow_hip=10, pose_bow_hip=8)
        if _sm.get("v21"):         # v21: one straight leg section, the lower torso fitted to the legs
            LEG_UNIFORM, TORSO_FIT, TORSO_FIT_BLEND = True, True, 6.0
    CHAR_HEIGHT = _sm["char"]
    MESH_SCALE = CHAR_HEIGHT / 120.0
    SOLE_PLATE, HEEL_INSET, BRACKET_T = 2.0, 1.0, 2.0
    if FOOT_STYLE == "slipper":
        SLIPPER_CUT_Y, LEG_CUT_Z = -8.2, 7.4
    HIP_LONG_END, HIP_YAW_LIFT = "front", _sm["hip_lift"]
    if _sm.get("w4"):
        # printed walls as they will be made: 4 mm for the body, legs, feet, arms and
        # backpack, 2 mm for the head and tail; every motor sits 4 mm inside the skin
        WALL_MM, SHELL_WALL = 4.0, {"head": 2.0, "tail": 2.0}
        WALL, FLOOR = 4.0, 4.0
    LEG_FOOT_CLEAR = 1.5
if SERVO != "XL430-W250-T":
    _sv = SERVOS[SERVO]
    _r = _sv["STALL"] / MOTOR_STALL_NM
    MOTOR_W, MOTOR_H, MOTOR_D = _sv["W"], _sv["H"], _sv["D"]
    MOTOR_AXIS_FROM_END, MOTOR_HORN, MOTOR_MASS_KG = _sv["AX"], _sv["HORN"], _sv["MASS"]
    MOTOR_ROTOR_INERTIA = MOTOR_ROTOR_INERTIA * _r * (MOTOR_GEAR / _sv["GEAR"]) ** 2
    MOTOR_STALL_NM, MOTOR_NOLOAD_RADS, MOTOR_GEAR = _sv["STALL"], _sv["NOLOAD"], _sv["GEAR"]
    # the servo's stiffness and losses scale with its torque (kp: stall at ~2.7 deg error)
    ACT_KP, ACT_KV = ACT_KP * _r, ACT_KV * _r
    JOINT_DAMPING, JOINT_FRICTION = JOINT_DAMPING * _r, JOINT_FRICTION * _r
    BRACKET_MASS_KG = BRACKET_MASS_KG * _sv["MASS"] / 0.0572
if SCALE_EXTRAS:
    # everything else shrinks with the character (reference: 500 mm, 9 inch display)
    _s = CHAR_HEIGHT / 500.0
    DISPLAY_OUTLINE = tuple(v * _s for v in DISPLAY_OUTLINE)
    DISPLAY_ACTIVE = tuple(v * _s for v in DISPLAY_ACTIVE)
    # smaller panels shed weight with their volume; bigger ones (portable monitors) grow
    # roughly with their area, not their volume
    DISPLAY_MASS_KG = DISPLAY_MASS_KG * (_s ** 3 if _s <= 1.0 else _s ** 2)
    DISPLAY_MOUNT_KG = DISPLAY_MOUNT_KG * _s ** 2
    if "SMALL" in globals() and SMALL.get("display_kg"):     # a lighter panel, same outline
        DISPLAY_MASS_KG = SMALL["display_kg"]
    if "SMALL" in globals() and SMALL.get("display") == "7in":
        # a real 7-inch HDMI portable monitor, not scaled with the robot: PeakDo 7" (152 g,
        # 9.4 mm thick; mini HDMI + USB-C), outline as the same-class Wisecoco 7" (175.5 x
        # 98.5 mm; PeakDo does not publish it), 16:9 active area of a 7-inch panel
        DISPLAY_OUTLINE, DISPLAY_ACTIVE = (9.4, 175.5, 98.5), (154.2, 86.6)
        DISPLAY_MASS_KG = 0.152
    HEAD_EXTRA_KG = HEAD_EXTRA_KG * _s ** 3
    # battery + controller: scales with the servos' power (XL430 x5: 0.30 kg), at least 60 g
    ELECTRONICS_KG = max(0.06, ELECTRONICS_KG * MOTOR_STALL_NM / 1.4)
    ELECTRONICS_POS = (22.0 * _s, 0.0, 30.0 * _s)
    if not ("SMALL" in globals() and SMALL.get("w4")):
        WALL_MM = max(1.2, WALL_MM * _s)             # printed wall: 3 mm at 500 mm
    CUT_WAIST = 22.0
    if "SMALL" in globals() and SMALL.get("elec") in ("real", "placed"):
        # the electronics where they actually fit (see the v14 electronics notes): battery
        # and Raspberry Pi in the backpack, servo driver (IMU) in the chest, regulators
        # in the belly.  (native offsets from the leg centre / crotch: + behind, + up)
        _bat = 0.085 if MOTOR_STALL_NM >= 0.9 else 0.060          # 3S for the 12 V XC330, else 2S
        ELECTRONICS_ITEMS = [(_bat, 10.77, 17.23), (0.012, 7.38, 19.08),     # battery, Pi Zero 2 W
                             (0.020, -4.00, 14.15),                           # Servo Driver with ESP32
                             (0.012, -0.62, 4.92), (0.005, -2.77, 1.85),      # 6 V UBEC, 5 V buck
                             (0.015, 0.0, 12.0)]                              # wiring
        _ep = _os.path.join(OUT_DIR, "electronics.json")
        if SMALL.get("elec") == "placed" and _os.path.exists(_ep):
            # placed by tools/place_electronics.py: (kg, behind, up, side) and the boxes to draw
            import json as _json
            _pl = _json.load(open(_ep))
            ELECTRONICS_ITEMS = [(p["kg"], p["behind"], p["up"], p["side"]) for p in _pl if not p.get("unplaced")]
            ELECTRONICS_BOXES = [(p["size_mm"], p["behind"], p["up"], p["side"]) for p in _pl if p["size_mm"] and not p.get("unplaced")]

# experiment hooks: move the battery or cap the foot height without editing a variant
# (HAETBURI_ELECTRONICS=belly|chest|backpack, HAETBURI_FOOT_MAX_H=<mm>)
if _os.environ.get("HAETBURI_FOOT_MAX_H"):
    FOOT_MAX_H = float(_os.environ["HAETBURI_FOOT_MAX_H"])
if _os.environ.get("HAETBURI_ELECTRONICS"):
    ELECTRONICS_AT = _os.environ["HAETBURI_ELECTRONICS"]
    if ELECTRONICS_AT == "chest":
        ELECTRONICS_POS = (22.0, 0.0, 30.0)          # torso frame, in front of the waist motor
