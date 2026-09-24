#!/usr/bin/env python3
"""Open the Haetburi scene in the interactive MuJoCo viewer.

    python3 sim/view.py [--variant v6_split_round]

Drag with the left mouse button to orbit, right button to pan, scroll to zoom.
Press Tab for the control panel, then the "Control" tab to drag the seven
joint sliders by hand.  Keys 2 / 3 toggle the visual and collision geom groups.
"""
import os

import mujoco
import mujoco.viewer

import variants


def main():
    a = variants.parser(__doc__).parse_args()
    m = mujoco.MjModel.from_xml_path(variants.scene(a.variant))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0, 0, 0.35]
        v.cam.distance = 1.6
        v.cam.azimuth, v.cam.elevation = 200, -12
        import time
        while v.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
