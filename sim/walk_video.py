"""Render a gait: an mp4 that follows the robot and a filmstrip of one cycle."""
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import variants
import walk

W, H = 640, 528                  # multiples of 16 for the video encoder


def render(variant, g, video=None, strip=None, fps=30):
    base = os.path.join(variants.ROOT, "robot", variant)
    video = video or os.path.join(base, "walk.mp4")
    strip = strip or os.path.join(base, "walk_strip.png")
    m = mujoco.MjModel.from_xml_path(variants.scene(variant))
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    frames, film = [], {"3/4": [], "side": []}
    period = 1.0 / g["freq"]
    t_cycle = walk.T_END - period                       # film the last full cycle
    state = {"next": 0.0, "film": t_cycle}

    def shot(d, az, el, dist):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [d.xpos[pel][0], d.xpos[pel][1], 0.28]
        r.update_scene(d, cam)
        return r.render()

    def rec(m_, d):
        if d.time >= state["next"]:
            state["next"] += 1.0 / fps
            img = Image.fromarray(shot(d, 215, -12, 1.25))
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 26], fill=(18, 20, 28))
            dr.text((8, 7), f"{variant}  t={d.time:5.1f}s  x={d.xpos[pel][0]*1000:+5.0f} mm  "
                            f"{g['freq']} Hz, roll {g['roll']} deg, hip {g['swing']} deg ({g.get('pair', 'anti')})",
                    fill=(240, 240, 240))
            frames.append(np.array(img))
        if d.time >= state["film"] and len(film["side"]) < 6:
            state["film"] += period / 6
            film["3/4"].append(shot(d, 215, -12, 1.1))
            film["side"].append(shot(d, 90, -4, 1.0))

    res = walk.run(variants.scene(variant), g, record=rec)
    import imageio.v2 as imageio
    imageio.mimsave(video, frames, fps=fps, quality=7)
    rows = [np.hstack(film[k]) for k in ("3/4", "side")]
    sheet = Image.fromarray(np.vstack(rows)).resize((W * 6 // 2, H * 2 // 2), Image.LANCZOS)
    dr = ImageDraw.Draw(sheet)
    dr.rectangle([0, 0, sheet.width, 22], fill=(18, 20, 28))
    dr.text((8, 5), f"{variant}: one gait cycle ({period:.1f} s) in 6 frames, 3/4 view (top) and side view "
                    f"(bottom); {res['dx']:.0f} mm forward in {walk.T_END - walk.RAMP:.0f} s", fill=(240, 240, 240))
    sheet.save(strip)
    print("wrote", os.path.relpath(video, variants.ROOT), "and", os.path.relpath(strip, variants.ROOT))
    return res
