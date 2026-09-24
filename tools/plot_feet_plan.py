#!/usr/bin/env python3
"""Plan view (from above) of the feet: sole outline, straight leg section,
and the source leg tip, with dimensions.   -> review/feet_v1_plan.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely.geometry as sg
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_feet as F  # noqa: E402

OUT = os.path.join(F.ROOT, "review", "feet_v1_plan.png")


def outlines(mesh, z):
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    polys = [sg.Polygon(l[:, :2]) for l in s.discrete if len(l) > 3]
    return [p for p in polys if p.area > 20 and abs(p.centroid.x) < 40]


def main():
    src, new = trimesh.load(F.SRC), trimesh.load(F.OUT)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    layers = [(new, 0.3, "#1b3789", "-", 2.2, "new sole (z=0.3)"),
              (new, F.FOOT_HEIGHT + 1.5, "#d08a1e", "-", 1.6,
               f"new straight leg (z={F.FOOT_HEIGHT + 1.5:.1f})"),
              (src, 0.5, "#888888", "--", 1.4, "source leg tip (z=0.5)"),
              (src, 8.0, "#bbbbbb", ":", 1.2, "source leg at z=8")]
    for mesh, z, col, ls, lw, lab in layers:
        for k, p in enumerate(outlines(mesh, z)):
            x, y = p.exterior.xy
            # plot with the face pointing up the page: page-up = -y_stl
            ax.plot(x, [-v for v in y], color=col, ls=ls, lw=lw, label=lab if k == 0 else None)
    for p in outlines(new, 0.3):
        b = p.bounds
        cx = (b[0] + b[2]) / 2
        ax.annotate(f"{b[2]-b[0]:.0f} mm wide", (cx, -b[1] + 1.2), ha="center", fontsize=9)
        ax.annotate(f"{b[3]-b[1]:.0f} mm long", (b[2] + 1.5 if cx > 0 else b[0] - 1.5, -(b[1] + b[3]) / 2),
                    ha="left" if cx > 0 else "right", va="center", fontsize=9, rotation=90)
    ax.annotate("face / toes", (0, 31), ha="center", fontsize=10, color="#444")
    ax.annotate("", (0, 30), (0, 20), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel("x (mm): robot right  <-  ->  robot left")
    ax.set_ylabel("fore-aft (mm), toes up")
    ax.set_title("feet_v1, plan view (source STL scale, 300 mm tall)")
    ax.legend(loc="lower center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
