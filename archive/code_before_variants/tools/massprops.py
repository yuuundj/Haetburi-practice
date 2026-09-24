"""Rigid-body mass properties for thin printed shells and for solid boxes."""
import numpy as np

import haetburi_params as P


def shell_props(mesh, wall_m=None, rho=None, drop_planes=()):
    """Mass properties of `mesh` treated as a thin shell of uniform wall thickness.

    `mesh` must already be expressed in the link frame, in metres.
    `drop_planes` lists z values (metres) of cut planes whose cap triangles are
    interior surfaces of the cut, not printed wall, and must not carry mass.
    Returns (mass_kg, com_m[3], inertia_about_com[3,3]).
    """
    wall_m = P.WALL_MM * P.RIB_FACTOR * 1e-3 if wall_m is None else wall_m
    rho = P.RHO_PLA * 1e6 if rho is None else rho          # g/mm^3 -> kg/m^3

    tris = mesh.triangles                                   # (n,3,3)
    area = mesh.area_faces                                  # (n,)
    cent = tris.mean(axis=1)                                # (n,3)

    keep = np.ones(len(area), dtype=bool)
    if len(drop_planes):
        nz = np.abs(mesh.face_normals[:, 2])
        for z in drop_planes:
            flat = nz > 0.999
            at = np.abs(tris[:, :, 2] - z).max(axis=1) < 1e-6
            keep &= ~(flat & at)

    area, cent = area[keep], cent[keep]
    m = area * wall_m * rho                                 # per-triangle mass
    return _lump(m, cent)


def box_props(mass, size_m, centre_m):
    """Mass properties of a homogeneous box (full 3x3 inertia about its own com)."""
    sx, sy, sz = size_m
    I = np.diag([mass * (sy * sy + sz * sz) / 12.0,
                 mass * (sx * sx + sz * sz) / 12.0,
                 mass * (sx * sx + sy * sy) / 12.0])
    return mass, np.asarray(centre_m, float), I


def point_props(mass, centre_m):
    return mass, np.asarray(centre_m, float), np.zeros((3, 3))


def _lump(masses, points):
    masses = np.asarray(masses, float)
    total = masses.sum()
    com = (masses[:, None] * points).sum(axis=0) / total
    r = points - com
    I = np.zeros((3, 3))
    I[0, 0] = (masses * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum()
    I[1, 1] = (masses * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum()
    I[2, 2] = (masses * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum()
    I[0, 1] = I[1, 0] = -(masses * r[:, 0] * r[:, 1]).sum()
    I[0, 2] = I[2, 0] = -(masses * r[:, 0] * r[:, 2]).sum()
    I[1, 2] = I[2, 1] = -(masses * r[:, 1] * r[:, 2]).sum()
    return total, com, I


def combine(items):
    """Merge (mass, com, I_about_own_com) triples into one body."""
    total = sum(m for m, _, _ in items)
    com = sum(m * np.asarray(c) for m, c, _ in items) / total
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        r = np.asarray(c) - com
        I += Ic + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return total, com, I
