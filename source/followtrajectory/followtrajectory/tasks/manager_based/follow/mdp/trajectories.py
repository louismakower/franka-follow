# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generators for the end-effector trajectory bank.

Each generator returns a list of ``[x, y, z, t]`` waypoints (root frame, seconds) describing a
*closed* loop: the first waypoint is repeated at ``t = period`` so the path satisfies scipy's
``bc_type="periodic"`` boundary condition and the downstream modulo-wrap stays seamless.

Coordinates are pure numpy/Python floats so this module has no Isaac Sim dependency and the
``TRAIN_BANK`` constant can be built at import time.
"""

from __future__ import annotations

import numpy as np

# Reachable workspace box for the Franka end-effector, expressed in the robot root frame, as
# ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)). Random loops sample inside an inset of this box.
WORKSPACE_BOUNDS = ((0.35, 0.65), (-0.2, 0.2), (0.15, 0.5))

Waypoints = list[list[float]]


def _plane_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors ``(u, v)`` spanning the plane with the given normal."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    # pick a reference axis that is not (near) parallel to the normal, then Gram-Schmidt it
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def make_circle(center, radius: float, period: float, normal=(0.0, 0.0, 1.0), n_samples: int = 24) -> Waypoints:
    """Closed circular loop of ``radius`` centred at ``center``, lying in the plane with ``normal``."""
    center = np.asarray(center, dtype=float)
    u, v = _plane_basis(normal)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples + 1)  # endpoint included -> first == last
    ts = np.linspace(0.0, period, n_samples + 1)
    out: Waypoints = []
    for theta, t in zip(thetas, ts):
        p = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
        out.append([float(p[0]), float(p[1]), float(p[2]), float(t)])
    return out


def make_figure8(center, rx: float, ry: float, period: float, normal=(0.0, 0.0, 1.0), n_samples: int = 32) -> Waypoints:
    """Closed figure-of-eight (Gerono lemniscate) centred at ``center`` in the plane with ``normal``.

    ``rx`` is the half-extent along the first in-plane axis, ``ry`` the lobe half-height along the
    second. The curve passes through ``center`` at the crossing.
    """
    center = np.asarray(center, dtype=float)
    u, v = _plane_basis(normal)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples + 1)
    ts = np.linspace(0.0, period, n_samples + 1)
    out: Waypoints = []
    for theta, t in zip(thetas, ts):
        a = rx * np.sin(theta)
        b = ry * np.sin(theta) * np.cos(theta)
        p = center + a * u + b * v
        out.append([float(p[0]), float(p[1]), float(p[2]), float(t)])
    return out


def make_random_loop(seed: int, n_pts: int, period: float, bounds=WORKSPACE_BOUNDS, inset: float = 0.85) -> Waypoints:
    """Closed loop through ``n_pts`` random waypoints sampled inside an inset of ``bounds``.

    Waypoints are ordered by angle about their centroid (projected onto x-y) to curb wild
    self-crossings while keeping the shape irregular. The box is shrunk by ``inset`` so the
    interpolating spline is less likely to overshoot the reachable workspace between waypoints.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * inset
    pts = rng.uniform(center - half, center + half, size=(n_pts, 3))
    # order around the centroid to reduce self-intersection
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)]
    # close the loop
    pts = np.vstack([pts, pts[0]])
    ts = np.linspace(0.0, period, n_pts + 1)
    return [[float(p[0]), float(p[1]), float(p[2]), float(t)] for p, t in zip(pts, ts)]


# ---------------------------------------------------------------------------------------------- #
# Training bank: ~9 closed loops. Shared by train and play for now; a held-out EVAL_BANK can be
# dropped in later by overriding ``CommandsCfg.trajectory.trajectories`` in the _PLAY config.
# ---------------------------------------------------------------------------------------------- #
TRAIN_BANK: list[Waypoints] = [
    # circles of varying radius / plane
    make_circle(center=(0.5, 0.0, 0.30), radius=0.12, period=5.0, normal=(0.0, 0.0, 1.0)),  # horizontal
    # make_circle(center=(0.5, 0.0, 0.35), radius=0.15, period=7.0, normal=(0.0, 1.0, 0.0)),  # vertical (x-z)
    # make_circle(center=(0.5, 0.0, 0.30), radius=0.10, period=4.0, normal=(1.0, 0.0, 1.0)),  # tilted
    # # figure-of-eights
    # make_figure8(center=(0.5, 0.0, 0.30), rx=0.14, ry=0.12, period=6.0, normal=(0.0, 0.0, 1.0)),
    # make_figure8(center=(0.5, 0.0, 0.35), rx=0.13, ry=0.12, period=8.0, normal=(0.0, 1.0, 0.0)),
    # # random closed loops (frozen via fixed seeds)
    # make_random_loop(seed=0, n_pts=5, period=6.0),
    # make_random_loop(seed=1, n_pts=6, period=7.0),
    # make_random_loop(seed=2, n_pts=7, period=8.0),
    # make_random_loop(seed=3, n_pts=6, period=5.0),
]
