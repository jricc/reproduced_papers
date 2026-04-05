"""
generate_aerofoil_training_sets.py

Generate the geometric point clouds used by the Sec. 3.3 TAF benchmark.

Outputs .npy files in TAF/NACA0012/:
  - X_in.npy            (inlet points)             shape (N_in, 2)
  - X_out.npy           (outlet points)            shape (N_out, 2)
  - X_top.npy           (top boundary points)      shape (N_top, 2)
  - X_bot.npy           (bottom boundary points)   shape (N_bot, 2)
  - X_wall.npy          (airfoil surface points)   shape (N_wall, 2)
  - X_wall_normals.npy  (x,y,nx,ny for wall)       shape (N_wall, 4)
  - X_data_int.npy      (internal CFD data points) shape (N_data_int, 2)
  - X_f.npy             (PDE collocation points)   shape (N_pde, 2)

Usage:
  python generate_aerofoil_training_sets.py

Dependencies:
  numpy, (matplotlib optional for quick plot)
"""

from pathlib import Path

import numpy as np

# ============================================================
# 0) Import benchmark geometry and sampling parameters
# ============================================================
try:
    # Package execution: python -m HQPINN.lib.TAF.generate_aerofoil_training_sets
    from ...config import (
        TAF_CHORD_X0,
        TAF_CHORD_X1,
        TAF_N_BOUNDARY,
        TAF_N_DATA_INTERNAL,
        TAF_N_DOMAIN_TOTAL,
        TAF_N_WALL,
        TAF_NEAR_AIRFOIL_FRACTION,
        TAF_NEAR_AIRFOIL_PAD_X,
        TAF_NEAR_AIRFOIL_PAD_Y,
        TAF_X_MAX,
        TAF_X_MIN,
        TAF_Y_MAX,
        TAF_Y_MIN,
    )
except ImportError:
    try:
        # Direct script execution: ensure HQPINN is importable
        import sys

        pkg_root = Path(__file__).resolve().parents[2]
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        from config import (
            TAF_CHORD_X0,
            TAF_CHORD_X1,
            TAF_N_BOUNDARY,
            TAF_N_DATA_INTERNAL,
            TAF_N_DOMAIN_TOTAL,
            TAF_N_WALL,
            TAF_NEAR_AIRFOIL_FRACTION,
            TAF_NEAR_AIRFOIL_PAD_X,
            TAF_NEAR_AIRFOIL_PAD_Y,
            TAF_X_MAX,
            TAF_X_MIN,
            TAF_Y_MAX,
            TAF_Y_MIN,
        )
    except ImportError:
        # Direct script execution from repo root
        from HQPINN.config import (
            TAF_CHORD_X0,
            TAF_CHORD_X1,
            TAF_N_BOUNDARY,
            TAF_N_DATA_INTERNAL,
            TAF_N_DOMAIN_TOTAL,
            TAF_N_WALL,
            TAF_NEAR_AIRFOIL_FRACTION,
            TAF_NEAR_AIRFOIL_PAD_X,
            TAF_NEAR_AIRFOIL_PAD_Y,
            TAF_X_MAX,
            TAF_X_MIN,
            TAF_Y_MAX,
            TAF_Y_MIN,
        )

FULL_DOMAIN_LOW = np.array([TAF_X_MIN, TAF_Y_MIN], dtype=float)
FULL_DOMAIN_HIGH = np.array([TAF_X_MAX, TAF_Y_MAX], dtype=float)


# ============================================================
# 1) NACA0012 thickness law
# ============================================================
# This is the standard NACA 4-digit half-thickness formula for a symmetric
# NACA0012 aerofoil. The code works in the normalized chord coordinate x/c, so
# the output is the half-thickness y_t(x/c) before any physical scaling.
def naca4_thickness(x):
    x = np.asarray(x)

    return 0.6 * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


# ============================================================
# 2) Closed NACA0012 wall polygon
# ============================================================
# The upper surface is traced from leading edge to trailing edge and the lower
# surface is traced back in reverse order. The resulting closed polygon is then
# reused for wall normals and for excluding interior collocation points.
def generate_naca0012_surface(
    n_points_along_chord=200, chord_start=None, chord_end=None
):
    if chord_start is None:
        chord_start = TAF_CHORD_X0
    if chord_end is None:
        chord_end = TAF_CHORD_X1

    x_local = np.linspace(0.0, 1.0, n_points_along_chord)
    yt = naca4_thickness(x_local)
    chord_length = chord_end - chord_start

    xu = chord_start + x_local * chord_length
    yu = +yt

    xl = chord_start + x_local[::-1] * chord_length
    yl = -yt[::-1]

    xs = np.concatenate([xu, xl])
    ys = np.concatenate([yu, yl])
    return xs, ys


# ============================================================
# 3) Wall points and outward unit normals
# ============================================================
# These arrays support the impermeability condition used in the TAF loss:
# the velocity must remain tangent to the wall, i.e. u.n = 0.
Xw_x, Xw_y = generate_naca0012_surface(
    n_points_along_chord=TAF_N_WALL // 2,
    chord_start=TAF_CHORD_X0,
    chord_end=TAF_CHORD_X1,
)
X_wall = np.stack([Xw_x, Xw_y], axis=-1)


def compute_normals(xs, ys):
    """Compute outward unit normals along the closed aerofoil polygon."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    tangents = np.stack([dx, dy], axis=-1)

    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms

    centroid = np.array([np.mean(xs), np.mean(ys)])
    vecs = np.stack([xs - centroid[0], ys - centroid[1]], axis=-1)
    dotp = np.sum(vecs * normals, axis=1)
    flip_mask = dotp < 0
    normals[flip_mask] *= -1.0
    return normals


Xw_normals = compute_normals(Xw_x, Xw_y)
X_wall_normals = np.concatenate([X_wall, Xw_normals], axis=1)


# ============================================================
# 4) Outer rectangular domain boundaries
# ============================================================
# The Sec. 3.3 computational box is sampled independently on each side so the
# training code can apply inlet, outlet, wall, and periodic terms separately.
y_in = np.linspace(TAF_Y_MIN, TAF_Y_MAX, TAF_N_BOUNDARY)
X_in = np.stack([np.full_like(y_in, TAF_X_MIN), y_in], axis=-1)
X_out = np.stack([np.full_like(y_in, TAF_X_MAX), y_in], axis=-1)

x_topbot = np.linspace(TAF_X_MIN, TAF_X_MAX, TAF_N_BOUNDARY)
X_top = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MAX)], axis=-1)
X_bot = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MIN)], axis=-1)


# ============================================================
# 5) Point-in-polygon test
# ============================================================
def point_in_polygon(xy_points, poly_x, poly_y):
    """Ray-casting test for whether each point falls inside the airfoil."""
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    n = len(poly_x)
    inside = np.zeros(len(x), dtype=bool)

    for i in range(n):
        j = (i + n - 1) % n
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi
        )
        inside ^= intersect

    return inside


poly_x = Xw_x
poly_y = Xw_y


def clip_sampling_box(low, high):
    """Clip a proposed sampling box so it remains inside the full TAF domain."""
    low = np.maximum(np.asarray(low, dtype=float), FULL_DOMAIN_LOW)
    high = np.minimum(np.asarray(high, dtype=float), FULL_DOMAIN_HIGH)
    if np.any(high <= low):
        raise ValueError(
            "Invalid sampling box after clipping: "
            f"low={low.tolist()} high={high.tolist()}"
        )
    return low, high


def points_in_box(points, low, high):
    """Return a boolean mask for points inside an axis-aligned box."""
    return (
        (points[:, 0] >= low[0])
        & (points[:, 0] <= high[0])
        & (points[:, 1] >= low[1])
        & (points[:, 1] <= high[1])
    )


def compute_near_airfoil_box(wall_points, pad_x, pad_y):
    """Build a padded local box around the airfoil for denser collocation."""
    if pad_x < 0.0 or pad_y < 0.0:
        raise ValueError(
            "Near-airfoil padding must be non-negative, "
            f"got pad_x={pad_x} and pad_y={pad_y}."
        )
    wall_min = wall_points.min(axis=0)
    wall_max = wall_points.max(axis=0)
    return clip_sampling_box(
        wall_min - np.array([pad_x, pad_y], dtype=float),
        wall_max + np.array([pad_x, pad_y], dtype=float),
    )


def sample_valid_fluid_points(
    n_points,
    low,
    high,
    poly_x,
    poly_y,
    rng,
    oversample_factor=1.5,
):
    """Sample points in a box, rejecting those that fall inside the airfoil."""
    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    sampled_chunks = []
    n_collected = 0

    while n_collected < n_points:
        remaining = n_points - n_collected
        n_try = max(int(np.ceil(remaining * oversample_factor)), remaining + 32)
        candidates = rng.uniform(low, high, size=(n_try, 2))
        outside = candidates[~point_in_polygon(candidates, poly_x, poly_y)]

        if len(outside) == 0:
            oversample_factor = max(oversample_factor * 1.5, 2.0)
            continue

        take = outside[:remaining]
        sampled_chunks.append(take)
        n_collected += len(take)

    return np.vstack(sampled_chunks)


def sample_domain_points(
    total_points,
    wall_points,
    poly_x,
    poly_y,
    near_airfoil_fraction,
    near_airfoil_pad_x,
    near_airfoil_pad_y,
    rng,
):
    """
    Sample a mixed collocation cloud with global coverage plus a near-airfoil bias.

    The global component preserves coverage of the whole fluid box, while the
    local component increases resolution around the profile where gradients are
    most likely to matter.
    """
    if not 0.0 <= near_airfoil_fraction <= 1.0:
        raise ValueError(
            "TAF_NEAR_AIRFOIL_FRACTION must be in [0, 1], "
            f"got {near_airfoil_fraction}."
        )

    near_box_low, near_box_high = compute_near_airfoil_box(
        wall_points=wall_points,
        pad_x=near_airfoil_pad_x,
        pad_y=near_airfoil_pad_y,
    )

    n_near = int(round(total_points * near_airfoil_fraction))
    n_global = total_points - n_near

    global_points = sample_valid_fluid_points(
        n_points=n_global,
        low=FULL_DOMAIN_LOW,
        high=FULL_DOMAIN_HIGH,
        poly_x=poly_x,
        poly_y=poly_y,
        rng=rng,
    )
    near_points = sample_valid_fluid_points(
        n_points=n_near,
        low=near_box_low,
        high=near_box_high,
        poly_x=poly_x,
        poly_y=poly_y,
        rng=rng,
    )

    points_outside = np.vstack([global_points, near_points])
    perm = rng.permutation(len(points_outside))
    points_outside = points_outside[perm]

    near_mask = points_in_box(points_outside, near_box_low, near_box_high)
    sampling_info = {
        "near_box_low": near_box_low,
        "near_box_high": near_box_high,
        "n_global_target": n_global,
        "n_near_target": n_near,
        "actual_near_count": int(np.count_nonzero(near_mask)),
    }
    return points_outside, sampling_info


def build_training_sets(rng=None):
    """Assemble the full geometry, boundary, and interior TAF point clouds."""
    if rng is None:
        rng = np.random.default_rng(0)

    if TAF_N_DATA_INTERNAL > TAF_N_DOMAIN_TOTAL:
        raise ValueError(
            "TAF_N_DATA_INTERNAL must be <= TAF_N_DOMAIN_TOTAL "
            f"(got {TAF_N_DATA_INTERNAL} > {TAF_N_DOMAIN_TOTAL})."
        )

    points_outside, sampling_info = sample_domain_points(
        total_points=TAF_N_DOMAIN_TOTAL,
        wall_points=X_wall,
        poly_x=poly_x,
        poly_y=poly_y,
        near_airfoil_fraction=TAF_NEAR_AIRFOIL_FRACTION,
        near_airfoil_pad_x=TAF_NEAR_AIRFOIL_PAD_X,
        near_airfoil_pad_y=TAF_NEAR_AIRFOIL_PAD_Y,
        rng=rng,
    )

    training_sets = {
        "X_in": X_in,
        "X_out": X_out,
        "X_top": X_top,
        "X_bot": X_bot,
        "X_wall": X_wall,
        "X_wall_normals": X_wall_normals,
        "X_data_int": points_outside[:TAF_N_DATA_INTERNAL],
        "X_f": points_outside[TAF_N_DATA_INTERNAL:],
    }
    return training_sets, sampling_info


def save_training_sets(training_sets, output_dir):
    """Persist the generated TAF point clouds to the canonical NACA0012 folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in training_sets.items():
        np.save(output_dir / f"{name}.npy", array)


def main():
    training_sets, sampling_info = build_training_sets()
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / "NACA0012"
    save_training_sets(training_sets, output_dir)

    print("Saved: X_in, X_out, X_top, X_bot, X_wall, X_wall_normals, X_data_int, X_f")
    print("Saved to directory:", output_dir)
    print(
        "Domain bounds: x_min,x_max =",
        TAF_X_MIN,
        TAF_X_MAX,
        "  y_min,y_max =",
        TAF_Y_MIN,
        TAF_Y_MAX,
    )
    print("Chord placed on [0,1] (LE at x=0, TE at x=1)")
    print(
        "Near-airfoil box: "
        f"x in [{sampling_info['near_box_low'][0]:.3f}, {sampling_info['near_box_high'][0]:.3f}], "
        f"y in [{sampling_info['near_box_low'][1]:.3f}, {sampling_info['near_box_high'][1]:.3f}]"
    )
    print(
        "Sampling mix: "
        f"{sampling_info['n_near_target']} near-airfoil + "
        f"{sampling_info['n_global_target']} global "
        f"(actual inside local box: {sampling_info['actual_near_count']}/{TAF_N_DOMAIN_TOTAL})"
    )


if __name__ == "__main__":
    main()
