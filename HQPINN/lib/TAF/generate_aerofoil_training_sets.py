"""
generate_aerofoil_training_sets.py

Generate the geometric point clouds used by the Sec. 3.3 TAF benchmark.

Outputs .npy files in TAF/NACA0012/:
  - X_in.npy            (inlet points)            shape (N_in, 2)
  - X_out.npy           (outlet points)           shape (N_out, 2)
  - X_top.npy           (top boundary points)     shape (N_top, 2)
  - X_bot.npy           (bottom boundary points)  shape (N_bot, 2)
  - X_wall.npy          (airfoil surface points)  shape (N_wall, 2)
  - X_wall_normals.npy  (x,y,nx,ny for wall)      shape (N_wall, 4)
  - X_data_int.npy      (internal CFD data points) shape (N_data_int, 2)
  - X_f.npy             (PDE collocation points)  shape (N_pde, 2)

Usage:
  python generate_aerofoil_training_sets.py
Dependencies:
  numpy, (matplotlib optional for quick plot)
"""

import numpy as np
from pathlib import Path

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
            TAF_X_MAX,
            TAF_X_MIN,
            TAF_Y_MAX,
            TAF_Y_MIN,
        )

rng = np.random.default_rng(0)


# ============================================================
# 1) NACA0012 thickness law
# ============================================================
# This is the standard NACA 4-digit half-thickness formula for a symmetric
# NACA0012 aerofoil. The code works in the normalized chord coordinate x/c, so
# the output is the half-thickness y_t(x/c) before any physical scaling.
def naca4_thickness(x):
    x = np.asarray(x)  # Ensure vectorized NumPy operations

    return 0.6 * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))  # √x term (singular slope at LE)
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
    # Work first in the dimensionless chord coordinate x/c in [0, 1].
    x_local = np.linspace(0.0, 1.0, n_points_along_chord)

    # Evaluate the half-thickness law on the normalized chord.
    yt = naca4_thickness(x_local)

    # Scale the normalized coordinate back to the physical chord interval.
    chord_length = chord_end - chord_start

    xu = chord_start + x_local * chord_length
    yu = +yt  # positive offset above camber line (y=0)

    # Lower surface (intrados)
    # Reverse x order so contour closes properly (TE → LE)
    xl = chord_start + x_local[::-1] * chord_length
    yl = -yt[::-1]  # negative offset below camber line

    # Concatenate the two surfaces into one closed wall contour.
    xs = np.concatenate([xu, xl])
    ys = np.concatenate([yu, yl])

    return xs, ys


# ============================================================
# 3) Wall points and outward unit normals
# ============================================================
# These arrays support the impermeability condition used in the TAF loss:
# the velocity must remain tangent to the wall, i.e. u.n = 0.

# Split `TAF_N_WALL` evenly between upper and lower surfaces.
Xw_x, Xw_y = generate_naca0012_surface(
    n_points_along_chord=TAF_N_WALL // 2,
    chord_start=TAF_CHORD_X0,
    chord_end=TAF_CHORD_X1,
)

# Stack wall coordinates into the point cloud used by the boundary loss.
X_wall = np.stack([Xw_x, Xw_y], axis=-1)


def compute_normals(xs, ys):
    """Compute outward unit normals along the closed aerofoil polygon."""

    # Tangents are computed along the polygon parameter, not as partial
    # derivatives with respect to the physical coordinates.
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    tangents = np.stack([dx, dy], axis=-1)

    # Rotate each tangent by -90 degrees to obtain a candidate normal.
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # Normalize to unit length.
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    normals /= norms

    # Flip the orientation if the normal points toward the polygon centroid.
    centroid = np.array([np.mean(xs), np.mean(ys)])

    # Outward normals should align with the centroid-to-boundary vector.
    vecs = np.stack([xs - centroid[0], ys - centroid[1]], axis=-1)

    # A negative dot product indicates an inward normal.
    dotp = np.sum(vecs * normals, axis=1)
    flip_mask = dotp < 0
    normals[flip_mask] *= -1.0

    return normals


# Compute outward normals
Xw_normals = compute_normals(Xw_x, Xw_y)

# Each row stores (x, y, nx, ny), exactly what the wall loss needs.
X_wall_normals = np.concatenate([X_wall, Xw_normals], axis=1)


# ============================================================
# 4) Outer rectangular domain boundaries
# ============================================================
# The Sec. 3.3 computational box is sampled independently on each side so the
# training code can apply inlet, outlet, wall, and periodic terms separately.
# NOTE:
# These are NOT the airfoil surface points.
# The airfoil wall points are generated separately.

# ---- Inlet (left vertical boundary, x = constant = TAF_X_MIN)
# We sample points uniformly along the y-direction.
y_in = np.linspace(TAF_Y_MIN, TAF_Y_MAX, TAF_N_BOUNDARY)

# Create (x, y) pairs:
# x is fixed at TAF_X_MIN
# y varies along the vertical boundary
X_in = np.stack([np.full_like(y_in, TAF_X_MIN), y_in], axis=-1)

# ---- Outlet (right vertical boundary, x = constant = TAF_X_MAX)
# Same y sampling as inlet.
X_out = np.stack([np.full_like(y_in, TAF_X_MAX), y_in], axis=-1)

# ---- Top boundary (horizontal boundary, y = constant = TAF_Y_MAX)
# Here x varies while y is fixed.
x_topbot = np.linspace(TAF_X_MIN, TAF_X_MAX, TAF_N_BOUNDARY)

X_top = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MAX)], axis=-1)

# ---- Bottom boundary (horizontal boundary, y = constant = TAF_Y_MIN)
X_bot = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MIN)], axis=-1)


# ---------------------------------------------------------
# 4) Point-in-polygon test (Ray Casting Algorithm)
# ---------------------------------------------------------
# This function determines whether each point in xy_points
# lies inside a closed polygon defined by (poly_x, poly_y).
#
# In our case:
#   - The polygon is the airfoil surface.
#   - We use this to remove collocation points that fall
#     inside the solid airfoil (non-physical region).
#
# Method:
#   Ray casting algorithm.
#   For each test point:
#       Cast a horizontal ray to +∞.
#       Count how many times it intersects the polygon edges.
#   If the number of intersections is odd → point is inside.
#   If even → point is outside.
def point_in_polygon(xy_points, poly_x, poly_y):

    # Extract x and y coordinates of test points
    x = xy_points[:, 0]
    y = xy_points[:, 1]

    # Number of vertices in polygon
    n = len(poly_x)

    # Boolean mask: True = inside polygon
    inside = np.zeros(len(x), dtype=bool)

    # Loop over each polygon edge
    for i in range(n):
        j = (i + n - 1) % n  # Previous vertex index (wrap-around)

        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        # Check if horizontal ray crosses this edge
        # Condition 1:
        #   The test point's y lies between yi and yj
        # Condition 2:
        #   The intersection x-coordinate is to the right of the point
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi
        )

        # Toggle inside status each time we detect an intersection
        inside ^= intersect

    return inside


# The airfoil polygon is defined by its boundary coordinates
poly_x = Xw_x
poly_y = Xw_y


# ---------------------------------------------------------
# 5) Domain collocation points (uniform sampling + airfoil filtering)
# ---------------------------------------------------------
# Goal:
#   Build a set of points in the *fluid domain* (rectangle minus airfoil interior).
#
# We create:
#   - X_data_int : internal points supervised by CFD data
#   - X_f        : collocation points used for the PDE residual
#
# Strategy:
#   1) Sample points uniformly in the bounding rectangle.
#   2) Remove points that fall inside the airfoil polygon.
#   3) If we don't have enough points left, sample extra points.
#   4) Shuffle and keep the requested total number.

# Oversample because many points will be rejected (inside the airfoil)
oversample_factor = 2.5
n_try = int(TAF_N_DOMAIN_TOTAL * oversample_factor)

# Uniform random points in the rectangular domain
# points shape: (n_try, 2) with columns (x, y)
points = rng.uniform(
    [TAF_X_MIN, TAF_Y_MIN],
    [TAF_X_MAX, TAF_Y_MAX],
    size=(n_try, 2),
)

# Identify points that lie inside the airfoil polygon (solid region)
inside_mask = point_in_polygon(points, poly_x, poly_y)

# Keep only points outside the airfoil → fluid region
points_outside = points[~inside_mask]

# Safety: if too many points were rejected (airfoil is large or oversampling too small),
# sample more points and filter again until we have enough.
while len(points_outside) < TAF_N_DOMAIN_TOTAL:
    needed = TAF_N_DOMAIN_TOTAL - len(points_outside)

    # Sample extra points (slightly more than needed to reduce chance of shortage)
    extra = rng.uniform(
        [TAF_X_MIN, TAF_Y_MIN],
        [TAF_X_MAX, TAF_Y_MAX],
        size=(int(needed * 1.5) + 100, 2),
    )

    # Filter extra points with the same inside-airfoil test
    extra_inside = point_in_polygon(extra, poly_x, poly_y)
    extra_out = extra[~extra_inside]

    # Append extra valid fluid points to the pool
    points_outside = np.vstack([points_outside, extra_out])

# Keep exactly the requested total number of valid fluid points
points_outside = points_outside[:TAF_N_DOMAIN_TOTAL]

# Shuffle points to avoid any spatial ordering bias
perm = rng.permutation(len(points_outside))
points_outside = points_outside[perm]

if TAF_N_DATA_INTERNAL > TAF_N_DOMAIN_TOTAL:
    raise ValueError(
        "TAF_N_DATA_INTERNAL must be <= TAF_N_DOMAIN_TOTAL "
        f"(got {TAF_N_DATA_INTERNAL} > {TAF_N_DOMAIN_TOTAL})."
    )

# Split internal fluid points:
#   - first chunk for CFD-supervised internal data
#   - remaining chunk for PDE residual points
X_data_int = points_outside[:TAF_N_DATA_INTERNAL]
X_f = points_outside[TAF_N_DATA_INTERNAL:]

# ----------------------------
# 6) Save files and print summary
# ----------------------------
script_path = Path(__file__).resolve()
output_dir = script_path.parent / "NACA0012"
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "X_in.npy", X_in)
np.save(output_dir / "X_out.npy", X_out)
np.save(output_dir / "X_top.npy", X_top)
np.save(output_dir / "X_bot.npy", X_bot)
np.save(output_dir / "X_wall.npy", X_wall)
np.save(output_dir / "X_wall_normals.npy", X_wall_normals)
np.save(output_dir / "X_data_int.npy", X_data_int)
np.save(output_dir / "X_f.npy", X_f)

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
