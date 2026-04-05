"""
Global constants for the HQPINN reproduction.

The values in this module define the benchmark domains, optimization settings,
and default architecture sizes used throughout the codebase. Comments reference
the four case studies of the paper:
- Appendix A.2: DHO
- Sec. 3.1: SEE
- Sec. 3.2: DEE
- Sec. 3.3: TAF
"""

import torch

# Numerical defaults shared by all experiments.
DTYPE = torch.float64
DEVICE = torch.device("cpu")
N_LAYERS = 3
DEFAULT_N_OUTPUTS = 3
# In the paper-style Euler settings, the quantum branch exposes one measured
# output channel per physical variable. DHO is the exception: the target is the
# scalar displacement u(t), even if the circuit internally uses several modes.

GAMMA = 1.4


# ============================================================
# Appendix A.2: damped harmonic oscillator
#   m u''(t) + mu u'(t) + k u(t) = 0,  t in (0, 1]
# ============================================================

DHO_LR = 0.002
DHO_N_EPOCHS = 1801
DHO_PLOT_EVERY = 100
DHO_N_SAMPLES = 200

M = 1.0
MU = 4.0
K = 400.0

# The DHO loss combines the two initial-condition constraints and the ODE
# residual. These coefficients reproduce the weighting used in this codebase.
LAMBDA1 = 1e-1
LAMBDA2 = 1e-4


DHO_NUM_HIDDEN_LAYERS = 2
DHO_HIDDEN_WIDTH = 16


# ============================================================
# Sec. 3.1: smooth Euler equation
#   1D compressible Euler on x in (-1, 1), t in (0, 2)
# ============================================================

SEE_LR = 5e-4
SEE_N_EPOCHS = 20000
SEE_PLOT_EVERY = 1000
SEE_NX_SAMPLES = 200
SEE_NT_SAMPLES = 200

SEE_X_MIN, SEE_X_MAX = -1.0, 1.0
SEE_T_MIN, SEE_T_MAX = 0.0, 2.0

SEE_N_IC = 50
SEE_N_BC = 50
SEE_N_F = 2000


SEE_CC_NUM_HIDDEN_LAYERS = 4
SEE_CC_HIDDEN_WIDTH = 10


# ============================================================
# Sec. 3.2: discontinuous Euler equation
#   1D compressible Euler on x in (0, 1), t in (0, 2)
# ============================================================

DEE_LR = 5e-4
DEE_N_EPOCHS = 20000
DEE_PLOT_EVERY = 1000
DEE_NX_SAMPLES = 200
DEE_NT_SAMPLES = 200

DEE_X_MIN, DEE_X_MAX = 0.0, 1.0
DEE_T_MIN, DEE_T_MAX = 0.0, 2.0

DEE_N_IC = 60
DEE_N_BC = 60
DEE_N_F = 1000

DEE_U = 0.1
DEE_P = 1.0
DEE_RHO_L = 1.4
DEE_RHO_R = 1.0
DEE_X0 = 0.5

DEE_CC_NUM_HIDDEN_LAYERS = 4
DEE_CC_HIDDEN_WIDTH = 10


# ============================================================
# Sec. 3.3: 2D transonic aerofoil flow
# ============================================================

# TAF predicts the primitive variables (rho, u, v, T).
TAF_N_OUTPUTS = 4

#           TAF_Y_MAX   →  X_top
#    -------------------------
#    |                       |
#    |        AILE           |
#    |                       |
#    -------------------------
#           TAF_Y_MIN   →  X_bot

#   TAF_X_MIN           TAF_X_MAX
TAF_R_GAS = 287.0

TAF_X_MIN = -1.0
TAF_X_MAX = 3.5
TAF_Y_MIN = -2.25
TAF_Y_MAX = 2.25

# Inlet primitive state used by the Sec. 3.3 boundary loss.
TAF_RHO_IN = 1.225
TAF_T_IN = 288.15

TAF_DOMAIN_SIDE = 4.5

TAF_CHORD_X0 = 0.0
TAF_CHORD_X1 = 1.0

TAF_N_BOUNDARY = 40
TAF_N_DOMAIN_TOTAL = 4000
TAF_N_DATA_INTERNAL = 400
TAF_N_WALL = 400
TAF_NEAR_AIRFOIL_FRACTION = 0.5
TAF_NEAR_AIRFOIL_PAD_X = 0.25
TAF_NEAR_AIRFOIL_PAD_Y = 0.25
TAF_PDE_NEAR_WEIGHT = 0.5
TAF_PDE_FAR_WEIGHT = 0.5

TAF_LR = 5e-4
TAF_ADAM_STEPS = 40000
TAF_LBFGS_STEPS = 2000
TAF_PLOT_EVERY = 500

TAF_EPSILON_LAMBDA = 0.1
TAF_P_OUT = 0.0

TAF_CC_NUM_HIDDEN_LAYERS = 4
TAF_CC_HIDDEN_WIDTH = 40

# Filenames produced by `generate_aerofoil_training_sets.py`.
TAF_X_IN_FILE = "X_in.npy"
TAF_X_OUT_FILE = "X_out.npy"
TAF_X_TOP_FILE = "X_top.npy"
TAF_X_BOT_FILE = "X_bot.npy"
TAF_X_WALL_FILE = "X_wall.npy"
TAF_X_WALL_NORMALS_FILE = "X_wall_normals.npy"
TAF_X_F_FILE = "X_f.npy"
TAF_X_DATA_INT_FILE = "X_data_int.npy"


def set_dtype(dtype: torch.dtype) -> None:
    """Propagate the runtime-selected dtype to modules importing `HQPINN.config`."""
    global DTYPE
    DTYPE = dtype
