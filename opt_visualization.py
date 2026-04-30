"""
opt_visualization.py
--------------------
Minimizes a 2D function using 5 optimizers:
  adam, newton, sgd, signsgd, lion

For each optimizer the full trajectory (x, y, f(x,y)) is recorded and
plotted as a path on a filled contour of the function.

Usage
-----
  python opt_visualization.py

To swap the target function edit FUNCTION at the bottom of the file.
Available presets: rosenbrock, himmelblau, beale, sphere
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # needed for projection='3d'
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class History:
    name: str
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    fs: List[float] = field(default_factory=list)

    def record(self, x: float, y: float, f: float):
        self.xs.append(x)
        self.ys.append(y)
        self.fs.append(f)

    def final(self):
        return self.xs[-1], self.ys[-1], self.fs[-1]


# ---------------------------------------------------------------------------
# Numerical gradient & Hessian (finite differences — black-box, no autograd)
# ---------------------------------------------------------------------------

def grad_fd(f: Callable, x: float, y: float, h: float = 1e-5) -> Tuple[float, float]:
    """Central-difference gradient."""
    gx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    gy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return gx, gy


def hess_diag_fd(f: Callable, x: float, y: float, h: float = 1e-5) -> Tuple[float, float]:
    """Diagonal of the Hessian via second-order finite differences."""
    f0 = f(x, y)
    hxx = (f(x + h, y) - 2 * f0 + f(x - h, y)) / (h ** 2)
    hyy = (f(x, y + h) - 2 * f0 + f(x, y - h)) / (h ** 2)
    # Clamp to avoid division by zero / wrong-sign steps (same logic as ZOO Newton)
    hxx = max(abs(hxx), 0.1)
    hyy = max(abs(hyy), 0.1)
    return hxx, hyy


# ---------------------------------------------------------------------------
# Optimizers  (plain Python / NumPy — operate on scalar (x, y) coords)
# ---------------------------------------------------------------------------

def run_adam(f, x0, y0, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, steps=500):
    history = History("Adam")
    x, y = x0, y0
    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0
    for t in range(1, steps + 1):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        vx = beta2 * vx + (1 - beta2) * gx ** 2
        vy = beta2 * vy + (1 - beta2) * gy ** 2
        corr = np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        x -= lr * corr * mx / (np.sqrt(vx) + eps)
        y -= lr * corr * my / (np.sqrt(vy) + eps)
    history.record(x, y, f(x, y))
    return history


def run_newton(f, x0, y0, lr=1.0, steps=500):
    history = History("Newton")
    x, y = x0, y0
    for _ in range(steps):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        hxx, hyy = hess_diag_fd(f, x, y)
        x -= lr * gx / hxx
        y -= lr * gy / hyy
    history.record(x, y, f(x, y))
    return history


def run_sgd(f, x0, y0, lr=0.01, steps=500):
    """Vanilla gradient descent — no momentum."""
    history = History("SGD")
    x, y = x0, y0
    for _ in range(steps):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        x -= lr * gx
        y -= lr * gy
    history.record(x, y, f(x, y))
    return history


def run_sgdsign(f, x0, y0, lr=0.01, steps=500):
    """SGDSign — step is lr * sign(g), no momentum."""
    history = History("SGDSign")
    x, y = x0, y0
    for _ in range(steps):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        x -= lr * np.sign(gx)
        y -= lr * np.sign(gy)
    history.record(x, y, f(x, y))
    return history


def run_signum(f, x0, y0, lr=0.01, beta1=0.9, steps=500):
    """Signum — step is lr * sign(m), where m is an EMA of gradients."""
    history = History("Signum")
    x, y = x0, y0
    mx, my = 0.0, 0.0
    for _ in range(steps):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        x -= lr * np.sign(mx)
        y -= lr * np.sign(my)
    history.record(x, y, f(x, y))
    return history


def run_lion(f, x0, y0, lr=0.001, beta1=0.9, beta2=0.99, steps=500):
    """Lion: EvoLved Sign Momentum — arxiv.org/abs/2302.06675"""
    history = History("Lion")
    x, y = x0, y0
    mx, my = 0.0, 0.0
    for _ in range(steps):
        history.record(x, y, f(x, y))
        gx, gy = grad_fd(f, x, y)
        # 1. update direction: sign of interpolated momentum
        ux = np.sign(beta1 * mx + (1 - beta1) * gx)
        uy = np.sign(beta1 * my + (1 - beta1) * gy)
        x -= lr * ux
        y -= lr * uy
        # 2. momentum update AFTER the step
        mx = beta2 * mx + (1 - beta2) * gx
        my = beta2 * my + (1 - beta2) * gy
    history.record(x, y, f(x, y))
    return history


# ---------------------------------------------------------------------------
# Preset 2-D functions
# ---------------------------------------------------------------------------

def rosenbrock(x, y, a=1, b=100):
    """Global minimum at (a, a²) = (1, 1)"""
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def himmelblau(x, y):
    """Four equal minima at (~3,2), (~-2.8,3.1), (~-3.8,-3.3), (~3.6,-1.8)"""
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def beale(x, y):
    """Global minimum at (3, 0.5)"""
    return ((1.5   - x + x * y)       ** 2 +
            (2.25  - x + x * y ** 2)  ** 2 +
            (2.625 - x + x * y ** 3)  ** 2)


def sphere(x, y):
    """Global minimum at (0, 0)"""
    return x ** 2 + y ** 2


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

COLORS = {
    "Adam":    "#e41a1c",
    "Newton":  "#377eb8",
    "SGD":     "#4daf4a",
    "SGDSign": "#ff7f00",
    "Signum":  "#FFD700",
    "Lion":    "#984ea3",
}


# Set Times New Roman as the global default for all text in every figure
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"   # matching math font

_TNR = {"fontfamily": "Times New Roman"}


def _apply_tnr(ax):
    """Apply Times New Roman to all text elements of a 2-D axes."""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("Times New Roman")
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontfamily="Times New Roman")


def _apply_tnr_3d(ax):
    """Apply Times New Roman to all text elements of a 3-D axes."""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()
                 + ax.get_zticklabels()):
        item.set_fontfamily("Times New Roman")
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontfamily="Times New Roman")


def plot_contour_2d(histories: List[History], f: Callable,
                    xlim: Tuple, ylim: Tuple,
                    title: str = "2-D contour",
                    resolution: int = 200,
                    true_min: List[Tuple] = None):
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    levels = np.unique(np.percentile(Z, np.linspace(0, 95, 30)))

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.75)
    cbar = plt.colorbar(cf, ax=ax, label="f(x, y)")
    cbar.ax.yaxis.label.set_fontfamily("Times New Roman")
    plt.setp(cbar.ax.get_yticklabels(), fontfamily="Times New Roman")
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.4)

    for h in histories:
        color = COLORS.get(h.name, "black")
        ax.plot(h.xs, h.ys, "-o", color=color, label=h.name,
                markersize=2, linewidth=1.5, alpha=0.85)
        ax.plot(h.xs[0], h.ys[0], "o", color=color, markersize=7)

    # ---- Final stars with overlap jitter ----
    # Compute a nudge radius in data units (~1.5% of axis span)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    nudge = 0.018 * max(x_span, y_span)

    finals = [(h.xs[-1], h.ys[-1], COLORS.get(h.name, "black")) for h in histories]
    offsets = [[0.0, 0.0] for _ in finals]
    # Jitter directions: cycle through 8 compass offsets
    dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            dx = finals[i][0] - finals[j][0]
            dy = finals[i][1] - finals[j][1]
            if (dx**2 + dy**2) ** 0.5 < nudge * 1.5:
                di = dirs[i % len(dirs)]
                dj = dirs[(i + 4) % len(dirs)]  # opposite direction
                offsets[i][0] += di[0] * nudge
                offsets[i][1] += di[1] * nudge
                offsets[j][0] += dj[0] * nudge
                offsets[j][1] += dj[1] * nudge

    for (fx, fy, color), (ox, oy) in zip(finals, offsets):
        ax.plot(fx + ox, fy + oy, "*", color=color, markersize=16,
                markeredgecolor="black", markeredgewidth=0.9,
                zorder=11, clip_on=False)

    if true_min:
        for i, (tx, ty) in enumerate(true_min):
            label = "True min" if i == 0 else "_nolegend_"
            ax.plot(tx, ty, "x", color="#00CED1", markersize=13,
                    markeredgewidth=3.0, zorder=10, label=label)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x", **_TNR); ax.set_ylabel("y", **_TNR)
    ax.set_title(f"{title} — 2-D contour", **_TNR)
    ax.legend(loc="upper right", fontsize=8, prop={"family": "Times New Roman"})
    _apply_tnr(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_contour2d.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


def plot_surface_3d(histories: List[History], f: Callable,
                    xlim: Tuple, ylim: Tuple,
                    title: str = "3-D surface",
                    resolution: int = 100,
                    true_min: List[Tuple] = None):
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    z_ceil = float(np.percentile(Z, 97))
    z_floor = float(np.min(Z_plot := np.clip(Z, None, z_ceil)))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Semi-transparent surface — low alpha so paths are clearly visible
    surf = ax.plot_surface(X, Y, Z_plot, cmap="viridis", alpha=0.30,
                           linewidth=0, antialiased=True, rcount=80, ccount=80)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.12, label="f(x, y)")
    cbar.ax.yaxis.label.set_fontfamily("Times New Roman")
    plt.setp(cbar.ax.get_yticklabels(), fontfamily="Times New Roman")

    for h in histories:
        color = COLORS.get(h.name, "black")
        fpath_clipped = np.clip(np.array(h.fs), None, z_ceil)
        # Line with a dot at every recorded point
        ax.plot(h.xs, h.ys, fpath_clipped, "-o", color=color, label=h.name,
                markersize=2, linewidth=1.5, alpha=0.9, zorder=5)
        # Start marker
        ax.scatter([h.xs[0]], [h.ys[0]], [fpath_clipped[0]],
                   color=color, s=70, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=6)
        # Final point: large star with black outline
        ax.scatter([h.xs[-1]], [h.ys[-1]], [fpath_clipped[-1]],
                   color=color, s=220, marker="*",
                   edgecolors="black", linewidths=0.9, zorder=7)

    if true_min:
        for i, (tx, ty) in enumerate(true_min):
            tz = np.clip(float(f(tx, ty)), None, z_ceil)
            lbl = "True min" if i == 0 else "_nolegend_"
            ax.scatter([tx], [ty], [tz], color="#00CED1", s=200, marker="x",
                       linewidths=3.0, zorder=10, label=lbl)
            # Vertical dashed line to floor so position is unambiguous
            ax.plot([tx, tx], [ty, ty], [z_floor, tz],
                    color="#00CED1", linewidth=1.2, linestyle=":", alpha=0.85)

    ax.set_xlabel("x", **_TNR, labelpad=8)
    ax.set_ylabel("y", **_TNR, labelpad=8)
    ax.set_zlabel("f(x, y)", **_TNR, labelpad=8)
    ax.set_title(f"{title} — 3-D surface", **_TNR, pad=12)
    ax.legend(loc="upper left", fontsize=8,
              prop={"family": "Times New Roman"},
              framealpha=0.7, borderpad=0.6)
    ax.view_init(elev=32, azim=-55)
    # Reduce pane clutter for readability
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgrey")
    ax.yaxis.pane.set_edgecolor("lightgrey")
    ax.zaxis.pane.set_edgecolor("lightgrey")
    _apply_tnr_3d(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_surface3d.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


def plot_convergence(histories: List[History],
                     title: str = "Convergence",
                     f_min: float = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in histories:
        color = COLORS.get(h.name, "black")
        ax.semilogy(h.fs, color=color, label=h.name, linewidth=1.5)
    if f_min is not None:
        # Shift tiny negative/zero minima to a small positive value for log scale
        f_min_plot = max(f_min, 1e-10)
        ax.axhline(f_min_plot, color="#00CED1", linewidth=1.4,
                   linestyle="--", label=f"Global min  f={f_min:.4g}", zorder=5)
    ax.set_xlabel("Step", **_TNR)
    ax.set_ylabel("f(x, y)  [log scale]", **_TNR)
    ax.set_title(f"{title} — Convergence", **_TNR)
    ax.legend(fontsize=8, prop={"family": "Times New Roman"})
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    _apply_tnr(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_convergence.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


def print_summary(histories: List[History]):
    print(f"\n{'Optimizer':<12} {'Final x':>10} {'Final y':>10} {'Final f':>14}")
    print("-" * 50)
    for h in histories:
        fx, fy, ff = h.final()
        print(f"{h.name:<12} {fx:>10.5f} {fy:>10.5f} {ff:>14.6e}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Choose function & starting point ----
    FUNCTION      = rosenbrock   # rosenbrock | himmelblau | beale | sphere
    START         = (-2.0, -1.0)  # (x0, y0)
    STEPS         = 2000
    PLOT_XLIM     = (-2.5, 2.5)
    PLOT_YLIM     = (-1.0, 3.5)
    TITLE         = "Rosenbrock"
    # Known global minimum (or minima) — used to mark the target on the plots.
    # rosenbrock: [(1, 1)]  |  himmelblau: 4 minima  |  beale: [(3, 0.5)]  |  sphere: [(0, 0)]
    TRUE_MIN      = [(1.0, 1.0)]

    # ---- Per-solver hyperparameters ----
    # Tune these if switching functions
    ADAM_LR       = 0.5
    NEWTON_LR     = 0.75
    SGD_LR        = 0.0001
    SGDSIGN_LR    = 0.01
    SIGNUM_LR     = 0.01
    LION_LR       = 0.0018

    x0, y0 = START

    print(f"Function : {FUNCTION.__name__}")
    print(f"Start    : {START}")
    print(f"Steps    : {STEPS}\n")

    histories = [
        run_adam    (FUNCTION, x0, y0, lr=ADAM_LR,     steps=STEPS),
        run_newton  (FUNCTION, x0, y0, lr=NEWTON_LR,   steps=STEPS),
        run_sgd     (FUNCTION, x0, y0, lr=SGD_LR,      steps=STEPS),
        run_sgdsign (FUNCTION, x0, y0, lr=SGDSIGN_LR,  steps=STEPS),
        run_signum  (FUNCTION, x0, y0, lr=SIGNUM_LR,   steps=STEPS),
        run_lion    (FUNCTION, x0, y0, lr=LION_LR,     steps=STEPS),
    ]

    print_summary(histories)
    plot_contour_2d(histories, FUNCTION, PLOT_XLIM, PLOT_YLIM,
                    title=TITLE, true_min=TRUE_MIN)
    plot_surface_3d(histories, FUNCTION, PLOT_XLIM, PLOT_YLIM,
                    title=TITLE, true_min=TRUE_MIN)
    f_min = min(FUNCTION(tx, ty) for tx, ty in TRUE_MIN) if TRUE_MIN else None
    plot_convergence(histories, title=TITLE, f_min=f_min)
