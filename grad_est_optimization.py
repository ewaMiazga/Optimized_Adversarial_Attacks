"""
grad_est_optimization.py
------------------------
Compares optimiser trajectories when using:
  • Exact (analytical) gradients
  • Finite-difference (FD) gradient estimates

Five optimisers are run under both gradient regimes:
  adam, newton, sgd-momentum, signsgd, lion

Three output PDFs are produced (same style as opt_visualization.py):
  *_contour2d.pdf   — 2-D filled contour + paths
  *_surface3d.pdf   — 3-D surface + paths
  *_convergence.pdf — semi-log convergence + global-minimum line

Exact vs FD is distinguished by line style:
  solid  ─── exact gradient
  dashed - - FD gradient

Usage
-----
  python grad_est_optimization.py

To change the test function edit FUNCTION, TRUE_MIN, START, PLOT_XLIM/YLIM
at the bottom of the file.
Available presets: rosenbrock | himmelblau | beale | sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

_TNR = {"fontfamily": "Times New Roman"}

COLORS = {
    "Adam":    "#e41a1c",
    "Newton":  "#377eb8",
    "SGD":     "#4daf4a",
    "SGDSign": "#ff7f00",
    "Signum":  "#FFD700",
    "Lion":    "#984ea3",
}

EXACT_LS = "-"    # solid   → exact gradient
FD_LS    = "--"   # dashed  → finite-difference gradient


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class History:
    name: str
    grad_type: str          # "exact" or "fd"
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    fs: List[float] = field(default_factory=list)

    def record(self, x, y, f):
        self.xs.append(float(x))
        self.ys.append(float(y))
        self.fs.append(float(f))

    def final(self):
        return self.xs[-1], self.ys[-1], self.fs[-1]

    @property
    def label(self):
        return f"{self.name} ({'exact' if self.grad_type == 'exact' else 'FD'})"

    @property
    def ls(self):
        return EXACT_LS if self.grad_type == "exact" else FD_LS


# ---------------------------------------------------------------------------
# Preset 2-D functions with analytical gradients
# ---------------------------------------------------------------------------

def rosenbrock(x, y, a=1, b=100):
    """Global minimum at (1, 1)."""
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def rosenbrock_grad(x, y, a=1, b=100):
    gx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    gy = 2 * b * (y - x ** 2)
    return gx, gy

def rosenbrock_hess_diag(x, y, a=1, b=100):
    hxx = 2 - 4 * b * y + 12 * b * x ** 2
    hyy = 2 * b
    return hxx, hyy


def himmelblau(x, y):
    """Four equal minima."""
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def himmelblau_grad(x, y):
    gx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
    gy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
    return gx, gy

def himmelblau_hess_diag(x, y):
    hxx = 12 * x ** 2 + 4 * y - 44 + 2
    hyy = 2 + 12 * y ** 2 + 4 * x - 28
    return hxx, hyy


def beale(x, y):
    """Global minimum at (3, 0.5)."""
    t1 = 1.5   - x + x * y
    t2 = 2.25  - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    return t1 ** 2 + t2 ** 2 + t3 ** 2

def beale_grad(x, y):
    t1 = 1.5   - x + x * y
    t2 = 2.25  - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    gx = (2 * t1 * (y - 1) +
          2 * t2 * (y ** 2 - 1) +
          2 * t3 * (y ** 3 - 1))
    gy = (2 * t1 * x +
          2 * t2 * 2 * x * y +
          2 * t3 * 3 * x * y ** 2)
    return gx, gy

def beale_hess_diag(x, y):
    t1 = 1.5   - x + x * y
    t2 = 2.25  - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    hxx = (2 * (y - 1) ** 2 +
           2 * (y ** 2 - 1) ** 2 +
           2 * (y ** 3 - 1) ** 2)
    hyy = (2 * x ** 2 +
           2 * (2 * x * y) ** 2 + 2 * t2 * 2 * x +
           2 * (3 * x * y ** 2) ** 2 + 2 * t3 * 6 * x * y)
    return hxx, hyy


def sphere(x, y):
    """Global minimum at (0, 0)."""
    return x ** 2 + y ** 2

def sphere_grad(x, y):
    return 2 * x, 2 * y

def sphere_hess_diag(x, y):
    return 2.0, 2.0


# Registry: function → (grad_fn, hess_diag_fn)
GRAD_REGISTRY = {
    rosenbrock: (rosenbrock_grad, rosenbrock_hess_diag),
    himmelblau: (himmelblau_grad, himmelblau_hess_diag),
    beale:      (beale_grad,      beale_hess_diag),
    sphere:     (sphere_grad,     sphere_hess_diag),
}


# ---------------------------------------------------------------------------
# Finite-difference gradient & Hessian diagonal
# ---------------------------------------------------------------------------
def grad_fd(f, x, y, h=1e-5):
    gx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    gy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return gx, gy

def hess_diag_fd(f, x, y, h=1e-5):
    f0 = f(x, y)
    hxx = max(abs((f(x + h, y) - 2 * f0 + f(x - h, y)) / h ** 2), 0.1)
    hyy = max(abs((f(x, y + h) - 2 * f0 + f(x, y - h)) / h ** 2), 0.1)
    return hxx, hyy


def _safe_hess(hxx, hyy):
    return max(abs(hxx), 0.1), max(abs(hyy), 0.1)


# ---------------------------------------------------------------------------
# Generic optimisers — accept a grad_fn and hess_fn callable
# ---------------------------------------------------------------------------

def run_adam(f, grad_fn, x0, y0, name="Adam", grad_type="exact",
             lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, steps=500):
    h = History(name, grad_type)
    x, y = x0, y0
    mx = my = vx = vy = 0.0
    for t in range(1, steps + 1):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        vx = beta2 * vx + (1 - beta2) * gx ** 2
        vy = beta2 * vy + (1 - beta2) * gy ** 2
        corr = np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        x -= lr * corr * mx / (np.sqrt(vx) + eps)
        y -= lr * corr * my / (np.sqrt(vy) + eps)
    h.record(x, y, f(x, y))
    return h


def run_newton(f, grad_fn, hess_fn, x0, y0, name="Newton", grad_type="exact",
               lr=1.0, steps=500):
    h = History(name, grad_type)
    x, y = x0, y0
    for _ in range(steps):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        hxx, hyy = _safe_hess(*hess_fn(x, y))
        x -= lr * gx / hxx
        y -= lr * gy / hyy
    h.record(x, y, f(x, y))
    return h


def run_sgd(f, grad_fn, x0, y0, name="SGD", grad_type="exact",
            lr=0.01, steps=500):
    """Vanilla gradient descent — no momentum."""
    h = History(name, grad_type)
    x, y = x0, y0
    for _ in range(steps):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        x -= lr * gx
        y -= lr * gy
    h.record(x, y, f(x, y))
    return h

def run_signsgd(f, grad_fn, x0, y0, name="SGDSign", grad_type="exact",
                lr=0.01, steps=500):
    """SGDSign — step is lr * sign(g), no momentum."""
    h = History(name, grad_type)
    x, y = x0, y0
    for _ in range(steps):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        x -= lr * np.sign(gx)
        y -= lr * np.sign(gy)
    h.record(x, y, f(x, y))
    return h


def run_signnum(f, grad_fn, x0, y0, name="Signum", grad_type="exact",
                lr=0.01, beta1=0.9, steps=500):
    """Signum — step is lr * sign(m), where m is an EMA of gradients."""
    h = History(name, grad_type)
    x, y = x0, y0
    mx = my = 0.0
    for _ in range(steps):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        x -= lr * np.sign(mx)
        y -= lr * np.sign(my)
    h.record(x, y, f(x, y))
    return h


def run_lion(f, grad_fn, x0, y0, name="Lion", grad_type="exact",
             lr=0.001, beta1=0.9, beta2=0.99, steps=500):
    h = History(name, grad_type)
    x, y = x0, y0
    mx = my = 0.0
    for _ in range(steps):
        h.record(x, y, f(x, y))
        gx, gy = grad_fn(x, y)
        ux = np.sign(beta1 * mx + (1 - beta1) * gx)
        uy = np.sign(beta1 * my + (1 - beta1) * gy)
        x -= lr * ux
        y -= lr * uy
        mx = beta2 * mx + (1 - beta2) * gx
        my = beta2 * my + (1 - beta2) * gy
    h.record(x, y, f(x, y))
    return h


# ---------------------------------------------------------------------------
# Helper: build exact and FD gradient/Hessian callables for a function
# ---------------------------------------------------------------------------
def make_grad_fns(f):
    """Return (exact_grad, exact_hess, fd_grad, fd_hess) for f."""
    exact_grad, exact_hess = GRAD_REGISTRY[f]
    fd_grad  = lambda x, y: grad_fd(f, x, y)
    fd_hess  = lambda x, y: hess_diag_fd(f, x, y)
    return exact_grad, exact_hess, fd_grad, fd_hess


# ---------------------------------------------------------------------------
# Run all optimisers under both gradient regimes
# ---------------------------------------------------------------------------
def run_all(f, x0, y0, steps,
            adam_lr, newton_lr, sgd_lr,
            sgdsign_lr, signum_lr, lion_lr):
    eg, eh, fg, fh = make_grad_fns(f)
    histories = []
    for gtype, gfn, hfn in [("exact", eg, eh), ("fd", fg, fh)]:
        histories += [
            run_adam    (f, gfn,      x0, y0, "Adam",    gtype, lr=adam_lr,    steps=steps),
            run_newton  (f, gfn, hfn, x0, y0, "Newton",  gtype, lr=newton_lr,  steps=steps),
            run_sgd     (f, gfn,      x0, y0, "SGD",     gtype, lr=sgd_lr,     steps=steps),
            run_signsgd (f, gfn,      x0, y0, "SGDSign", gtype, lr=sgdsign_lr, steps=steps),
            run_signnum (f, gfn,      x0, y0, "Signum",  gtype, lr=signum_lr,  steps=steps),
            run_lion    (f, gfn,      x0, y0, "Lion",    gtype, lr=lion_lr,    steps=steps),
        ]
    return histories


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _apply_tnr(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("Times New Roman")
    leg = ax.get_legend()
    if leg:
        plt.setp(leg.get_texts(), fontfamily="Times New Roman")


def _apply_tnr_3d(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()
                 + ax.get_zticklabels()):
        item.set_fontfamily("Times New Roman")
    leg = ax.get_legend()
    if leg:
        plt.setp(leg.get_texts(), fontfamily="Times New Roman")


# ---------------------------------------------------------------------------
# Plot 1 — 2-D contour
# ---------------------------------------------------------------------------
def plot_contour_2d(histories: List[History], f: Callable,
                    xlim, ylim, title="", resolution=200,
                    true_min=None):
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    levels = np.unique(np.percentile(Z, np.linspace(0, 95, 30)))

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.75)
    cbar = plt.colorbar(cf, ax=ax, label="f(x, y)")
    cbar.ax.yaxis.label.set_fontfamily("Times New Roman")
    plt.setp(cbar.ax.get_yticklabels(), fontfamily="Times New Roman")
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.4)

    # Draw paths
    for h in histories:
        color = COLORS[h.name]
        ax.plot(h.xs, h.ys, h.ls + "o", color=color,
                markersize=2, linewidth=1.4, alpha=0.80,
                label=h.label)
        ax.plot(h.xs[0], h.ys[0], "o", color=color, markersize=7, zorder=8)

    # Final stars with overlap jitter
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    nudge   = 0.018 * max(x_span, y_span)
    dirs    = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    finals  = [(h.xs[-1], h.ys[-1], COLORS[h.name], h.grad_type) for h in histories]
    offsets = [[0.0, 0.0] for _ in finals]
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            dx = finals[i][0] - finals[j][0]
            dy = finals[i][1] - finals[j][1]
            if (dx**2 + dy**2) ** 0.5 < nudge * 1.5:
                di = dirs[i % len(dirs)]
                dj = dirs[(i + 4) % len(dirs)]
                offsets[i][0] += di[0] * nudge
                offsets[i][1] += di[1] * nudge
                offsets[j][0] += dj[0] * nudge
                offsets[j][1] += dj[1] * nudge
    for (fx, fy, color, gtype), (ox, oy) in zip(finals, offsets):
        marker = "*" if gtype == "exact" else "P"   # star=exact, filled-plus=FD
        ax.plot(fx + ox, fy + oy, marker, color=color, markersize=16,
                markeredgecolor="black", markeredgewidth=0.9,
                zorder=11, clip_on=False)

    if true_min:
        for i, (tx, ty) in enumerate(true_min):
            lbl = "True min" if i == 0 else "_nolegend_"
            ax.plot(tx, ty, "x", color="#00CED1", markersize=13,
                    markeredgewidth=3.0, zorder=10, label=lbl)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x", **_TNR); ax.set_ylabel("y", **_TNR)
    ax.set_title(f"{title} — 2-D contour  (solid=exact, dashed=FD)", **_TNR)
    ax.legend(loc="upper right", fontsize=7,
              prop={"family": "Times New Roman"}, ncol=2)
    _apply_tnr(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_grad_contour2d.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — 3-D surface
# ---------------------------------------------------------------------------
def plot_surface_3d(histories: List[History], f: Callable,
                    xlim, ylim, title="", resolution=100,
                    true_min=None):
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    z_ceil  = float(np.percentile(Z, 97))
    Z_plot  = np.clip(Z, None, z_ceil)
    z_floor = float(Z_plot.min())

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z_plot, cmap="viridis", alpha=0.30,
                           linewidth=0, antialiased=True, rcount=80, ccount=80)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.12, label="f(x, y)")
    cbar.ax.yaxis.label.set_fontfamily("Times New Roman")
    plt.setp(cbar.ax.get_yticklabels(), fontfamily="Times New Roman")

    for h in histories:
        color = COLORS[h.name]
        fp    = np.clip(np.array(h.fs), None, z_ceil)
        ax.plot(h.xs, h.ys, fp, h.ls + "o", color=color,
                markersize=2, linewidth=1.5, alpha=0.9,
                label=h.label, zorder=5)
        ax.scatter([h.xs[0]], [h.ys[0]], [fp[0]],
                   color=color, s=70, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=6)
        ax.scatter([h.xs[-1]], [h.ys[-1]], [fp[-1]],
                   color=color,
                   s=220, marker="*" if h.grad_type == "exact" else "P",
                   edgecolors="black", linewidths=0.9, zorder=7)

    if true_min:
        for i, (tx, ty) in enumerate(true_min):
            tz  = np.clip(float(f(tx, ty)), None, z_ceil)
            lbl = "True min" if i == 0 else "_nolegend_"
            ax.scatter([tx], [ty], [tz], color="#00CED1", s=200, marker="x",
                       linewidths=3.0, zorder=10, label=lbl)
            ax.plot([tx, tx], [ty, ty], [z_floor, tz],
                    color="#00CED1", linewidth=1.2, linestyle=":", alpha=0.85)

    ax.set_xlabel("x", **_TNR, labelpad=8)
    ax.set_ylabel("y", **_TNR, labelpad=8)
    ax.set_zlabel("f(x, y)", **_TNR, labelpad=8)
    ax.set_title(f"{title} — 3-D surface  (solid=exact, dashed=FD)", **_TNR, pad=12)
    ax.legend(loc="upper left", fontsize=7,
              prop={"family": "Times New Roman"},
              framealpha=0.7, borderpad=0.6, ncol=2)
    ax.view_init(elev=32, azim=-55)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("lightgrey")
    _apply_tnr_3d(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_grad_surface3d.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — convergence
# ---------------------------------------------------------------------------
def plot_convergence(histories: List[History], title="", f_min=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in histories:
        color = COLORS[h.name]
        ax.semilogy(h.fs, color=color, linestyle=h.ls,
                    linewidth=1.5, label=h.label)
    if f_min is not None:
        f_plot = max(f_min, 1e-10)
        ax.axhline(f_plot, color="#00CED1", linewidth=1.4,
                   linestyle="--",
                   label=f"Global min  f={f_min:.4g}", zorder=5)
    ax.set_xlabel("Step", **_TNR)
    ax.set_ylabel("f(x, y)  [log scale]", **_TNR)
    ax.set_title(f"{title} — Convergence  (solid=exact, dashed=FD)", **_TNR)
    ax.legend(fontsize=7, prop={"family": "Times New Roman"}, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    _apply_tnr(ax)

    fig.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_grad_convergence.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(histories: List[History]):
    print(f"\n{'Optimizer':<18} {'Grad':>6} {'Final x':>10} "
          f"{'Final y':>10} {'Final f':>14}")
    print("-" * 62)
    for h in histories:
        fx, fy, ff = h.final()
        print(f"{h.name:<18} {h.grad_type:>6} {fx:>10.5f} "
              f"{fy:>10.5f} {ff:>14.6e}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ---- Choose function & domain ----
    FUNCTION      = rosenbrock   # rosenbrock | himmelblau | beale | sphere
    START         = (-2.0, -1.0)
    STEPS         = 2000
    PLOT_XLIM     = (-2.5, 2.5)
    PLOT_YLIM     = (-1.0, 3.5)
    TITLE         = "Rosenbrock"
    # rosenbrock: [(1,1)]  |  himmelblau: 4 pts  |  beale: [(3,0.5)]  |  sphere: [(0,0)]
    TRUE_MIN      = [(1.0, 1.0)]

    # ---- Per-solver hyperparameters ----
    ADAM_LR       = 0.5
    NEWTON_LR     = 0.75
    SGD_LR        = 0.0001
    SGDSIGN_LR    = 0.001
    SIGNUM_LR     = 0.01
    LION_LR       = 0.0018

    x0, y0 = START
    print(f"Function : {FUNCTION.__name__}")
    print(f"Start    : {START}")
    print(f"Steps    : {STEPS}\n")

    histories = run_all(
        FUNCTION, x0, y0, STEPS,
        ADAM_LR, NEWTON_LR, SGD_LR, SGDSIGN_LR, SIGNUM_LR, LION_LR
    )

    print_summary(histories)

    f_min = min(FUNCTION(tx, ty) for tx, ty in TRUE_MIN) if TRUE_MIN else None

    plot_contour_2d(histories, FUNCTION, PLOT_XLIM, PLOT_YLIM,
                    title=TITLE, true_min=TRUE_MIN)
    plot_surface_3d(histories, FUNCTION, PLOT_XLIM, PLOT_YLIM,
                    title=TITLE, true_min=TRUE_MIN)
    plot_convergence(histories, title=TITLE, f_min=f_min)
