"""Whiteboard-style figures for the STK120 mentor app.

Run from the App folder:  python figurer/make_figures.py
Each figure is kept simple enough to be redrawn by hand on a whiteboard.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
INK = "#1f2a37"
BLUE = "#1c64f2"
RED = "#e02424"
GREEN = "#057a55"
ORANGE = "#d97706"
GREY = "#9ca3af"

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

rng = np.random.default_rng(120)


def board_axes(ax, xlabel="x", ylabel="y"):
    """Plain axes without ticks, like a hand-drawn graph."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top", rotation=0, labelpad=10)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=False, facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- pass 1
def weak_noise(x, target=0.3):
    """Noise that gives a correlation close to target (searches fixed seeds, reproducible)."""
    for seed in range(500):
        e = np.random.default_rng(seed).normal(0, 2.8, len(x))
        if abs(np.corrcoef(x, 0.3 * x + e)[0, 1] - target) < 0.02:
            return e
    return e


def korrelation():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    n = 40
    x = np.linspace(0, 10, n)
    sets = [
        (x, 2 + 0.8 * x + rng.normal(0, 0.9, n), "Starkt positivt"),
        (x, 2 + 0.3 * x + weak_noise(x), "Svagt positivt"),
        (x, 0.35 * (x - 5) ** 2 + rng.normal(0, 0.6, n), "Tydligt samband, men inte linjärt"),
    ]
    for ax, (xx, yy, title) in zip(axes, sets):
        ax.scatter(xx, yy, s=22, color=BLUE)
        board_axes(ax)
        r = np.corrcoef(xx, yy)[0, 1]
        ax.set_title(f"{title}\nr = {r:.2f}".replace(".", ","), fontsize=12)
    fig.tight_layout()
    save(fig, "p1_korrelation")


def t_fordelning():
    from math import gamma, pi, sqrt
    df = 42
    t = np.linspace(-4, 4, 400)
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    f = c * (1 + t ** 2 / df) ** (-(df + 1) / 2)
    tcrit, tobs = 2.02, 2.6
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, f, color=INK, lw=2)
    for side in (1, -1):
        m = side * t >= tcrit
        ax.fill_between(t[m], f[m], color=RED, alpha=0.18)
        m2 = side * t >= tobs
        ax.fill_between(t[m2], f[m2], color=RED, alpha=0.55)
    ax.axvline(tcrit, color=RED, ls="--", lw=1.4); ax.axvline(-tcrit, color=RED, ls="--", lw=1.4)
    ax.axvline(tobs, color=BLUE, lw=2.2)
    ax.text(-tcrit - 0.1, 0.36, "kritiska värden\n(α/2 i varje svans,\nljusröda ytor)", color=RED, ha="right", fontsize=10)
    ax.text(tobs + 0.08, 0.36, "observerat t", color=BLUE, fontsize=10)
    ax.annotate("p-värdet = de mörka ytorna\n(bortom ±observerat t)", xy=(2.95, 0.006), xytext=(2.2, 0.17), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(0, 0.16, "Om H0 är sann\nhamnar t oftast här", ha="center", fontsize=10, color=GREY)
    ax.set_yticks([]); ax.set_xticks([-tcrit, 0, tcrit]); ax.set_xticklabels(["−t krit", "0", "t krit"])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 0.48)
    save(fig, "p1_tfordelning")


# ---------------------------------------------------------------- pass 2
def residual():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([3.0, 4.2, 2.6, 5.4, 5.2, 8.6, 6.6, 8.2, 8.9])
    b1, b0 = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xx = np.linspace(0.5, 9.5, 2)
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    ax.scatter(x, y, s=34, color=BLUE, zorder=3)
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, b0 + b1 * xi], color=GREY, lw=1)
    i = 5  # x=6, clearly above the line
    yh = b0 + b1 * x[i]
    ax.plot([x[i], x[i]], [y[i], yh], color=RED, lw=3)
    ax.scatter([x[i]], [yh], s=40, facecolor="white", edgecolor=INK, zorder=4)
    ax.annotate("y (observerat)", xy=(x[i], y[i]), xytext=(3.2, 9.2), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("ŷ (på linjen)", xy=(x[i], yh), xytext=(7.0, 4.6), fontsize=11,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(x[i] - 0.2, (y[i] + yh) / 2, "e = y − ŷ > 0\nmodellen underskattar", color=RED, fontsize=10,
            va="center", ha="right")
    j = 2  # x=3, below the line
    ax.plot([x[j], x[j]], [y[j], b0 + b1 * x[j]], color=ORANGE, lw=3)
    ax.text(x[j] + 0.2, y[j] + 0.4, "e < 0\nmodellen överskattar", color=ORANGE, fontsize=10, va="center")
    ax.text(8.0, b0 + b1 * 9.5 + 0.2, "ŷ = b₀ + b₁x", fontsize=12, ha="center")
    board_axes(ax)
    save(fig, "p2_residual")


def population_stickprov():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xx = np.linspace(0, 10, 2)
    ax.plot(xx, 2 + 0.7 * xx, color=GREY, lw=2.4, ls="--", label="populationen: y = β₀ + β₁x + ε (okänd)")
    for col, seed, k in [(BLUE, 3, 1), (GREEN, 11, 2)]:
        g = np.random.default_rng(seed)
        x = g.uniform(0.5, 9.5, 10)
        y = 2 + 0.7 * x + g.normal(0, 1.6, 10)
        b1, b0 = np.polyfit(x, y, 1)
        ax.scatter(x, y, s=24, color=col, alpha=0.85)
        lab = f"stickprov {k}: ŷ = {b0:.1f} + {b1:.2f}x".replace(".", ",")
        ax.plot(xx, b0 + b1 * xx, color=col, lw=2, label=lab)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    board_axes(ax)
    ax.set_title("Varje stickprov ger nya b, men β är densamma", fontsize=12, pad=12)
    save(fig, "p2_population_stickprov")


def abc_figur():
    """The exam's A/B/C figure (250325 q 11, 250820 q 9), redrawn."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b0, b1 = 1.0, 0.55
    xx = np.linspace(0.3, 9.5, 2)
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    ybar, xbar = 3.6, (3.6 - b0) / b1
    ax.axhline(ybar, color=GREEN, lw=1.6)
    ax.plot([xbar, xbar], [0, ybar], color=GREEN, lw=1.2)
    xi, yi = 7.0, 6.9
    yh = b0 + b1 * xi
    ax.plot([xi, xi], [0, yi], color=GREY, ls=":", lw=1.2)
    ax.plot([0, xi], [yi, yi], color=GREY, ls=":", lw=1.2)
    for yv, lab in [(yi, "A"), (yh, "B"), (ybar, "C")]:
        ax.scatter([xi], [yv], color=RED, zorder=4, s=36)
        ax.text(xi - 0.25, yv + 0.12, lab, fontsize=13, ha="right")
    ax.text(9.6, b0 + b1 * 9.5, "ŷᵢ = b₀ + b₁xᵢ", fontsize=11, va="center")
    ax.text(9.6, ybar, "ȳ", fontsize=13, color=GREEN, va="center")
    ax.set_xticks([xbar, xi]); ax.set_xticklabels(["x̄", "xᵢ"])
    ax.set_yticks([yi]); ax.set_yticklabels(["yᵢ"])
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 8)
    ax.text(xi + 0.3, yi - 0.1, "A–B: residual (y − ŷ)", color=RED, fontsize=10, va="center")
    ax.text(xi + 0.3, (yh + ybar) / 2, "B–C: förklarad del (ŷ − ȳ)", color=GREEN, fontsize=10, va="center")
    ax.set_xlabel("x", loc="right"); ax.set_ylabel("y", loc="top", rotation=0)
    save(fig, "p2_abc")


# ---------------------------------------------------------------- pass 3
def dummy_parallell():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xx = np.linspace(0, 10, 2)
    b0, b1, b2 = 2.0, 0.6, 2.0
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2.2)
    ax.plot(xx, b0 + b2 + b1 * xx, color=BLUE, lw=2.2)
    ax.text(10.1, b0 + b1 * 10, "d = 0 (referensgrupp)\nŷ = b₀ + b₁x", va="center", fontsize=10)
    ax.text(10.1, b0 + b2 + b1 * 10, "d = 1\nŷ = (b₀ + b₂) + b₁x", va="center", color=BLUE, fontsize=10)
    ax.annotate("", xy=(4, b0 + b2 + b1 * 4), xytext=(4, b0 + b1 * 4), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(4.15, b0 + b1 * 4 + b2 / 2, "b₂ = skillnad i nivå\n(samma för alla x)", color=RED, va="center", fontsize=10)
    ax.text(7.0, b0 + b1 * 7.0 - 1.0, "båda linjerna har\nsamma lutning b₁", fontsize=10, color=GREY, va="top")
    board_axes(ax)
    save(fig, "p3_dummy_parallell")


def interaktion_cykel():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xx = np.linspace(0, 12, 2)
    ax.plot(xx, 42 - 3.1 * xx, color=INK, lw=2.2)
    ax.plot(xx, 50.4 - 1.2 * xx, color=BLUE, lw=2.2)
    ax.text(12.2, 42 - 3.1 * 12, "utan cykelväg:\n42 − 3,1·avstånd", va="center", fontsize=10)
    ax.text(12.2, 50.4 - 1.2 * 12, "med cykelväg:\n50,4 − 1,2·avstånd", va="center", color=BLUE, fontsize=10)
    ax.scatter([5], [44.4], s=60, color=RED, zorder=4)
    ax.text(5.3, 46, "5 km: 44,4 %", color=RED, fontsize=10)
    ax.annotate("", xy=(0.15, 50.4), xytext=(0.15, 42), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2))
    ax.text(0.45, 44.6, "8,4 vid 0 km", color=GREEN, fontsize=9, va="center")
    ax.set_ylim(0, 56)
    board_axes(ax, "avstånd (km)", "cykelresande (%)")
    save(fig, "p3_interaktion_cykel")


def sst_uppdelning():
    x = np.array([1, 2, 3, 4, 5, 6.5, 9, 10])
    y = np.array([2.0, 3.1, 2.6, 4.6, 4.0, 8.4, 6.9, 7.8])
    b1, b0 = np.polyfit(x, y, 1)
    ybar = y.mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xx = np.linspace(0.5, 10.5, 2)
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    ax.axhline(ybar, color=GREY, ls="--", lw=1.6)
    ax.text(10.4, ybar - 0.15, "ȳ (medelvärdet)", color=GREY, fontsize=10, ha="right", va="top")
    ax.scatter(x, y, s=30, color=BLUE, zorder=3)
    i = 5
    yh = b0 + b1 * x[i]
    ax.plot([x[i] + 0.15] * 2, [ybar, y[i]], color=INK, lw=2.5)
    ax.plot([x[i] - 0.15] * 2, [ybar, yh], color=GREEN, lw=4)
    ax.plot([x[i] - 0.15] * 2, [yh, y[i]], color=RED, lw=4)
    ax.text(x[i] + 0.4, (ybar + y[i]) / 2 + 0.3, "y − ȳ\ntotal → SST", fontsize=10, va="center")
    ax.text(x[i] - 0.4, (ybar + yh) / 2, "ŷ − ȳ  förklarad → SSR", color=GREEN, fontsize=10, ha="right", va="center")
    ax.text(x[i] - 0.4, (yh + y[i]) / 2, "y − ŷ  oförklarad → SSE", color=RED, fontsize=10, ha="right", va="center")
    board_axes(ax)
    ax.set_title("SST = SSR + SSE     R² = SSR / SST", fontsize=12)
    save(fig, "p3_sst_uppdelning")


def r2_justerat():
    n = 25
    g = np.random.default_rng(7)
    x1 = g.normal(size=n)
    y = 1 + 1.2 * x1 + g.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    r2, r2a = [], []
    for k in range(1, 9):
        if k > 1:
            X = np.column_stack([X, g.normal(size=n)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        r = 1 - sse / sst
        r2.append(r)
        r2a.append(1 - (1 - r) * (n - 1) / (n - k - 1))
    ks = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ks, r2, marker="o", color=BLUE, lw=2, label="R²: kan aldrig minska")
    ax.plot(ks, r2a, marker="o", color=RED, lw=2, label="justerat R²: straffar onödiga variabler")
    ax.set_xticks(ks)
    ax.set_xlabel("antal x-variabler (k); efter den första är alla rent brus")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0,5", "1"])
    save(fig, "p3_r2_justerat")


# ---------------------------------------------------------------- pass 4
def ki_pi():
    n = 25
    x = np.sort(rng.uniform(1, 9, n))
    y = 3 + 0.8 * x + rng.normal(0, 1.1, n)
    b1, b0 = np.polyfit(x, y, 1)
    res = y - (b0 + b1 * x)
    se = np.sqrt((res ** 2).sum() / (n - 2))
    xx = np.linspace(0, 10, 200)
    sxx = ((x - x.mean()) ** 2).sum()
    se_mean = se * np.sqrt(1 / n + (xx - x.mean()) ** 2 / sxx)
    se_pred = np.sqrt(se ** 2 + se_mean ** 2)
    t = 2.07
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    yh = b0 + b1 * xx
    ax.fill_between(xx, yh - t * se_pred, yh + t * se_pred, color=ORANGE, alpha=0.18, label="prediktionsintervall: en enskild ny observation")
    ax.fill_between(xx, yh - t * se_mean, yh + t * se_mean, color=BLUE, alpha=0.3, label="konfidensintervall: medelvärdet av y")
    ax.plot(xx, yh, color=INK, lw=2)
    ax.scatter(x, y, s=20, color=INK, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    board_axes(ax)
    ax.text(9.9, (yh - t * se_pred).min(), "banden är smalast vid x̄\noch bredast ut mot kanterna", fontsize=9, ha="right", va="bottom", color=GREY)
    save(fig, "p4_ki_pi")


def lena_ph():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    n = 80
    # L: linearity broken -> curved residual pattern
    x = np.linspace(0, 10, n)
    ax = axes[0, 0]
    ax.scatter(x, 0.25 * (x - 5) ** 2 - 2 + rng.normal(0, 0.6, n), s=12, color=BLUE)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_title("L  Linjäritet\nbrott: böjt mönster i residualerna", fontsize=10)
    board_axes(ax, "x", "e")
    # E: exogeneity, cannot be seen
    ax = axes[0, 1]
    ax.scatter(x, rng.normal(0, 1, n), s=12, color=GREY)
    ax.axhline(0, color=GREY, lw=1)
    ax.text(5, 0, "syns INTE\ni en residualplot", ha="center", va="center", fontsize=11, color=RED,
            bbox=dict(facecolor="white", edgecolor=RED))
    ax.set_title("E  Exogenitet (ingen endogenitet)\nbrott: t.ex. utelämnad variabel", fontsize=10)
    board_axes(ax, "x", "e")
    # N: normality broken -> skewed histogram
    ax = axes[0, 2]
    ax.hist(rng.exponential(1, 400) - 1, bins=20, color=BLUE, edgecolor="white")
    ax.set_title("N  Normalfördelade feltermer\nbrott: skevt histogram", fontsize=10)
    board_axes(ax, "e", "")
    # A: autocorrelation -> waves over time
    tt = np.arange(n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.85 * e[i - 1] + rng.normal(0, 0.5)
    ax = axes[1, 0]
    ax.plot(tt, e, color=BLUE, marker="o", ms=3, lw=1)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_title("A  Ingen autokorrelation\nbrott: residualerna \"följer\" varandra över tid", fontsize=10)
    board_axes(ax, "tid", "e")
    # P: perfect multicollinearity
    ax = axes[1, 1]
    x1 = rng.uniform(0, 10, 30)
    ax.scatter(x1, 2 * x1, s=14, color=BLUE)
    ax.set_title("P  Ingen perfekt multikollinjäritet\nbrott: x₂ = 2·x₁ exakt", fontsize=10)
    board_axes(ax, "x₁", "x₂")
    # H: heteroskedasticity -> funnel
    ax = axes[1, 2]
    ax.scatter(x, rng.normal(0, 0.05 + 0.45 * x, n), s=12, color=BLUE)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_title("H  Homoskedasticitet\nbrott: trattform", fontsize=10)
    board_axes(ax, "x eller ŷ", "e")
    fig.tight_layout()
    save(fig, "p4_lena_ph")


def kvadratisk():
    b0, b1, b2 = 1.0, 2.4, -0.2
    xx = np.linspace(0, 12, 200)
    f = lambda x: b0 + b1 * x + b2 * x ** 2
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xx, f(xx), color=INK, lw=2.2)
    for x0, col in [(2, GREEN), (9, RED)]:
        slope = b1 + 2 * b2 * x0
        tx = np.linspace(x0 - 1.8, x0 + 1.8, 2)
        ax.plot(tx, f(x0) + slope * (tx - x0), color=col, lw=2)
        ax.scatter([x0], [f(x0)], color=col, zorder=4)
        lab = f"x = {x0}: lutning b₁ + 2b₂x = {slope:.1f}".replace(".", ",")
        ax.text(x0, f(x0) - 1.2 if x0 == 2 else f(x0) + 0.6, lab, color=col, fontsize=9, ha="center")
    xv = -b1 / (2 * b2)
    ax.axvline(xv, color=GREY, ls="--", lw=1.4)
    ax.text(xv + 0.15, 1.2, "vändpunkt\nx = −b₁ / (2b₂) = 6", ha="left", fontsize=10)
    board_axes(ax)
    ax.set_title("ŷ = b₀ + b₁x + b₂x²  (b₂ < 0 ger en topp)", fontsize=12)
    save(fig, "p4_kvadratisk")


# ---------------------------------------------------------------- pass 5
def logmodeller():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.linspace(0.3, 10, 200)
    axes[0].plot(x, 1 + 2 * np.log(x), color=BLUE, lw=2.2)
    axes[0].set_title("Lin-log: y = β₀ + β₁ln x\n1 % mer x → β₁/100 enheter y", fontsize=10)
    axes[1].plot(x, np.exp(0.25 * x), color=GREEN, lw=2.2)
    axes[1].set_title("Log-lin: ln y = β₀ + β₁x\n1 enhet mer x → ca 100·β₁ % y", fontsize=10)
    axes[2].plot(x, 2 * x ** 0.5, color=ORANGE, lw=2.2)
    axes[2].set_title("Log-log: ln y = β₀ + β₁ln x\n1 % mer x → β₁ % y (elasticitet)", fontsize=10)
    for ax in axes:
        board_axes(ax)
    fig.tight_layout()
    save(fig, "p5_logmodeller")


def lpm_logit():
    n = 60
    x = np.sort(rng.uniform(0, 10, n))
    p = 1 / (1 + np.exp(-(x - 5) * 1.1))
    y = (rng.uniform(size=n) < p).astype(float)
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(-1, 11, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y + rng.normal(0, 0.015, n), s=18, color=INK, alpha=0.6)
    ax.plot(xx, b0 + b1 * xx, color=RED, lw=2.2, label="LPM: rak linje")
    ax.plot(xx, 1 / (1 + np.exp(-(xx - 5) * 1.1)), color=BLUE, lw=2.2, label="logit: S-kurva")
    ax.axhspan(1, 1.35, color=RED, alpha=0.08); ax.axhspan(-0.35, 0, color=RED, alpha=0.08)
    ax.text(10.8, 1.17, "\"sannolikhet\" > 1", color=RED, fontsize=9, ha="right")
    ax.text(-0.8, -0.2, "\"sannolikhet\" < 0", color=RED, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_xticks([])
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("x", loc="right")
    ax.legend(frameon=False, fontsize=10, loc="center left")
    save(fig, "p5_lpm_logit")


def fixa_effekter():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = [BLUE, GREEN, ORANGE]
    allx, ally = [], []
    for g, (cx, cy) in enumerate([(2, 2), (5, 5), (8, 8)]):
        x = cx + rng.uniform(-1.2, 1.2, 10)
        y = cy - 0.8 * (x - cx) + rng.normal(0, 0.3, 10)
        ax.scatter(x, y, s=22, color=cols[g])
        tx = np.array([cx - 1.4, cx + 1.4])
        ax.plot(tx, cy - 0.8 * (tx - cx), color=cols[g], lw=2)
        ax.text(cx + 1.5, cy - 0.8 * 1.4, f"enhet {g + 1}", color=cols[g], fontsize=9)
        allx += list(x); ally += list(y)
    b1, b0 = np.polyfit(allx, ally, 1)
    tx = np.array([0, 10])
    ax.plot(tx, b0 + b1 * tx, color=GREY, lw=2, ls="--")
    ax.text(0.2, 9.2, "streckad: utan fixa effekter\n→ positiv lutning (fel)", color=GREY, fontsize=9, va="top")
    ax.text(6.2, 2.0, "med fixa effekter:\nen egen konstant per enhet,\njämför bara inom enheten\n→ negativ lutning", fontsize=9)
    board_axes(ax)
    save(fig, "p5_fixa_effekter")


def ovb():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=INK, lw=1.8)
    ax.text(0.15, 0.25, "x\n(t.ex. utbildning)", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.85, 0.25, "y\n(t.ex. lön)", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.5, 0.85, "z utelämnad\n(t.ex. förmåga)", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fde8e8", edgecolor=RED, lw=1.8), fontsize=11)
    ar = dict(arrowstyle="-|>", lw=2, color=INK, mutation_scale=18)
    ax.annotate("", xy=(0.7, 0.25), xytext=(0.3, 0.25), arrowprops=ar)
    ax.annotate("", xy=(0.22, 0.42), xytext=(0.42, 0.72), arrowprops=dict(ar, color=RED))
    ax.annotate("", xy=(0.78, 0.42), xytext=(0.58, 0.72), arrowprops=dict(ar, color=RED))
    ax.text(0.5, 0.17, "β₁ vill vi skatta", ha="center", fontsize=10)
    ax.text(0.2, 0.62, "1. z hänger\nihop med x", color=RED, fontsize=9, ha="right")
    ax.text(0.8, 0.62, "2. z påverkar y", color=RED, fontsize=9, ha="left")
    ax.text(0.5, -0.02, "Båda villkoren uppfyllda → b₁ blir skev (OVB)", ha="center", fontsize=10, color=RED)
    save(fig, "p5_ovb")


if __name__ == "__main__":
    for fn in [korrelation, t_fordelning, residual, population_stickprov, abc_figur, dummy_parallell, interaktion_cykel,
               sst_uppdelning, r2_justerat, ki_pi, lena_ph, kvadratisk, logmodeller, lpm_logit, fixa_effekter, ovb]:
        fn()
        print("ok", fn.__name__)
