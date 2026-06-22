"""Regenerate figures/pipeline_overview.png with correct pipeline order."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("white")

BOX_W = 3.6
BOX_H = 1.6
RADIUS = 0.15
BOX_COLOR = "#d5e8d4"
EDGE_COLOR = "#82b366"
TEXT_COLOR = "#222222"
ARROW_COLOR = "#555555"
TITLE_Y = 4.55

# ── box positions ─────────────────────────────────────────────────────────────
# Top row   (y_center = 3.3):  col 0, 1, 2
# Bottom row (y_center = 1.3): col 2, 1, 0  (snake: right→left)
# So: top-right (Signal Processing) → down → bottom-right (Window Features)
#     then bottom-right → left → bottom-mid (CDF/EMD) → left → bottom-left (Clustering Output)

COL_X = [0.4, 5.0, 9.6]   # left edge of each column
TOP_Y    = 2.3             # bottom edge of top-row boxes
BOTTOM_Y = 0.2             # bottom edge of bottom-row boxes
CY_TOP    = TOP_Y    + BOX_H / 2
CY_BOTTOM = BOTTOM_Y + BOX_H / 2

boxes = [
    # top row (left → right)
    dict(x=COL_X[0], y=TOP_Y,    title="Raw Harp CSV",
         body="Variable-width rows\nMotion packets: RegisterAddress = 34"),
    dict(x=COL_X[1], y=TOP_Y,    title="Cleaning",
         body="Counter gap detection\nInsert NaN rows, discard bad windows\nInterpolate remaining values"),
    dict(x=COL_X[2], y=TOP_Y,    title="Signal Processing",
         body="ADC scaling to physical units\nMedian filter + high-pass filter\nCompute 4 derived channels"),
    # bottom row (right → left — snake pattern)
    dict(x=COL_X[2], y=BOTTOM_Y, title="Window Features",
         body="Non-overlapping 300 ms windows\nPer-channel normalized histograms\n30-dimensional feature vector"),
    dict(x=COL_X[1], y=BOTTOM_Y, title="CDF / EMD Space",
         body="Histogram to CDF per channel\nL1 distance between CDFs\nEquivalent to 1D Wasserstein/EMD"),
    dict(x=COL_X[0], y=BOTTOM_Y, title="Clustering Output",
         body="AP-family methods or HDBSCAN\nClusterIdx, timestamp, folder label"),
]


def draw_box(ax, x, y, title, body):
    rect = mpatches.FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle=f"round,pad=0,rounding_size={RADIUS}",
        linewidth=1.4, edgecolor=EDGE_COLOR, facecolor=BOX_COLOR,
        zorder=2,
    )
    ax.add_patch(rect)
    cx = x + BOX_W / 2
    # title
    ax.text(cx, y + BOX_H - 0.28, title,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=TEXT_COLOR, zorder=3)
    # body
    ax.text(cx, y + BOX_H - 0.58, body,
            ha="center", va="top", fontsize=7.5, color="#444444",
            linespacing=1.55, zorder=3)


def arrow(ax, x1, y1, x2, y2, connectionstyle="arc3,rad=0.0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_COLOR,
                    lw=1.5,
                    mutation_scale=14,
                    connectionstyle=connectionstyle,
                ),
                zorder=1)


for b in boxes:
    draw_box(ax, **b)

# ── arrows ────────────────────────────────────────────────────────────────────
# Top row: box 0 → 1 → 2  (right-pointing)
gap = 0.08
arrow(ax, COL_X[0] + BOX_W + gap, CY_TOP, COL_X[1] - gap, CY_TOP)
arrow(ax, COL_X[1] + BOX_W + gap, CY_TOP, COL_X[2] - gap, CY_TOP)

# Down arrow: Signal Processing (top-right) → Window Features (bottom-right)
arrow(ax, COL_X[2] + BOX_W / 2, TOP_Y - gap,
          COL_X[2] + BOX_W / 2, BOTTOM_Y + BOX_H + gap)

# Bottom row: Window Features → CDF/EMD → Clustering Output  (left-pointing)
arrow(ax, COL_X[2] - gap,        CY_BOTTOM, COL_X[1] + BOX_W + gap, CY_BOTTOM)
arrow(ax, COL_X[1] - gap,        CY_BOTTOM, COL_X[0] + BOX_W + gap, CY_BOTTOM)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7, TITLE_Y, "Python behavioral clustering pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color=TEXT_COLOR)

plt.tight_layout(pad=0.3)
out = "pipeline_overview.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
