"""Generate figures/data_stages_example.png
Shows one window's 30-dim histogram feature vector (Stage 3),
coloured by channel.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Stage-3 window (t_center = 339.627 s) ────────────────────────────────────
y_GA   = [0.0, 0.0, 0.0, 0.0, 0.5333, 0.4667, 0.0, 0.0, 0.0, 0.0, 0.0]          # 11 bins
z      = [0.0, 0.0, 0.0, 0.1167, 0.1833, 0.3, 0.05, 0.3, 0.05, 0.0, 0.0]        # 11 bins
z_gyro = [0.2167, 0.0333, 0.1833, 0.1833, 0.1333, 0.25]                           # 6 bins
log_a  = [0.0167, 0.9833]                                                           # 2 bins

values = y_GA + z + z_gyro + log_a   # 30 values
x      = np.arange(30)

colors = (
    ['#4C72B0'] * 11 +   # y_GA — blue
    ['#DD8452'] * 11 +   # z — orange
    ['#55A868'] * 6  +   # z_gyro — green
    ['#C44E52'] * 2      # log_a — red
)

fig, ax = plt.subplots(figsize=(11, 3.6))
bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.4)

# channel dividers
for boundary in [11, 22, 28]:
    ax.axvline(boundary - 0.5, color='#888', lw=0.8, ls='--')

# channel labels below x-axis
label_info = [
    (5,   r'$y_\mathrm{GA}$  (bins 0–10)',  '#4C72B0'),
    (16,  r'$z$  (bins 11–21)',              '#DD8452'),
    (24.5,r'$z_\mathrm{gyro}$  (bins 22–27)','#55A868'),
    (28.5,r'$\log a_\mathrm{tot}$  (28–29)', '#C44E52'),
]
for xc, lbl, col in label_info:
    ax.text(xc, -0.13, lbl, ha='center', va='top',
            fontsize=8.5, color=col, transform=ax.get_xaxis_transform())

ax.set_xlim(-0.7, 29.7)
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(30)], fontsize=7)
ax.set_xlabel('Bin index', labelpad=22, fontsize=10)
ax.set_ylabel('Normalised count', fontsize=10)
ax.set_title(
    'Stage 3 — 30-dimensional histogram feature vector\n'
    r'(window centre $t = 339.63\,$s, $\sum h = 1$ per channel)',
    fontsize=10.5)
ax.yaxis.grid(True, lw=0.4, alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig('data_stages_example.png', dpi=180, bbox_inches='tight')
print("Saved data_stages_example.png")
