import os
import matplotlib.pyplot as plt
import numpy as np

# Data (per-kernel order: FP16, FP8, INT8, INT4)
metrics = {
    "Mem throughput [GByte/s]": [260, 168, 206, 174],
    "L1 Hit Rate [%]": [60.6, 72.0, 71.6, 62.0],
    "L2 Hit Rate [%]": [33.3, 33.2, 34.0, 44.1],
    "Mem Busy [%]": [82.7, 62.7, 80.1, 61.0],
    "Max Bandwidth [%]": [87.0, 56.0, 69.0, 58.0],
    "Mem Pipes Busy [%]": [33.5, 42.3, 54.0, 51.6],
}

kernels = ["FP16", "FP8", "INT8", "INT4"]
colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4"]

out_dir = os.path.join("prof", "images", "run7")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "quantizations_memory_chart.png")

# Build grouped bar chart with a clean, professional style (manual rc params)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
})

labels = list(metrics.keys())
num_groups = len(labels)
num_bars = len(kernels)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(num_groups)
width = 0.18
# Twin axis: left = GByte/s (Mem throughput), right = percent metrics
ax2 = ax.twinx()

# Left axis range for Mem throughput
ax.set_ylim(150, 270)
    # ax.set_ylabel('Gbyte/s')  # Commenting out to avoid overwriting the left-axis label

# Right axis range for percent metrics
ax2.set_ylim(0, 100)
ax2.set_ylabel('%')

# Plot bars: for 'Mem throughput' use ax, for others use ax2
for i in range(num_bars):
    vals = [metrics[label][i] for label in labels]
    for gi, label in enumerate(labels):
        xpos = x[gi] + (i - (num_bars-1)/2) * width
        if 'Mem throughput' in label:
            ax.bar(xpos, vals[gi], width, color=colors[i], edgecolor='none')
        else:
            ax2.bar(xpos, vals[gi], width, color=colors[i], edgecolor='none')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_title('Quantizations - Memory Workload')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
# Create combined legend
handles_left, labels_left = ax.get_legend_handles_labels()
handles_right, labels_right = ax2.get_legend_handles_labels()
# We didn't add labels during bar plotting; create proxy handles for kernels
from matplotlib.patches import Patch
proxies = [Patch(facecolor=colors[i], label=kernels[i]) for i in range(len(kernels))]
ax.legend(handles=proxies, frameon=False, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(out_path)
