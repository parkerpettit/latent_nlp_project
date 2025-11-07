# -*- coding: utf-8 -*-
# Enhanced loss curve plotting script with arXiv red/gray theme
# Aligned with the second program's visual style

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# --------- Configuration ---------
SEARCH_DIRS = ["dataset_and_results/ablation_wo_curri"] 
FILE_GLOBS = ["**/*.txt", "*.txt", "*.log"]  # recursive first, then shallow
SMOOTH_WINDOW = 7  # moving average window size (adjust if you like)
SHOW_EACH_RUN = True  # set True to overlay each run's (smoothed) curve, faintly
SAVE_PATH = "plot_fig/figures/without_curri.png"
MAX_STEPS = 6000
SHOW_STATS_BOS = False
SHOW_LEGEND = False

# ----- arXiv Red / Gray palette -----
COLORS = {
    # primary elements
    'primary':   '#B31B1B',  # arXiv Red: main trend line
    'fill':      '#F6D6D6',  # light red fill for band
    'band_edge': '#7F7F7F',  # neutral gray for band borders

    # neutrals
    'individual':'#CFCFCF',  # individual runs (light gray)
    'text':      '#2F2F2F',  # titles/labels
    'tick':      '#3A3A3A',  # ticks/axis labels
    'axes_edge': '#BDBDBD',  # axes frame
    'grid':      '#D9D9D9',  # gridlines

    # optional accent
    'accent':    '#8E1212',  # deeper red for highlights (if needed)
}

# ---------------------------------

def find_txt_files(search_dirs, patterns):
    files = []
    for d in search_dirs:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat), recursive=True))
    # dedupe while preserving order
    seen = set()
    out = []
    for f in files:
        absf = os.path.abspath(f)
        if absf not in seen and os.path.isfile(absf):
            seen.add(absf)
            out.append(absf)
    return out

def parse_normal_losses(path):
    """Extract all 'Normal loss: <float>' values from a text file."""
    pat = re.compile(r"Normal\s+loss:\s*([0-9]*\.?[0-9]+)")
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
    return vals

def moving_average(x, w):
    if w <= 1:
        return x
    # Use convolution with 'same' length output
    kernel = np.ones(w, dtype=float) / float(w)
    # pad to reduce edge effects
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    y = np.convolve(xpad, kernel, mode='valid')
    return y

def setup_plot_style():
    """Set up arXiv-style plot theme (serif font + neutral axes)"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['axes_edge'],
        'axes.linewidth': 1.2,

        'grid.color': COLORS['grid'],
        'grid.alpha': 0.55,
        'grid.linewidth': 0.8,

        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['tick'],
        'xtick.color': COLORS['tick'],
        'ytick.color': COLORS['tick'],

        'font.family': 'serif',
        # 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif', 'Noto Serif CJK SC'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'legend.fontsize': 11,

        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# 1) Find files
all_files = find_txt_files(SEARCH_DIRS, FILE_GLOBS)

# 2) Parse losses from each file
runs = []
meta = []
for f in all_files:
    vals = parse_normal_losses(f)
    if len(vals) >= 2:  # keep only logs with at least 2 points
        runs.append(np.array(vals, dtype=float))
        meta.append((f, len(vals)))

print(f"Found {len(all_files)} .txt files, usable runs with Normal loss >=2 points: {len(runs)}")
for f, L in meta:
    print(f"  - {f}  (len={L})")

if len(runs) == 0:
    print("\nNo usable runs found.")
else:
    # Set up arXiv plot style
    setup_plot_style()
    
    # 3) Align by truncating to the minimum run length
    min_len = min(min(r.shape[0] for r in runs), MAX_STEPS)
    aligned = np.stack([r[:min_len] for r in runs], axis=0)  # shape: [num_runs, min_len]

    # 4) Compute mean and std across runs at each step
    mean_per_step = aligned.mean(axis=0)
    std_per_step = aligned.std(axis=0, ddof=0)

    # 5) Smooth (moving average)
    sm_mean = moving_average(mean_per_step, SMOOTH_WINDOW)
    upper = mean_per_step + std_per_step
    lower = mean_per_step - std_per_step
    sm_upper = moving_average(upper, SMOOTH_WINDOW)
    sm_lower = moving_average(lower, SMOOTH_WINDOW)

    x = np.arange(1, min_len + 1)

    # 6) Create arXiv-themed plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot individual runs (light gray + low alpha)
    if SHOW_EACH_RUN:
        for i, r in enumerate(runs):
            r_sm = moving_average(r[:min_len], SMOOTH_WINDOW)
            ax.plot(x, r_sm,
                    color=COLORS['individual'], alpha=0.30,
                    linewidth=1.2, linestyle='-', zorder=1)

    # Main trend line (arXiv red) with subtle white outline for clarity
    main_line = ax.plot(
        x, sm_mean,
        color=COLORS['primary'], linewidth=2.5,
        label="Mean Normal Loss (smoothed)", zorder=3, alpha=0.95
    )[0]
    main_line.set_path_effects([pe.Stroke(linewidth=5.0, foreground='white'), pe.Normal()])

    # Confidence band (light red fill)
    ax.fill_between(
        x, sm_lower, sm_upper,
        color=COLORS['fill'], alpha=0.50,
        label="±1 std (smoothed)", zorder=2
    )

    # Band borders (neutral gray dashed)
    ax.plot(x, sm_upper, color=COLORS['band_edge'], alpha=0.85, linewidth=1.2, linestyle='--', zorder=2)
    ax.plot(x, sm_lower, color=COLORS['band_edge'], alpha=0.85, linewidth=1.2, linestyle='--', zorder=2)
    
    # Title and labels
    ax.set_title('Training Loss Analysis - Normal Loss Across Runs',
                 fontsize=18, color=COLORS['text'], pad=20)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Normal Loss', fontsize=14)
    
    # Grid & layering
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Stats (optional box)
    min_loss = np.min(sm_mean)
    min_step = x[np.argmin(sm_mean)]
    final_loss = sm_mean[-1]
    
    stats_text = (
        f'Minimum Loss: {min_loss:.4f} (Step {min_step})\n'
        f'Final Loss: {final_loss:.4f}\n'
        f'Total Runs: {len(runs)}\n'
        f'Steps Analyzed: {min_len}'
    )
    if SHOW_STATS_BOS:
        ax.annotate(
            stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                      edgecolor=COLORS['axes_edge'], alpha=0.9, linewidth=1.2),
            fontsize=10, va='top', color=COLORS['text']
        )
    
    # Legend
    legend_handles = []
    legend_labels = []
    legend_handles.append(Line2D([0], [0], color=COLORS['primary'], linewidth=3.5, alpha=0.95))
    legend_labels.append('Mean Normal Loss (smoothed)')
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=COLORS['fill'], alpha=0.50))
    legend_labels.append('±1 std (smoothed)')
    if SHOW_EACH_RUN:
        legend_handles.append(Line2D([0], [0], color=COLORS['individual'], alpha=0.30, linewidth=1.2))
        legend_labels.append(f'Individual Runs (n={len(runs)})')

    if SHOW_LEGEND:
        legend = ax.legend(legend_handles, legend_labels, loc='upper right',
                           frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor(COLORS['axes_edge'])
    
    # Limits & ticks
    ax.set_xlim(x[0] - min_len*0.02, x[-1] + min_len*0.02)
    ax.set_ylim(-1.5, 7)  # keep your custom range
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=1)

    # Axes frame
    for spine in ax.spines.values():
        spine.set_color(COLORS['axes_edge'])
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Also save PDF version
    pdf_path = SAVE_PATH.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()

    print(f"\nEnhanced figure saved to: {SAVE_PATH}")
    print(f"PDF version saved to: {pdf_path}")
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("TRAINING ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Total runs analyzed: {len(runs)}")
    print(f"Steps per run: {min_len}")
    print(f"Smoothing window: {SMOOTH_WINDOW}")
    print(f"Minimum loss: {min_loss:.6f} (at step {min_step})")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {mean_per_step[0] - final_loss:.6f}")
    print(f"Final std: {std_per_step[-1]:.6f}")
    print(f"{'='*50}")
