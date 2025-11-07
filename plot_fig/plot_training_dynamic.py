# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# arXiv color scheme and global style settings
# -----------------------------
COLORS = {
    'primary_red': '#B31B1B',
    'deep_red':    '#8E1212',
    'fill_red':    '#E8C1C1',

    'primary_gray': '#7F7F7F',
    'text':         '#2F2F2F',
    'tick':         '#3A3A3A',
    'axes_edge':    '#BDBDBD',
    'grid':         '#D9D9D9',
    'fill_gray':    '#D6D6D6',

    'individual':   '#CFCFCF',
}

def setup_plot_style():
    """Set arXiv-style plotting: serif font, white background, neutral gray axes and grid"""
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
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'legend.fontsize': 11,

        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# -----------------------------
# New: Parse Normal loss and Random contrast directly from combined_log.txt
# -----------------------------
def parse_combined_log(file_path):
    """
    Parse the merged log file directly and extract:
    - Normal loss
    - Random contrast
    Returns two lists.
    """
    normal_pattern = r'Normal loss:\s*([\d.]+)'
    contrast_pattern = r'Random contrast:\s*([\d.]+)'

    normal_values = []
    contrast_values = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        normal_matches = re.findall(normal_pattern, content)
        contrast_matches = re.findall(contrast_pattern, content)

        normal_values = [float(x) for x in normal_matches]
        contrast_values = [float(x) for x in contrast_matches]

        print(f"✅ Successfully parsed from {Path(file_path).name}:")
        print(f"   - Normal loss: {len(normal_values)} data points")
        print(f"   - Random contrast: {len(contrast_values)} data points")

        if len(normal_values) != len(contrast_values):
            print(f"⚠️  Warning: Sequence lengths differ, truncating to minimum length.")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return [], []

    min_len = min(len(normal_values), len(contrast_values))
    return normal_values[:min_len], contrast_values[:min_len]


# -----------------------------
# Plotting functions (unchanged)
# -----------------------------
def plot_smooth_curve(ax, values, x_values, y_lim, color_main, color_band, label, title, loss_type='normal'):
    if not values:
        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        return

    if len(values) > 2:
        try:
            from scipy.interpolate import make_interp_spline
            x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
            k = min(3, len(values)-1)
            spl = make_interp_spline(x_values, values, k=k)
            values_smooth = spl(x_smooth)
            line = ax.plot(x_smooth, values_smooth, linewidth=2.6,
                           color=color_main, label=label, alpha=0.95, zorder=3)[0]
            line.set_path_effects([pe.Stroke(linewidth=5.0, foreground='white'), pe.Normal()])

            std_val = np.std(values)
            ax.fill_between(x_smooth, values_smooth - 0.5*std_val, values_smooth + 0.5*std_val,
                            alpha=0.55, color=color_band, label='±0.5σ', zorder=2)
        except ImportError:
            if len(values) >= 3:
                degree = min(3, len(values)-1)
                coeffs = np.polyfit(x_values, values, degree)
                poly_func = np.poly1d(coeffs)
                x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
                values_smooth = poly_func(x_smooth)
                line = ax.plot(x_smooth, values_smooth, linewidth=2.6,
                               color=color_main, label=label, alpha=0.95, zorder=3)[0]
                line.set_path_effects([pe.Stroke(linewidth=5.0, foreground='white'), pe.Normal()])

                residuals = values - poly_func(x_values)
                std_residual = np.std(residuals)
                ax.fill_between(x_smooth, values_smooth - std_residual, values_smooth + std_residual,
                                alpha=0.35, color=color_band, label='Fit err.', zorder=2)
            else:
                line = ax.plot(x_values, values, linewidth=2.6,
                               color=color_main, label=label, alpha=0.95, zorder=3)[0]
                line.set_path_effects([pe.Stroke(linewidth=5.0, foreground='white'), pe.Normal()])
    else:
        line = ax.plot(x_values, values, linewidth=2.6,
                       color=color_main, label=label, alpha=0.95, zorder=3)[0]
        line.set_path_effects([pe.Stroke(linewidth=5.0, foreground='white'), pe.Normal()])

    mean_val = np.mean(values)
    min_val = np.min(values)
    max_val = np.max(values)

    spine_color = COLORS['primary_gray'] if loss_type == 'normal' else COLORS['primary_red']
    text_color = COLORS['tick'] if loss_type == 'normal' else COLORS['deep_red']

    ax.set_title(title, fontsize=14, color=text_color, pad=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)

    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    ax.tick_params(axis="both", which="major", labelsize=10)

    if y_lim is not None:
        ax.set_ylim(0, y_lim)
    else:
        y_margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.01
        ax.set_ylim(max(0, min_val - y_margin), max_val + y_margin)
    ax.set_xlim(0.5, len(values) + 0.5)

    legend = ax.legend(frameon=True, loc="upper right", fontsize=9,
                       fancybox=True, shadow=False, framealpha=0.92)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(spine_color)


def plot_dual_loss_comparison(normal_values, contrast_values, save_path=None):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)

    if normal_values:
        x_normal = np.arange(1, len(normal_values) + 1)
        plot_smooth_curve(
            ax1, normal_values, x_normal, y_lim=2.25,
            color_main=COLORS['primary_gray'], color_band=COLORS['fill_gray'],
            label='Cross-Entropy Loss', title='Cross-Entropy Loss',
            loss_type='normal'
        )

    if contrast_values:
        x_contrast = np.arange(1, len(contrast_values) + 1)
        plot_smooth_curve(
            ax2, contrast_values, x_contrast, y_lim=None,
            color_main=COLORS['primary_red'], color_band=COLORS['fill_red'],
            label='Separation Loss', title='Separation Loss',
            loss_type='contrast'
        )

    fig.suptitle('Training Loss: Cross-Entropy (Left) and Separation (Right)',
                 fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Chart saved to: {save_path}")

    # Optional backup save
    backup_path = "plot_fig/figures/training_curve.png"
    try:
        plt.savefig(backup_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Chart saved to: {backup_path}")
    except Exception as e:
        print(f"⚠️  Backup save failed: {e}")

    return fig, (ax1, ax2)


def analyze_dual_loss(normal_values, contrast_values):
    print("\n" + "="*80)
    print("Dual Loss Analysis Report (Based on Merged Log)")
    print("="*80)

    if normal_values:
        print(f"\n• Cross-Entropy Loss:")
        print(f"  Data points: {len(normal_values)}")
        print(f"  Mean: {np.mean(normal_values):.6f}")
        print(f"  Std Dev: {np.std(normal_values, ddof=1):.6f}")
        print(f"  Min / Max: {np.min(normal_values):.6f} / {np.max(normal_values):.6f}")
        if len(normal_values) > 1:
            x = np.arange(len(normal_values))
            slope = np.polyfit(x, normal_values, 1)[0]
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            print(f"  Trend: {trend} (slope: {slope:.8f})")

    if contrast_values:
        print(f"\n• Separation Loss:")
        print(f"  Data points: {len(contrast_values)}")
        print(f"  Mean: {np.mean(contrast_values):.6f}")
        print(f"  Std Dev: {np.std(contrast_values, ddof=1):.6f}")
        print(f"  Min / Max: {np.min(contrast_values):.6f} / {np.max(contrast_values):.6f}")
        if len(contrast_values) > 1:
            x = np.arange(len(contrast_values))
            slope = np.polyfit(x, contrast_values, 1)[0]
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            print(f"  Trend: {trend} (slope: {slope:.8f})")

    if normal_values and contrast_values:
        ratio = np.mean(contrast_values) / np.mean(normal_values)
        print(f"\n• Comparison: mean(Separation) / mean(Cross-Entropy) = {ratio:.4f}")
        if len(contrast_values) == len(normal_values):
            corr = np.corrcoef(contrast_values, normal_values)[0, 1]
            desc = "strong correlation" if abs(corr) > 0.7 else "moderate correlation" if abs(corr) > 0.3 else "weak correlation"
            print(f"  Correlation coefficient: {corr:.4f} ({desc})")


# -----------------------------
# Main program: read only + plotting
# -----------------------------
def main():
    # Update this path to your actual merged log file
    combined_log_path = r"dataset_and_results\training_dynamic.log"

    print("Loading data from merged log file...")
    normal_values, contrast_values = parse_combined_log(combined_log_path)

    if not normal_values or not contrast_values:
        print("❌ No valid data available. Exiting.")
        return

    analyze_dual_loss(normal_values, contrast_values)

    print("\nGenerating charts...")
    plot_dual_loss_comparison(
        normal_values=normal_values,
        contrast_values=contrast_values,
        save_path="plot_fig/figures/training_curve.png"
    )
    print("\n✅ Done! The chart is now fully based on static merged data and is reproducible.")


if __name__ == "__main__":
    main()