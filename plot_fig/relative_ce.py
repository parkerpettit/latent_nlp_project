#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description: This script compares "Training" vs. "Training-Free" methods
# by plotting Rate vs. Relative Cross-Entropy curves. It includes analysis
# of the vertical gap, Area Under the Gap (AUG), and rate savings.
# It automatically parses rate units like "bits/token" or "bits/hidden state".

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
import matplotlib.patheffects as pe
import argparse

# === Configuration ===
MARKER_TARGET = 15          # Number of markers to display per curve
MAX_POINTS_PER_LINE = None  # Downsample curves to this many points (None = no downsampling)
SMOOTH_WINDOW = 3           # Light smoothing window for curves (=1 to disable)
OUTPUT_DIR = "plot_fig/figures" # Default output directory

# === Rate Savings Analysis Parameters ===
RATE_ANALYSIS_POINTS = [89075.6, 178151.2]  # Analyze rate savings at these specific rate points

# --- Optional library for non-overlapping text labels ---
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# ==================================
# Regex and Helpers for Unit Parsing
# ==================================
# Header format: ===== No-label Summary (bits/token) =====
# or:            ===== No-label Summary (bits/hidden state) =====
NO_LABEL_HEADER_RE = re.compile(
    r'===== No-label Summary \((?P<unit>bits/[^\)]+)\) ====='
)

# Supports both '→' and '->' arrows
ARROW = r'(?:→|->)'

# Rate line format:
# "Rate bits/token (full → comp)  : 14.3 → 2.1"
# "Rate bits/hidden state (full → comp)  : 609734.9 → 11134.5"
RATE_LINE_RE = re.compile(
    rf'Rate\s+(?P<unit>bits/\s*[A-Za-z ]+)\s*\(full\s*{ARROW}\s*comp\)\s*:\s*'
    rf'(?P<full>[0-9.]+)\s*{ARROW}\s*(?P<comp>[0-9.]+)'
)

def iter_no_label_sections_with_unit(content: str):
    """
    Iterates through "No-label Summary (...)" sections in the content.
    Yields tuples of (unit, section_text), where unit is e.g., 'bits/token'.
    """
    matches = list(NO_LABEL_HEADER_RE.finditer(content))
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        unit = m.group('unit').strip()
        sections.append((unit, content[start:end]))
    return sections

def detect_rate_unit_from_text(text: str, default_unit: str = 'bits/token'):
    """
    Tries to detect the rate unit from text, first from headers, then from rate lines.
    Falls back to a default unit if none is found.
    """
    m = NO_LABEL_HEADER_RE.search(text)
    if m:
        return m.group('unit').strip()
    m2 = RATE_LINE_RE.search(text)
    if m2:
        return m2.group('unit').strip()
    return default_unit


# =========================
# Parsing Functions
# =========================
def parse_drop_tail_txt(filename):
    """
    Parses logs based on the 'drop tail' method.
    Adapts to different rate units (bits/token, bits/hidden state, etc.).
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Determine a global unit as a fallback
    global_unit = detect_rate_unit_from_text(content, default_unit='bits/token')

    sections = re.split(r'===== Evaluating drop tail (\d+)%', content)
    for i in range(1, len(sections), 2):
        drop_pct = int(sections[i])
        section_content = sections[i + 1]

        # Find the last CE line
        progress_lines = re.findall(
            r'drop=\d+%:.*?CEc=([0-9.]+).*?CEf=([0-9.]+)',
            section_content
        )
        if not progress_lines:
            continue
        ce_comp, ce_full = map(float, progress_lines[-1])

        # Find the rate line
        rate_match = RATE_LINE_RE.search(section_content)
        if rate_match:
            unit = rate_match.group('unit').strip()
            rate_comp = float(rate_match.group('comp'))
        else:
            unit = global_unit
            rate_comp = 0.0  # Set to 0 if not found

        data.append({
            'drop_pct': drop_pct,
            'comp_rate_bits_per_token': rate_comp,
            'CE_full_bits': ce_full,
            'CE_comp_bits': ce_comp,
            'method_detail': f'drop_{drop_pct}%',
            'rate_unit': unit
        })
    return pd.DataFrame(data)


def parse_length_values_txt(filename):
    """Parses training compression data based on the 'length' parameter, with unit adaptation."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = iter_no_label_sections_with_unit(content)
    for unit, section in sections:
        ce_full_match = re.search(r'CE_full\(y_full\)\s*:\s*mean=([0-9.]+)', section)
        ce_comp_match = re.search(r'CE_comp\(y_full\)\s*:\s*mean=([0-9.]+)', section)
        length_match = re.search(r'length\s*=\s*(\d+)', section)
        rate_match = RATE_LINE_RE.search(section)

        if not (ce_full_match and ce_comp_match and length_match and rate_match):
            continue # Skip section if any key field is missing

        ce_full = float(ce_full_match.group(1))
        ce_comp = float(ce_comp_match.group(1))
        length_value = int(length_match.group(1))
        unit_from_rate = rate_match.group('unit').strip()
        rate_comp = float(rate_match.group('comp'))

        data.append({
            'length_value': length_value,
            'comp_rate_bits_per_token': rate_comp,
            'CE_full_bits': ce_full,
            'CE_comp_bits': ce_comp,
            'method_detail': f'length_{length_value}',
            'rate_unit': unit_from_rate
        })

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('length_value').reset_index(drop=True)
    return df


def parse_k_values_txt(filename):
    """Parses data based on K-values, with unit adaptation."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = iter_no_label_sections_with_unit(content)
    if not sections: # Fallback for older data formats
        sections_raw = content.split('===== No-label Summary (bits/token) =====')
        sections = [('bits/token', s) for s in sections_raw[1:]]

    for i, (unit, section) in enumerate(sections, 1):
        # Try to extract K value
        k_patterns = [r'/K(\d+K?)', r'K(\d+K?)/', r'/(\d+)K', r'\bK\s*=\s*(\d+)\b']
        k_value = None
        for pattern in k_patterns:
            k_matches = re.findall(pattern, section[:600])
            if k_matches:
                k_value = k_matches[-1]
                break
        if not k_value and 'ZERO' in section[:200]:
            k_value = '0'
        if not k_value:
            continue

        # CE values
        ce_comp_m = re.search(r'CE_comp\(y_full\)\s*:\s*mean=([0-9.]+)', section)
        ce_full_m = re.search(r'CE_full\(y_full\)\s*:\s*mean=([0-9.]+)', section)
        if not (ce_comp_m and ce_full_m):
            continue

        ce_comp = float(ce_comp_m.group(1))
        ce_full = float(ce_full_m.group(1))

        # Rate value
        rate_m = RATE_LINE_RE.search(section)
        if not rate_m:
            continue
        unit_from_rate = rate_m.group('unit').strip()
        rate_comp = float(rate_m.group('comp'))

        # Normalize K value
        if k_value == '0':
            k_numeric, k_display = 0, '0'
        elif isinstance(k_value, str) and k_value.endswith('K'):
            k_numeric = int(k_value[:-1]) * 1000
            k_display = k_value
        else:
            k_numeric = int(k_value)
            k_display = k_value

        data.append({
            'k_value': k_display,
            'k_numeric': k_numeric,
            'comp_rate_bits_per_token': rate_comp,
            'CE_full_bits': ce_full,
            'CE_comp_bits': ce_comp,
            'method_detail': f'K{k_display}',
            'rate_unit': unit_from_rate or unit
        })

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('k_numeric').reset_index(drop=True)
    return df


def detect_file_format(filename):
    """Detects the log file format based on its content."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'drop tail' in content:
        return 'drop_tail'
    if re.search(r'length\s*=\s*\d+', content):
        return 'length_values'
    if re.search(r'/K\d+K?/', content) or re.search(r'\bK\s*=\s*\d+\b', content):
        return 'k_values'
    return 'unknown'


def load_and_process_data(filename, label):
    """Loads and processes a single data file, dispatching to the correct parser."""
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found, skipping {label}.")
        return None
    file_format = detect_file_format(filename)
    print(f"Detected format for {label} file: {file_format}")

    if file_format == 'drop_tail':
        df = parse_drop_tail_txt(filename)
        x_column, x_label = 'drop_pct', 'Drop Percentage (%)'
    elif file_format == 'length_values':
        df = parse_length_values_txt(filename)
        x_column, x_label = 'length_value', 'Length Value'
    elif file_format == 'k_values':
        df = parse_k_values_txt(filename)
        x_column, x_label = 'k_numeric', 'K Value'
    else:
        print(f"Error: Could not identify the format of {filename}.")
        return None

    if df.empty:
        print(f"Error: Failed to parse valid data from {filename}.")
        return None

    # Add derived columns
    df["Delta_CE_bits"] = df["CE_comp_bits"] - df["CE_full_bits"]
    df["Delta_CE_pct"] = (df["Delta_CE_bits"] / df["CE_full_bits"]) * 100.0
    df["method"] = label
    df["x_column"] = x_column
    df["x_label"] = x_label

    # Fallback for rate unit if missing
    if "rate_unit" not in df.columns or df["rate_unit"].isna().all():
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        unit = detect_rate_unit_from_text(content, default_unit='bits/token')
        df["rate_unit"] = unit

    print(f"Successfully parsed {len(df)} rows of {label} data from {filename}; Unit: {df['rate_unit'].iloc[0]}")
    return df


def set_matplotlib_style():
    """Sets a consistent, publication-quality style for all plots."""
    plt.rcParams.update({
        "figure.dpi": 220, "savefig.dpi": 220, "font.family": "serif",
        "font.size": 18, "axes.labelsize": 18, "axes.titlesize": 20,
        "legend.fontsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.linewidth": 0.8, "lines.linewidth": 2.6, "lines.markersize": 7.5,
        "figure.autolayout": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "path.simplify": True,
        "path.simplify_threshold": 0.5, "agg.path.chunksize": 8000,
    })


def beautify_axes(ax):
    """Applies common style elements to an axis object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(which='major', alpha=0.28, linewidth=0.8)
    ax.grid(which='minor', alpha=0.10, linewidth=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.tick_params(which="minor", direction="in", length=2.5, width=0.6)


def moving_average(y, window=3):
    """Applies a simple moving average to smooth a curve."""
    if window is None or window <= 1 or window > len(y):
        return np.asarray(y, dtype=float)
    w = int(window)
    if w % 2 == 0: w += 1
    kernel = np.ones(w) / w
    y = np.asarray(y, dtype=float)
    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")

def _pad_ylim(ax, y_series_list, top=0.15, bottom=0.05):
    """Pads the y-axis limits to prevent labels from being cut off."""
    ys = np.concatenate([np.asarray(s, dtype=float) for s in y_series_list if len(s) > 0]) \
        if y_series_list else np.array([])
    if ys.size == 0: return
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    if ymin == ymax:
        pad = (abs(ymax) + 1.0) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        yr = ymax - ymin
        ax.set_ylim(ymin - yr * bottom, ymax + yr * top)

def _dedup_legend(ax):
    """Removes duplicate labels from a legend."""
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen[l] = 1
        new_h.append(h)
        new_l.append(l)
    if new_h:
        ax.legend(new_h, new_l, fontsize=11, loc='best')

# ================================
# Statistics (R0/R5/R50/δ/AUG)
# ================================
def prepare_curve(df, smooth_window=3):
    """Sorts by rate, calculates DeltaCE%, optionally smooths, and deduplicates x by averaging y."""
    df_ = df.sort_values("comp_rate_bits_per_token").copy()
    x = df_["comp_rate_bits_per_token"].to_numpy(dtype=float)
    y = df_["Delta_CE_pct"].to_numpy(dtype=float)
    if smooth_window and smooth_window > 1:
        y = moving_average(y, smooth_window)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    # Deduplicate x values by averaging corresponding y values
    xu, idx = np.unique(x, return_inverse=True)
    yu = np.zeros_like(xu, dtype=float)
    for i, xv in enumerate(xu):
        yu[i] = np.mean(y[idx == i])
    return xu, yu

def threshold_rate(x, y, thresh):
    """Finds the minimum rate R where y <= thresh via linear interpolation."""
    above = y > thresh
    if np.all(above): return None
    idx = np.where(~above)[0][0]
    if idx == 0: return float(x[0])
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    if y1 == y0: return float(x1)
    r = x0 + (thresh - y0) * (x1 - x0) / (y1 - y0)
    return float(r)

def half_degradation_rate(x, y):
    """Finds the minimum rate R where y <= y(min_rate) / 2."""
    if len(y) == 0: return None
    v0 = y[0]
    return threshold_rate(x, y, v0 / 2.0)

def vertical_gap_and_auc(x_tf, y_tf, x_tr, y_tr, num=256):
    """Calculates the vertical gap (delta) and Area Under the Gap (AUG) between two curves."""
    xmin = max(min(x_tf), min(x_tr))
    xmax = min(max(x_tf), max(x_tr))
    if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin <= 0 or xmax <= xmin:
        return None, None, None
    grid = np.exp(np.linspace(np.log(xmin), np.log(xmax), num))
    ytf = np.interp(grid, x_tf, y_tf)
    ytr = np.interp(grid, x_tr, y_tr)
    delta = ytf - ytr
    auc = float(np.trapz(delta, x=np.log(grid)))
    return grid, delta, auc

# =========================
# Rate Savings Calculation
# =========================
def find_rate_savings(x_training, y_training, x_training_free, y_training_free, analysis_points):
    """
    Calculates the rate savings of the training method compared to the training-free
    method at specified target rates.
    """
    savings_info = []
    for target_rate in analysis_points:
        if target_rate < min(x_training) or target_rate > max(x_training):
            print(f"Warning: Target rate {target_rate} is outside the training data range.")
            continue
        training_delta_ce = np.interp(target_rate, x_training, y_training)
        
        # Find the rate for the training-free method to achieve the same Delta CE
        valid_indices = y_training_free <= training_delta_ce
        if not np.any(valid_indices):
            print(f"Warning: At rate {target_rate}, training-free method cannot achieve the same performance.")
            continue
        
        first_valid_idx = np.where(valid_indices)[0][0]
        if first_valid_idx == 0:
            training_free_rate = x_training_free[0]
        else:
            x0, x1 = x_training_free[first_valid_idx-1], x_training_free[first_valid_idx]
            y0, y1 = y_training_free[first_valid_idx-1], y_training_free[first_valid_idx]
            if y1 == y0:
                training_free_rate = x1
            else:
                training_free_rate = x0 + (training_delta_ce - y0) * (x1 - x0) / (y1 - y0)
        
        rate_savings = training_free_rate - target_rate
        savings_pct = (rate_savings / training_free_rate) * 100 if training_free_rate > 0 else 0
        savings_info.append({
            'training_rate': target_rate,
            'training_delta_ce': training_delta_ce,
            'training_free_rate': training_free_rate,
            'rate_savings': rate_savings,
            'savings_pct': savings_pct
        })
    return savings_info


def annotate_rate_savings(ax, x_training, y_training, x_training_free, y_training_free, 
                          analysis_points, training_color='#B22222', training_free_color='#5F6B6D'):
    """Annotates rate savings on the plot with horizontal lines and labels."""
    savings_info = find_rate_savings(x_training, y_training, x_training_free, y_training_free, analysis_points)
    if not savings_info: return savings_info
    
    for info in savings_info:
        tr, tc, tfr = info['training_rate'], info['training_delta_ce'], info['training_free_rate']
        rs, sp = info['rate_savings'], info['savings_pct']
        
        ax.plot([tr, tfr], [tc, tc], color='red', linewidth=2.0, alpha=0.75, zorder=10)
        ax.plot(tr, tc, 'o', color=training_color, markersize=9, mfc='white', mew=2.0, zorder=11)
        ax.plot(tfr, tc, 's', color=training_free_color, markersize=9, mfc='white', mew=2.0, zorder=11)
        
        mid_x = (tr + tfr) / 2
        ax.annotate(f'Save {rs:.1f}\n({sp:.1f}%)',
                    xy=(mid_x, tc), xytext=(mid_x, tc - 20),
                    ha='center', va='bottom', fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    print("\n=== Rate Savings Analysis ===")
    for info in savings_info:
        print(f"At target rate={info['training_rate']:.1f}:")
        print(f"  Training method Delta_CE = {info['training_delta_ce']:.2f}%")
        print(f"  Training-free method needs rate = {info['training_free_rate']:.1f} to match")
        print(f"  Rate Saved: {info['rate_savings']:.1f} ({info['savings_pct']:.1f}%)\n")
    return savings_info

# ========================
# File & Plot Management
# ========================
def create_output_dir(directory):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")

def save_fig(base_path_and_name, transparent_export=True):
    """Saves the current figure in multiple formats."""
    for ext in ("png", "pdf", "svg"):
        filepath = f"{base_path_and_name}.{ext}"
        plt.savefig(filepath, bbox_inches="tight", dpi=220)
    if transparent_export:
        filepath = f"{base_path_and_name}_transparent.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=220, transparent=True)

# =========================
# Main Plotting Function
# =========================
def _get_rate_unit(df1, df2, fallback='bits/token'):
    """Extracts the rate unit from dataframes, preferring the second one."""
    for df in [df2, df1]:
        if df is not None and 'rate_unit' in df.columns and not df['rate_unit'].isna().all():
            return str(df['rate_unit'].iloc[0]).strip()
    return fallback

def plot_comparison(df_training_free, df_training, save_figures=True,
                    smooth_window=SMOOTH_WINDOW,
                    transparent_export=True,
                    rate_analysis_points=RATE_ANALYSIS_POINTS):

    # Colorblind-friendly + arXiv-style palette
    colors = {'Training Free': '#5F6B6D', 'Training': '#B22222'}
    markers = {'Training Free': 'o', 'Training': 's'}
    linestyles = {'Training Free': '-', 'Training': '-'}

    rate_unit = _get_rate_unit(df_training_free, df_training, fallback='bits/token')

    # ---------- Plot: Rate vs Relative CE increase (%) + Vertical Gap + Rate Savings ----------
    fig, ax1 = plt.subplots(figsize=(12, 9))
    beautify_axes(ax1)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:,.0f}"))
    ax1.axhline(0, color="#666666", linewidth=1.0, alpha=0.6)

    y_for_padding_left = []
    x_tf = y_tf = x_tr = y_tr = None

    # Left Axis: Delta CE% curves for Training Free & Training
    if df_training_free is not None:
        x, y_raw = df_training_free["comp_rate_bits_per_token"].values, df_training_free["Delta_CE_pct"].values
        y = moving_average(y_raw, smooth_window)
        markevery = max(1, len(x) // MARKER_TARGET)
        line1, = ax1.plot(x, y, label="Training Free", color=colors['Training Free'],
                          marker=markers['Training Free'], linestyle=linestyles['Training Free'],
                          linewidth=2.6, markersize=7.5, mfc="white", mew=1.1,
                          solid_capstyle="round", markevery=markevery, zorder=3)
        line1.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
        y_for_padding_left.append(y)
        x_tf, y_tf = prepare_curve(df_training_free, smooth_window=smooth_window)

    if df_training is not None:
        x, y_raw = df_training["comp_rate_bits_per_token"].values, df_training["Delta_CE_pct"].values
        y = moving_average(y_raw, smooth_window)
        markevery = max(1, len(x) // MARKER_TARGET)
        line2, = ax1.plot(x, y, label="Training", color=colors['Training'],
                          marker=markers['Training'], linestyle=linestyles['Training'],
                          linewidth=2.6, markersize=7.5, mfc="white", mew=1.1,
                          solid_capstyle="round", markevery=markevery, zorder=3)
        line2.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
        y_for_padding_left.append(y)
        x_tr, y_tr = prepare_curve(df_training, smooth_window=smooth_window)

    # --- Vertical Gap shading + AUG ---
    if (x_tf is not None) and (x_tr is not None):
        xmin = max(min(x_tf), min(x_tr))
        xmax = min(max(x_tf), max(x_tr))
        if np.isfinite(xmin) and np.isfinite(xmax) and xmin > 0 and xmax > xmin:
            grid = np.exp(np.linspace(np.log(xmin), np.log(xmax), 512))
            ytf, ytr = np.interp(grid, x_tf, y_tf), np.interp(grid, x_tr, y_tr)
            delta = ytf - ytr

            # Shade the area between curves
            ax1.fill_between(grid, ytf, ytr, where=(delta >= 0), interpolate=True, alpha=0.12,
                             color="#C84E3A", label="Vertical gap δ>0 (TF worse)")
            ax1.fill_between(grid, ytf, ytr, where=(delta < 0), interpolate=True, alpha=0.12,
                             color="#2C7BB6", label="Vertical gap δ<0 (Training worse)")

            # AUG (∫δ d log R)
            aug = float(np.trapz(delta, x=np.log(grid)))
            ax1.text(0.02, 0.98, f"AUG = {aug:.3f}", transform=ax1.transAxes, ha="left", va="top",
                     fontsize=11, color="#444",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.9))

            # --- Horizontal Rate Savings ---
            annotate_rate_savings(ax1, x_tr, y_tr, x_tf, y_tf, rate_analysis_points,
                                  training_color=colors['Training'], training_free_color=colors['Training Free'])

    ax1.set_xlabel(f"Rate ({rate_unit}, comp)")
    ax1.set_ylabel("Relative CE increase (%)")
    ax1.set_title("Rate vs CE Increase · Training Free vs Training")

    if y_for_padding_left:
        _pad_ylim(ax1, y_for_padding_left, top=0.30, bottom=0.10)

    _dedup_legend(ax1)
    plt.tight_layout()
    if save_figures:
        output_filepath = os.path.join(OUTPUT_DIR, "relative_ce_curve")
        save_fig(output_filepath, transparent_export)

    print(f"\nPlot saved successfully to directory: {OUTPUT_DIR}")

# =========================
# Command-Line Interface
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Rate-vs-CE comparison plots from log files."
    )
    parser.add_argument(
        "--training-free-log",
        default=r'dataset_and_results\relative_ce_result\eval_untrained.log',
        type=str,
        help="Path to the log file for the 'Training Free' method. Example: 'logs/eval_untrained.log'"
    )
    parser.add_argument(
        "--training-log",
        default=r'dataset_and_results\relative_ce_result\eval_trained.log',
        type=str,
        help="Path to the log file for the 'Training' method. Example: 'logs/eval_trained.log'"
    )
    args = parser.parse_args()

    set_matplotlib_style()
    create_output_dir(OUTPUT_DIR)

    print("=== Loading Training Free data ===")
    df_training_free = load_and_process_data(args.training_free_log, "Training Free")

    print("\n=== Loading Training data ===")
    df_training = load_and_process_data(args.training_log, "Training")

    if df_training_free is None and df_training is None:
        print("Error: No data files were loaded successfully. Exiting.")
        return

    # Save combined data to a CSV file
    dfs_to_save = [d for d in [df_training_free, df_training] if d is not None]
    if dfs_to_save:
        combined_df = pd.concat(dfs_to_save, ignore_index=True)
        output_csv = os.path.join(OUTPUT_DIR, "comparison_data.csv")
        combined_df.round(4).to_csv(output_csv, index=False)
        print(f"\nCombined data saved to: {output_csv}")

    # Calculate and print statistics
    if df_training is not None:
        x_tr, y_tr = prepare_curve(df_training, smooth_window=SMOOTH_WINDOW)
        R0_tr = threshold_rate(x_tr, y_tr, 0.0)
        R5_tr = threshold_rate(x_tr, y_tr, 5.0)
        R50_tr = half_degradation_rate(x_tr, y_tr)
        unit = _get_rate_unit(None, df_training)
        print("\n=== Summary statistics (Training) ===")
        print(f"Break-even R0 : {R0_tr:.6g} ({unit})" if R0_tr is not None else "Break-even R0 : N/A")
        print(f"5% tol.    R5  : {R5_tr:.6g} ({unit})" if R5_tr is not None else "5% tol.    R5  : N/A")
        print(f"Half-deg.  R50 : {R50_tr:.6g} ({unit})" if R50_tr is not None else "Half-deg.  R50 : N/A")

    if (df_training_free is not None) and (df_training is not None):
        x_tf, y_tf = prepare_curve(df_training_free, smooth_window=SMOOTH_WINDOW)
        x_tr, y_tr = prepare_curve(df_training, smooth_window=SMOOTH_WINDOW)
        grid, delta, auc = vertical_gap_and_auc(x_tf, y_tf, x_tr, y_tr, num=512)
        if grid is not None:
            print(f"AUG (∫δ d log R): {auc:.6g}" if auc is not None else "AUG: N/A")

    print("\n=== Generating comparison plot ===")
    plot_comparison(
        df_training_free, df_training,
        save_figures=True,
        smooth_window=SMOOTH_WINDOW,
        transparent_export=True,
        rate_analysis_points=RATE_ANALYSIS_POINTS
    )

    print(f"\nProcess complete! Files are saved in '{OUTPUT_DIR}'.")
    print(" - relative_ce_curve.{png,pdf,svg}")
    print(" - relative_ce_curve_transparent.png")
    print(" - comparison_data.csv")

if __name__ == "__main__":
    main()