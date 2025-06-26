#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
Refactored: SOS filtering applied inside a standalone function.
"""

def post_process_wicking_rate(df, show_plots=True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, sosfiltfilt
    from scipy.interpolate import CubicSpline
    from io import BytesIO
    from PIL import Image

    # 1) Extract data
    t = df["Time"].values
    h = df["Height"].values

    if not np.all(np.diff(t) > 0):
        raise ValueError("Time vector must be strictly increasing")

    # 2) Uniform interpolation
    t_uniform = np.linspace(t.min(), t.max(), len(t))
    h_uniform = np.interp(t_uniform, t, h)

    # 3) SOS Butterworth
    def apply_butterworth_sos(data, cutoff, fs, order=6):
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, cutoff / (0.5 * fs), btype='low', output='sos')
        return sosfiltfilt(sos, data)

    # 4) Filtering params
    fs = 1 / np.mean(np.diff(t_uniform))
    cutoff = 0.25  # Hz
    order = 4

    h_filtered = apply_butterworth_sos(h_uniform, cutoff, fs, order)

    # 5) Derivative Splines
    cs_raw      = CubicSpline(t, h)
    cs_interp   = CubicSpline(t_uniform, h_uniform)
    cs_filtered = CubicSpline(t_uniform, h_filtered)

    wicking_rate_raw      = np.abs(cs_raw.derivative()(t))
    wicking_rate_interp   = np.abs(cs_interp.derivative()(t_uniform))
    wicking_rate_filtered = np.abs(cs_filtered.derivative()(t_uniform))

    # 6) Store
    df["Time_Uniform"] = t_uniform
    df["Filtered Height (Uniform)"] = h_uniform
    df["Filtered Height (Raw)"] = h_filtered
    df["Wicking Rate (Spline)"] = wicking_rate_raw
    df["Wicking Rate Filtered (Spline)"] = wicking_rate_filtered

    # 7) Plot: Heights
    from io import BytesIO
    from PIL import Image

    buf_height = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(t, h, label="Raw Height", alpha=0.4)
    plt.plot(t_uniform, h_uniform, label="Interpolated", linewidth=2)
    plt.plot(t_uniform, h_filtered, label="Filtered Height", linewidth=2, linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Height (mm)")
    plt.title("Height: Raw vs Filtered (Raw & Uniform)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf_height, format="png")
    buf_height.seek(0)
    height_plot_image = Image.open(buf_height)

    # 8) Plot: Wicking Rate
    buf_wick = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(t_uniform, wicking_rate_filtered, label="wicking_rate_filtered", color="red", alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Wicking Rate (mm/s)")
    plt.title("Raw vs Smoothed Wicking Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf_wick, format="png")
    buf_wick.seek(0)
    wicking_plot_image = Image.open(buf_wick)

    if show_plots:
        plt.show()

    return df, height_plot_image, wicking_plot_image
