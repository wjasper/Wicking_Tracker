#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from scipy.linalg import lstsq

def derivative(t, h, npoints):
    slope = []
    #build the design matrix M
    M = np.column_stack((np.ones_like(t), t, t*t, t*t*t))

    M_subset = M[0:npoints]
    h_subset = h[0:npoints]
    a, res, rnk, s = lstsq(M_subset,h_subset)
    for i in range(npoints//2):
        slope.append(a[1] + 2*a[2]*t[i] + 3*a[3]*t[i]**2)
            
    for i in range(t.size - npoints):
        j = i + npoints//2
        M_subset = M[i:i+npoints]
        h_subset = h[i:i+npoints]
        a, res, rnk, s = lstsq(M_subset,h_subset)
        slope.append(a[1] + 2*a[2]*t[j] + 3*a[3]*t[j]**2)

    M_subset = M[t.size-npoints:t.size]
    h_subset = h[t.size-npoints:t.size]
    a, res, rnk, s = lstsq(M_subset,h_subset)
    for i in range(t.size - npoints//2, t.size):
         slope.append(a[1] + 2*a[2]*t[i] + 3*a[3]*t[i]**2)

    return slope

# 1) Load data
# df = pd.read_csv("data.csv")
df = pd.read_csv("/home/pi/opencv/Wicking_Tracker/output/AW Trail 10_20250626_191938/data.csv")
t = df["Time"].to_numpy()   # converted to numpy array
h = df["Height"].to_numpy()  # converted to numpy array
raw_rate = df["Wicking Rate"].to_numpy()
npoints = 128

# 2) Ensure monotonic time
if not np.all(np.diff(t) > 0):
    raise ValueError("Time vector must be strictly increasing")

# 3) Resample to a perfectly uniform grid
t_uniform = np.linspace(t.min(), t.max(), len(t))
h_uniform = np.interp(t_uniform, t, h)

print(t_uniform, h_uniform)

# 4) Butterworth filter definition
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    Wn  = cutoff / nyq
    return butter(order, Wn, btype='low', analog=False)

def apply_butterworth_sos(data, cutoff, fs, order=6):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff / (0.5 * fs), btype='low', output='sos')
    return sosfiltfilt(sos, data)

# Estimate sampling frequency and apply filter
fs = 1 / np.mean(np.diff(t_uniform))
cutoff = 0.25  # Hz
order = 4

# Filtered signals
h_filtered = apply_butterworth_sos(h_uniform, cutoff, fs, order)

# 5) Derivative Splines
cs_raw      = CubicSpline(t, h)
cs_interp   = CubicSpline(t_uniform, h_uniform)
cs_filtered = CubicSpline(t_uniform, h_filtered)

wicking_rate_raw      = np.abs(cs_raw.derivative()(t))
wicking_rate_interp   = np.abs(cs_interp.derivative()(t_uniform))
wicking_rate_filtered = np.abs(cs_filtered.derivative()(t_uniform))

wickingRate_2 = derivative(t_uniform, h_filtered, npoints)


# 6) Store
df["Time_Uniform"] = t_uniform
df["Filtered Height (Uniform)"] = h_uniform
df["Filtered Height (Raw)"] = h_filtered
df["Wicking Rate (Spline)"] = wicking_rate_raw
df["Wicking Rate Filtered (Spline)"] = wicking_rate_filtered
df["Wicking Rate 2"] = wickingRate_2

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
plt.show()

# 8) Plot: Wicking Rate
buf_wick = BytesIO()
plt.figure(figsize=(12, 6))
plt.plot(t_uniform, wickingRate_2, label="wicking_rate_filtered_2", color="red", alpha=0.8)
plt.xlabel("Time [s]")
plt.ylabel("Wicking Rate (mm/s)")
plt.title("Raw vs Smoothed Wicking Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(buf_wick, format="png")
buf_wick.seek(0)
wicking_plot_image = Image.open(buf_wick)
    
plt.show()

