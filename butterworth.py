import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

# 1) Load data
df = pd.read_csv("data.csv")
t = df["Time"].values
h = df["Height"].values
raw_rate = df["Wicking Rate"].values

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

def apply_butterworth(data, cutoff, fs, order=6, cascade=2):
    b, a = butter_lowpass(cutoff, fs, order)
    y = data.copy()
    for _ in range(cascade):
        y = filtfilt(b, a, y)
    return y

# 5) Estimate sampling frequency and apply filter
fs = 1 / np.mean(np.diff(t_uniform))
cutoff = 0.07  # Hz
order = 8
cascade = 2

# Filtered signals
# h_uniform = apply_butterworth(h_uniform, cutoff, fs, order, cascade)
h_filtered= apply_butterworth(h, cutoff, fs, order, cascade)

# 6) Wicking Rate (Spline) from filtered_uniform
cs1 = CubicSpline(t_uniform, h_uniform)
cs2 = CubicSpline(t_uniform, h_filtered)
cs3 = CubicSpline(t, h)

wicking_rate_uniform = np.abs(cs1.derivative()(t_uniform))
wicking_rate_filtered = np.abs(cs2.derivative()(t_uniform))
wicking_rate_spline = np.abs(cs3.derivative()(t))

# 7) Store in DataFrame
df["Time_Uniform"] = t_uniform
df["Filtered Height (Uniform)"] = h_uniform
df["Filtered Height (Raw)"] = h_filtered
df["Wicking Rate (Spline)"] = wicking_rate_spline

# 8) Plot Height Comparisons
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

# 9) Plot Wicking Rate: Raw vs Smoothed
plt.figure(figsize=(12, 6))
plt.plot(t, wicking_rate_spline, label="wicking_rate_spline", color="black", linestyle="--", alpha=0.5)
plt.plot(t_uniform, wicking_rate_filtered, label="wicking_rate_filtered", color="red", alpha=0.8)
# plt.plot(t_uniform, wicking_rate_uniform, label="wicking_rate_uniform", color="blue", alpha=0.8)
plt.xlabel("Time [s]")
plt.ylabel("Wicking Rate (mm/s)")
plt.title("Raw vs Smoothed Wicking Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
