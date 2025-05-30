import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import CubicSpline

# 1) Load data
df = pd.read_csv("data.csv")
t = df["Time"].values
h = df["Height"].values

# 2) Ensure monotonic time
if not np.all(np.diff(t) > 0):
    raise ValueError("Time vector must be strictly increasing")

# 3) Resample to a perfectly uniform grid
#np.linspace(start, stop, num_points)
#np.interp(new_times, original_times, original_heights)

t_uniform = np.linspace(t.min(), t.max(), len(t))
h_uniform = np.interp(t_uniform, t, h)

# 4) Drop any isolated outliers with a small median filter
#    kernel_size must be odd; 5 is usually a good starting point
h_med = medfilt(h_uniform, kernel_size=5)

# 5) Butterworth design + cascade
def butter_lowpass(cutoff, fs, order=8):
    nyq = 0.5 * fs
    Wn  = cutoff / nyq
    return butter(order, Wn, btype='low', analog=False)

def apply_butterworth(data, cutoff, fs, order=8, cascade=2):
    b, a = butter_lowpass(cutoff, fs, order)
    y = data.copy()
    for _ in range(cascade):
        y = filtfilt(b, a, y)
    return y

# estimate sampling frequency on the uniform grid
fs = 1 / np.mean(np.diff(t_uniform))

# tune these to taste
cutoff  = 0.05   # Hz, lower → smoother
order   = 8
cascade = 2

h_filt = apply_butterworth(h_med, cutoff, fs, order, cascade)

# 6) Compute the rate using both methods
# Method 1: Simple gradient
wicking_rate_gradient = np.abs(np.gradient(h_filt, t_uniform))

# Method 2: Cubic spline
cs = CubicSpline(t_uniform, h_filt)
wicking_rate_spline = np.abs(cs.derivative()(t_uniform))

# 7) Put results back into the DataFrame (for convenience)
df["Time_Uniform"] = t_uniform
df["Filtered Height"] = h_filt
df["Wicking Rate (Gradient)"] = wicking_rate_gradient
df["Wicking Rate (Spline)"] = wicking_rate_spline

# 8) Plot to compare
# Figure 1: Height comparison
plt.figure(figsize=(12, 6))
plt.plot(t, h, label="raw height", alpha=0.5)
plt.plot(t_uniform, h_filt, label="median→butterworth", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Height (mm)")
plt.title("Height vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Figure 2: Wicking rate comparison
plt.figure(figsize=(12, 6))
plt.plot(t_uniform, wicking_rate_gradient, label="Gradient Method", color="C1", alpha=0.7)
plt.plot(t_uniform, wicking_rate_spline, label="Cubic Spline Method", color="C2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Rate (mm/s)")
plt.title("Wicking Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
