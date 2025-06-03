import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load data.csv
df = pd.read_csv("data.csv")

# Extract time and height signal
t = np.array(df["Time"].values)
signal = np.array(df["Height"].values)

# Estimate sampling frequency
# fs = 1 / np.mean(np.diff(t))
fs = 3
print(fs)

# Butterworth filter design
def butter_lowpass(cutoff, fs, order=8):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff, fs, order=8):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# Apply filter to Height
cutoff = 0.05  # Hz
filtered_height = apply_filter(signal, cutoff, fs)
df["Filtered Height"] = filtered_height

# Compute Wicking Rate using cubic fit on filtered height
filtered_height = apply_filter(signal, cutoff=0.1, fs=fs)

# Cubic spline derivative
spline = CubicSpline(t, filtered_height)
wicking_rate = np.abs(spline.derivative()(t))

df["Filtered Wicking Rate"] = wicking_rate

plt.figure(figsize=(10, 6))

# Plot raw wicking rate (from CSV)
if "Wicking Rate" in df.columns:
    plt.plot(df["Time"], df["Wicking Rate"], label="Raw Wicking Rate", color='gray', linestyle='--', alpha=0.6)

# Plot filtered wicking rate (spline derivative)
plt.plot(df["Time"], df["Filtered Wicking Rate"], label="Filtered Wicking Rate", color='orange')

plt.xlabel("Time [s]")
plt.ylabel("Wicking Rate (mm/s)")
plt.title("Raw vs Filtered Wicking Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Height"], label="Raw Height", color='gray', alpha=0.6)
plt.plot(df["Time"], df["Filtered Height"], label="Filtered Height", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Height (mm)")
plt.title("Raw vs Filtered Height")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()