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

# Plot Wicking Rate
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Filtered Wicking Rate"], label="Filtered Wicking Rate", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Wicking Rate (mm/s)")
plt.title("Wicking Rate from Filtered Height")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Filtered Height"], label="Filtered Height", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Filtered Height mms")
plt.title("Filtered Height")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Height"], label="Height", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Height mms")
plt.title("Height")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()