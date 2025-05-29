import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_final_wicking_rate(csv_path, apply_smoothing=True):
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if not all(col in df.columns for col in ["Time", "Height"]):
        print("CSV is missing required columns.")
        return

    # Compute Wicking Rate from Time-Height data using cubic polynomial
    rate_list = [0] * len(df)
    for i in range(3, len(df)):
        t_window = df["Time"].iloc[i-3:i+1].values
        h_window = df["Height"].iloc[i-3:i+1].values
        coeffs = np.polyfit(t_window, h_window, 3)
        a, b, c, _ = coeffs
        t_latest = t_window[-1]
        rate = 3 * a * t_latest**2 + 2 * b * t_latest + c
        rate_list[i] = abs(rate)

    df["Computed Wicking Rate"] = rate_list

    # Plot
    plt.figure("Final Wicking Rate Plot")
    plt.plot(df["Time"], df["Computed Wicking Rate"], label="Wicking Rate (computed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Wicking Rate (mm/s)")
    plt.title("Wicking Rate Over Time (Final Plot)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df
