#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import BytesIO
from PIL import Image
from scipy.optimize import curve_fit

def wicking_rate(t, H, tau, A):
    rate = []
    for i in range(len(t)):
        if t[i] != 0:
            rate.append(H/tau*np.exp(-t[i]/tau) + A/(2*np.sqrt(t[i])))
        else:   
            rate.append(0.0)
    rate = np.array(rate)
    return rate

def model_f(t, H, tau, A):
    return H*(1 - np.exp(-t/tau)) + A*np.sqrt(t)

def post_process_wicking_rate(df,
                            show_plots=True,
                            height_yticks_spacing=10,
                            wicking_yticks_spacing=0.5):

    # 1) Extract data
    t_data = df["Time"].values
    h_data = df["Height"].values

    if not np.all(np.diff(t_data) > 0):
        raise ValueError("Time vector must be strictly increasing")

    # 2) Generate nonliner model of height
    popt,pcov = curve_fit(model_f, t_data, h_data, p0=[31, 9, 4])
    H_opt, tau_opt, A_opt = popt
    
    #3) Generate wicking Rate
    t_model = np.linspace(0, max(t_data), len(t_data))
    h_model = model_f(t_model, H_opt, tau_opt, A_opt)
    h_rate_model = wicking_rate(t_model, H_opt, tau_opt, A_opt)

    # 4) Store models
    df["Time_Uniform"] = t_model
    df["Height_Model"] = h_model
    df["Wicking_Rate"] = h_rate_model
    df["Modeled Avg Wicking Rate"] = df.apply(lambda row: row["Height_Model"] / row["Time_Uniform"] if row["Time_Uniform"] > 0 else 0,axis=1)
    
    # 5) Plot: Heights
    from io import BytesIO
    from PIL import Image

    buf_height = BytesIO()
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t_data, h_data, label="Raw Height", alpha=0.4)
    ax1.plot(t_model, h_model, label="Modeled", linewidth=2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Height (mm)")
    ax1.set_title("Height: Raw and Model")
    ax1.legend()
    ax1.grid(True)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(height_yticks_spacing))
    fig1.tight_layout()
    fig1.savefig(buf_height, format="png")
    buf_height.seek(0)
    height_plot_image = Image.open(buf_height)

    # 6) Plot: Wicking Rate
    buf_wick = BytesIO()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t_model, h_rate_model, label="Wicking Rate", color="red", alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Wicking Rate (mm/s)")
    ax2.set_title("Wicking Rate")
    ax2.legend()
    ax2.grid(True)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(wicking_yticks_spacing))
    fig2.tight_layout()
    fig2.savefig(buf_wick, format="png")
    buf_wick.seek(0)
    wicking_plot_image = Image.open(buf_wick)

    if show_plots:
        plt.show()

    return df, height_plot_image, wicking_plot_image
