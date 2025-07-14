#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
Refactored: SOS filtering applied inside a standalone function.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def post_process_wicking_rate(df, show_plots=True):

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

    # 5) Plot: Heights
    from io import BytesIO
    from PIL import Image

    buf_height = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(t_data, h_data, label="Raw Height", alpha=0.4)
    plt.plot(t_model, h_model, label="Modeled", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Height (mm)")
    plt.title("Height: Raw and Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf_height, format="png")
    buf_height.seek(0)
    height_plot_image = Image.open(buf_height)

    # 6) Plot: Wicking Rate
    buf_wick = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(t_model, h_rate_model, label="wicking_rate", color="red", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Wicking Rate (mm/s)")
    plt.title("Wicking Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf_wick, format="png")
    buf_wick.seek(0)
    wicking_plot_image = Image.open(buf_wick)

    if show_plots:
        plt.show()

    return df, height_plot_image, wicking_plot_image
