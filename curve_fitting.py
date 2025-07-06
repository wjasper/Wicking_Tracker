#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from scipy.linalg import lstsq
from scipy.optimize import curve_fit

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

def wicking_rate(t, H, tau, A):
    return H/tau*np.exp(-t/tau) + A/(2*np.sqrt(t))

def model_f(t, H, tau, A):
    return H*(1 - np.exp(-t/tau)) + A*np.sqrt(t)

def model_g(t, A, B, C, D):
    return  A + B*t + C*t**2  +D*t**3

# 1) Load data
# df = pd.read_csv("data.csv")
df = pd.read_csv("/home/pi/opencv/Wicking_Tracker/output/AW Trail 10_20250626_191938/data.csv")

t_data = df["Time"].to_numpy()   # converted to numpy array
h_data = df["Height"].to_numpy() # converted to numpy array

popt,pcov = curve_fit(model_f, t_data, h_data, p0=[31, 9, 4])

print("popt =", popt)
print("pcov =", pcov)
H_opt, tau_opt, A_opt = popt

t_model = np.linspace(min(t_data), max(t_data), 100)
h_model = model_f(t_model, H_opt, tau_opt, A_opt)

h_rate_model = wicking_rate(t_model, H_opt, tau_opt, A_opt)

#plot data
plt.scatter(t_data, h_data)
plt.plot(t_model, h_model, color='r')
plt.xlabel("Time [s]")
plt.ylabel("Height (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.imshow(np.log(np.abs(pcov)))
plt.colorbar()
plt.show()

plt.plot(t_model, h_rate_model, color='r')
plt.xlabel("Time [s]")
plt.ylabel("Wicking Rate (mm/s)")
plt.title("Wicking Rate")
plt.grid(True)
plt.tight_layout()
plt.show()




