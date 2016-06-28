# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:47:12 2016

@author: au194693
"""
import numpy as np
import matplotlib.pyplot as plt

def calc_ISPC_time(data):
    result = np.empty([data.shape[0]])

    for i in range(data.shape[0]):
        result[i] = np.abs(np.mean(np.exp(
            1j*(np.angle(data[i, 52, :]) -
            np.angle(data[i, 71, :])))))

    return result
    

def plot_correlation(data):
    x = np.arange(0, len(data), 1)
    plt.figure()
    plt.plot(x, data, "bo")
    
    a, b = np.polyfit(x, data, 1)
    plt.plot(x, x*a + b)
    print(a, b)