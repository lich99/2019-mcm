# -*- encoding: utf-8 -*-

"""
Created on Sat Sep 14 20:43:12 2019

@author: Chenghai Li
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import interpolate

file = open('3.txt','r')

x3 = []
y3 = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    x3.append(row[0])
    y3.append(row[1])
    
file.close()
    
E = interpolate.interp1d(x3, y3, kind="cubic")

def func(w, p):

    r = w[0]
   
    rp = r / E(p)
    
    return np.array([rp])


t = np.arange(100, 200, 0.01) 
track1 = odeint(func, (0.85), t, hmax=0.001)
plt.xlabel("P-Pressure  (MPa)")
plt.ylabel("ρ-Density  (mg/mm³)")
plt.plot(t,track1) 
