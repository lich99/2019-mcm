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

p1 = np.arange(100, 200, 0.01) 
r1 = odeint(func, (0.85), p1, hmax=0.001).reshape((10000))

p2 = np.arange(100, 0, -0.01) 
r2 = odeint(func, (0.85), p2, hmax=0.001).reshape((10000))

p2 = p2[::-1]
r2 = r2[::-1]

p = np.hstack((p2,p1))
r = np.hstack((r2,r1))

plt.xlabel("P-Pressure  (MPa)",fontproperties="STSong")
plt.ylabel("ρ-Density  (mg/mm³)",fontproperties="STSong")

l1, = plt.plot(p, r, linewidth=1.0) 
plt.legend(handles=[l1], labels=['ρ-P'], fontsize = 10, loc='lower right')

#plt.savefig(r"ρ-P image.png", dpi = 300)
