# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:22:08 2019

@author: Chenghai Li
"""

import math
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

file = open('pr.txt','r')

p = []
r = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    p.append(row[0])
    r.append(row[1])
    
file.close()
    
pr = interpolate.interp1d(p, r, kind="cubic")

file = open('2_1.txt','r')

x2_1 = []
y2_1 = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    x2_1.append(row[0])
    y2_1.append(row[1])
    
file.close()

l2_1 = interpolate.interp1d(x2_1, y2_1, kind="cubic")

file = open('2_2.txt','r')

x2_2 = []
y2_2 = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    x2_2.append(row[0])
    y2_2.append(row[1])
    
file.close()

l2_2 = interpolate.interp1d(x2_2, y2_2, kind="cubic")

def l(x):
    
    x = (x -12.5) % 100
    if x < 0.45:
        return l2_1(x)
    if 0.45 <= x <= 2:
        return 2
    if 2 < x < 2.45:
        return l2_2(x)
    if 2.45 <= x <= 100:
        return 0
    
A = 0.7*0.7*math.pi
v = 500*5*5*math.pi

omiga = math.pi/111

def v0(t):
    
    return  -(-2.413 * math.sin( omiga * t + math.pi/2 ) + 4.826) * math.pi * 2.5 * 2.5 + 162.1374326208532

def v0t(t):
    
    return  omiga * 2.413 * math.cos ( omiga * t + math.pi/2 ) * math.pi * 2.5 * 2.5


def func(w, t, step):

    p0, p1= w
    
    tr = t % 100
            
    if p0 >= p1 and v0t(t) < 0:
        
        Q0 = 0.85 * A * math.sqrt(2 * ( p0 - p1 ) / pr(p0) ) 
 
    else:   
        
        Q0 = 0

    if p0 < 0.5 and v0t(t) > 0: 
        
        r0t = 0
        p0t = 0
        
    else:
    
        r0t = (-Q0 * pr(p0) * v0(t) - v0(t) * pr(p0) * v0t(t)) / (v0(t) ** 2)
        p0t = E(p0) / pr(p0) * r0t
    
 
    A1 = math.pi * l(tr) * math.sin( math.pi / 20 ) * (4 * 1.25 + l(tr) * math.sin( math.pi / 20 ) * math.cos( math.pi / 20 ) )
    A2 = math.pi * 0.7 * 0.7
    
    Q1 =   0.85 * min(A1, A2) * math.sqrt(2 * p1 / pr(p1))
    
    r1t = (Q0 * pr(p0) - Q1 * pr(p1)) / v
    p1t = E(p1) / pr(p1) * r1t

    return np.array([p0 + p0t * step, p1 + p1t * step])

time_range = 1000
step_length = 0.01

t = np.arange(0, time_range, step_length)

out = []
w = np.array([0.5, 100])

for i in range (time_range / step_length):
    
    w = func(w, t[i], 0.01)
    out.append(w)


show = []

for i in range(len(out)):
    show.append(out[i][1])
    
plt.plot(show)


