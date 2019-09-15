# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 09:25:40 2019
@author: Chenghai Li
"""

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import interpolate

file = open('3.txt','r')

x = []
y = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    x.append(row[0])
    y.append(row[1])
    
E=interpolate.interp1d(x,y,kind="cubic")

'''
xnew=np.linspace(0,200,1001)
ynew=E(xnew)
plt.plot(x,y,c = 'r')
'''

A = 0.7*0.7*math.pi
v = 500*5*5*math.pi

test_t = 3.5
step = 0.001
batch_time = 1

#t = [0, t1, t1+2.4, t2]

t1 = [0.233]
t2 = []

def Q(r, p, t):
    
    time = int(t // 100) 
    
    remain = t % 100
    
    if 0 <= remain <= t1[time]:
        
        if p > 160:
            return 0
        
        return 0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.87112238 )
    
    if t1[time] < remain <= t1[time] + 0.2:
        
        if p > 160:
            return 0
        
        return 0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.87112238 ) - 100 * (remain - t1[time])
    
    if t1[time] + 0.2 < remain <= t1[time] + 2.2:
        
        if p > 160:
            return 0
        
        return 0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.87112238 ) - 20
    
    if t1[time] + 2.2 < remain <= t1[time] + 2.4:
        
        if p > 160:
            return 0
        
        return 0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.87112238 ) - 100 * (t1[time] + 2.4 - remain)
    
    if t1[time] + 2.4 < remain:
        
        if p > 160:
            return 0
        
        return 0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.87112238 )

    
def func(w, t):

    r, p= w
    time = int(t // 100)
    remain = t % 100 
    
    if len(t2) == time + 1:
        return np.array([0, 0])
    
    if remain > t1[time]+2.4 and p >= 100:
 
        t2.append(remain)
        return np.array([0, 0])
    
    rt = (1 / v) * Q(r, p, t) * r
    pt = E(p) / r * rt
    
    return np.array([rt, pt])


t = np.arange(0, test_t, step) 
track1 = odeint(func, (0.85, 100), t, hmax=0.001 )



out = []
for obj in track1:
    out.append(obj[1])

#print(out[-1]-out[0])
sort = sorted(out)
#print(sort[-1]-out[0],sort[1]-out[0])
print('lmax:',sort[-1]-out[0]+sort[1]-out[0])

cut = int(t2[0] / step)+2
l = np.array(out[0:cut])
mean = l.mean()-100
l1 = (abs(l-100)).mean()
l2 = ((l-100)**2).mean()

print('mean:',mean,'\n','l1:',l1,'\n','l2:',l2)


plt.hlines(100, 0, test_t, colors = "c", linestyles = "dashed")
#plt.hlines(150, 0, int(test_t/step), colors = "c", linestyles = "dashed")

plt.ylabel("P-Pressure  (MPa)",fontproperties="STSong", fontsize = 14)
plt.xlabel("T-Time  (ms)",fontproperties="STSong", fontsize = 14)

plt.plot(t,out)
plt.savefig(r"ρ-P image.png", dpi = 300)
