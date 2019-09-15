# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:24:10 2019

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
tl = [0.76,0.46,0.233,0.05]
A = 0.7*0.7*math.pi
v = 500*5*5*math.pi

test_t = 5
step = 0.01
batch_time = 1

#t = [0, t1, t1+2.4, t2]

t1 = [0.233]
t2 = [2.866]
t2_true = [2.831]

c = 0
def Q(r, p, t):
    global c,tl
    
    t1 = [tl[c]]
  
    
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
    '''
    if t2[0] < remain:
        
        return 0
    '''
    
def func(w, t):

    r, p= w
    
    time = int(t // 100)
    remain = t % 100 
    
    if remain > tl[c]+2.4 and p >= 100:
 
        t2.append(remain)
        return np.array([0, 0])
    
    rt = (1 / v) * Q(r, p, remain) * r
    pt = E(p) / r * rt
    
    return np.array([rt, pt])



t = np.arange(0, test_t, step) 
A = 0.6*0.6*math.pi
track1 = odeint(func, (0.85, 100), t, hmax=0.1 )

A = 0.65*0.65*math.pi
c = 1
track2 = odeint(func, (0.85, 100), t, hmax=0.1 )
A = 0.7*0.7*math.pi
c = 2
track3 = odeint(func, (0.85, 100), t, hmax=0.1 )
A = 0.75*0.75*math.pi
c = 3
track4 = odeint(func, (0.85, 100), t, hmax=0.1 )

out = []
for obj in track1:
    out.append(obj[1])

print(out[-1]-out[0])
sort = sorted(out)
print(sort[-1]-out[0],sort[1]-out[0])
print(sort[-1]-out[0]+sort[1]-out[0])


plt.hlines(100, 0, int(test_t), colors = "c", linestyles = "dashed")
#plt.hlines(150, 0, int(test_t/step), colors = "c", linestyles = "dashed")
t1 = [0.0]
l1, = plt.plot(t, out)

out = []
for obj in track2:
    out.append(obj[1])

l2, = plt.plot(t, out)

t1 = [0.233]
out = []
for obj in track3:
    out.append(obj[1])

l3, = plt.plot(t, out)

out = []
for obj in track4:
    out.append(obj[1])

l4, = plt.plot(t, out)

plt.ylabel("P-Pressure  (MPa)",fontsize=14)
plt.xlabel("T-Time  (ms)",fontsize=14)


plt.legend(handles=[l1 , l2, l3, l4], labels=['Width = 0.60mm','Width = 0.65mm','Width = 0.7mm','Width = 0.75mm'], fontsize = 10, loc='upper right')

plt.savefig(r"ρ-P image.png", dpi = 300)
