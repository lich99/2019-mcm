# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:22:08 2019

@author: Chenghai Li
"""

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
from scipy import interpolate

device = torch.device('cuda:0')

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
    
pr_pre = interpolate.interp1d(p, r, kind="cubic")

def pr(p):
    
    if p>0.1001:
        return pr_pre(p)
    else:
        return [0.8043390]

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
        return torch.tensor(l2_1(x), dtype = torch.double)
    if 0.45 <= x <= 2:
        return torch.tensor([2], dtype = torch.double)
    if 2 < x < 2.45:
        return torch.tensor(l2_2(x), dtype = torch.double)
    if 2.45 <= x <= 100:
        return torch.tensor([0], dtype = torch.double)
    
A = 0.7*0.7*math.pi
v = 500*5*5*math.pi

omiga = math.pi/111

def v0(t):
    
    return  torch.tensor([-(-2.413 * math.sin( omiga * t + math.pi/2 ) + 4.826) * math.pi * 2.5 * 2.5 + 162.1374326208532], dtype = torch.double)

def v0t(t):
    
    return  torch.tensor([omiga * 2.413 * math.cos ( omiga * t + math.pi/2 ) * math.pi * 2.5 * 2.5], dtype = torch.double)


class func(nn.Module):
    
    def forward(self, t, w):

        p0, p1= w
    
        tr = t % 100
            
        if p0 >= p1 and v0t(t) < 0:
        
            Q0 = 0.85 * A * math.sqrt(2 * ( p0 - p1 ) / torch.tensor(pr(p0), dtype = torch.double) ) 
 
        else:   
        
            Q0 = torch.tensor([0], dtype = torch.double)

        if p0 < 0.5 and v0t(t) > 0: 
        
            r0t = torch.tensor([0], dtype = torch.double)
            p0t = torch.tensor([0], dtype = torch.double)
        
        else:
          
            r0t = (-Q0 * torch.tensor(pr(p0), dtype = torch.double) * v0(t) - v0(t) * torch.tensor(pr(p0), dtype = torch.double) * v0t(t)) / (v0(t) ** 2)
            p0t = torch.tensor(E(p0), dtype = torch.double) / torch.tensor(pr(p0), dtype = torch.double) * r0t
    
 
        A1 = math.pi * l(tr) * math.sin( math.pi / 20 ) * (4 * 1.25 + l(tr) * math.sin( math.pi / 20 ) * math.cos( math.pi / 20 ) )
        A2 = math.pi * 0.7 * 0.7
    
        Q1 =   0.85 * min(A1, A2) * math.sqrt(2 * p1 / torch.tensor(pr(p1), dtype = torch.double))
    
        r1t = (Q0 * torch.tensor(pr(p0), dtype = torch.double) - Q1 * torch.tensor(pr(p1), dtype = torch.double)) / v
        p1t = torch.tensor(E(p1), dtype = torch.double) / torch.tensor(pr(p1), dtype = torch.double) * r1t

        return torch.tensor([p0t, p1t],dtype=torch.double)

time_range = 20
step_length = 0.01


time_start=time.time()

t = torch.tensor( np.arange(0, time_range, step_length), dtype=torch.double) 
y0 = torch.tensor([0.5, 100], dtype=torch.double)
track1 = odeint(func(), y0, t, method='rk4')

time_end=time.time()

print('totally cost',time_end-time_start)

'''
show = []

for i in range(len(track1)):
    show.append(track1[i][0])
    
plt.plot(show)
'''

