# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:14:03 2019

@author: Chenghai Li
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

file = open('3.txt','r')

device = torch.device('cuda:0')

x = []
y = []
   
for row in file.readlines():
    row = row.split()
    row[0] = float(row[0])
    row[1] = float(row[1])
    x.append(row[0])
    y.append(row[1])
    
E=interpolate.interp1d(x,y,kind="cubic")

A = 0.7*0.7*math.pi
v = 500*5*5*math.pi

t1 = torch.tensor([0.20],dtype=torch.double,requires_grad=True)
t2 = torch.tensor([2.7],dtype=torch.double,requires_grad=True)
test_t = 10
step = 0.001
#t = [0, t1, t1+2.4, t2]

class func(nn.Module):
    
    def forward(self, t, w):
        
        r, p= w
        
        if 0 <= t <= t1:
            Q = torch.tensor(0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.85 ),dtype=torch.double).clone().detach()
        if t1 < t <= t1+0.2:
            Q = torch.tensor([0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.85 ) - 100 * (t - t1)],dtype=torch.double).clone().detach()
        if t1+0.2 < t <= t1+2.2:
            Q = torch.tensor(0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.85 ) - 20,dtype=torch.double).clone().detach()
        if t1+2.2 < t <= t1+2.4:
            Q = torch.tensor(0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.85 ) - 100 * (t1 + 2.4 - t),dtype=torch.double).clone().detach()
        if t1+2.4 < t <= t2:
            Q = torch.tensor(0.85 * A * math.sqrt(2 * ( 160 - p ) / 0.85 ),dtype=torch.double).clone().detach()
        if t2 < t:
            Q = torch.tensor(0.,dtype=torch.double).clone().detach()
    
        
        rt = (1 / v) * Q * r
        pt = torch.tensor(E(p)) / r * rt
    
        return torch.tensor([rt, pt],dtype=torch.double)

itra = 20
loss_show = []
t2_list = []
optimizer = optim.RMSprop([t2], lr=0.01)
t = torch.tensor(np.arange(0, test_t, step),dtype=torch.double) 
y0 = torch.tensor([0.85, 100],dtype=torch.double)

for i in range (itra):
    
    optimizer.zero_grad()
    
    track1 = odeint(func(), y0, t, method='rk4')
    loss_2  = track1[-1][1] - y0[1]
 
    t_back = t2 * loss_2
    t_back.backward()
    optimizer.step()
   
    loss_show.append(math.log(loss_2**2))
    t2_list.append(t2)
    
    print(t2)
    print('itinerate:',i,' ','loss:',loss_2)
    
plt.plot(loss_show)


