# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:40:37 2019

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

xnew=np.linspace(0,200,1001)
ynew=E(xnew)
l1, = plt.plot(x3,y3)
l2, = plt.plot(xnew,ynew)

out = []
for obj in xnew:
    out.append(0.02893*obj**2+3.077*obj+1572)
    
l3, = plt.plot(xnew,out)
plt.legend(handles=[l1 , l2, l3], labels=['Data point','Interpolation','Fitting'], fontsize = 10, loc='lower right')
plt.xlabel("P-Pressure  (MPa)",fontsize=14)
plt.ylabel("E-Elastic Modulus  (MPa)",fontsize=14)
plt.savefig(r"ρ-P image.png", dpi = 300)