# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:53:29 2025

@author: Ludwig
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, y, sigma, x0, y0):
    A = np.exp(-(x-x0)**2/(2*sigma**2))* np.exp(-(y-y0)**2/(2*sigma**2))
    return(A/np.sum(A))

x = np.linspace(-1.5, 1.5, 501) # position in mm
dx =  x[1] - x[0]
y = np.linspace(-1.5, 1.5, 501) # position in mm
dy = y[1] -y[0]

X,Y = np.meshgrid(x,y)

r = np.sqrt((X)**2 + (Y)**2)
phi = np.arctan2(Y, X)

seg = 2*np.pi/20
rad_bull_eye = 0.1
rad_single_bull = 0.2
rad_triple = [0.5,0.6]
rad_double = [0.9,1]
numbers = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

scoreboard = np.zeros_like(r)
for i, num in enumerate(numbers):
    i = i-10
    ind = np.where((phi> i*seg-seg/2)*(phi<= i*seg+seg/2))
    # print(i)
    scoreboard[ind]=num
ind = np.where((phi> 10*seg-seg/2))    
scoreboard[ind]=numbers[0]
factor = np.ones_like(r)
factor[np.where((r<=rad_double[1]) * (r>rad_double[0]))] = 2
factor[np.where((r<=rad_triple[1]) * (r>rad_triple[0]))] = 3
scoreboard = scoreboard * factor
scoreboard[np.where(r<= rad_single_bull)] = 25
scoreboard[np.where(r<= rad_bull_eye)] = 50
scoreboard[np.where(r> rad_double[1])] = 0


fig,ax = plt.subplots()

extent_xy = [x[0]-0.5*dx, x[-1]+0.5*dx, y[0]-0.5*dy, y[-1]+0.5*dy]

imxy = ax.imshow(scoreboard,
                    cmap='hot',
                    origin='lower',
                    aspect=1,
                    extent=extent_xy)

sigma, x0, y0 = 0.01, 0, 0

throw = gaussian(X, Y, sigma, x0, y0)

fig2,ax2 = plt.subplots()

imxy = ax2.imshow(throw,
                    cmap='hot',
                    origin='lower',
                    aspect=1,
                    extent=extent_xy)

result = throw*scoreboard

fig3,ax3 = plt.subplots()

imxy = ax3.imshow(result,
                    cmap='hot',
                    origin='lower',
                    aspect=1,
                    extent=extent_xy)

print(np.sum(result))
