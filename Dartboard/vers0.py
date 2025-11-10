# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:53:29 2025

@author: Ludwig
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
from matplotlib.patches import Circle

def gaussian(x, y, sigma, x0, y0):
    A = np.exp(-(x-x0)**2/(2*sigma**2))* np.exp(-(y-y0)**2/(2*sigma**2))
    return(A/np.sum(A))

x = np.linspace(-250, 250, 201) # position in mm
dx =  x[1] - x[0]
y = np.linspace(-250, 250, 201) # position in mm
dy = y[1] -y[0]

X,Y = np.meshgrid(x,y)

r = np.sqrt((X)**2 + (Y)**2)
phi = np.arctan2(Y, X)

seg = 2*np.pi/20
rad_bull_eye = 6.35
rad_single_bull = 15.9
rad_triple = [99,107]
rad_double = [162,170]
numbers = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

scoreboard = np.zeros_like(r)
for i, num in enumerate(numbers):
    # i = i-10
    ind = np.where((phi+np.pi> i*seg-seg/2)*(phi+np.pi<= i*seg+seg/2))
    # print(i)
    scoreboard[ind]=num
ind = np.where((phi> 10*seg-seg/2))    
scoreboard[ind]=numbers[0]
scoreboard = np.swapaxes(scoreboard, 0, 1)
scoreboard = np.flip(scoreboard)
factor = np.ones_like(r)
factor[np.where((r<=rad_double[1]) * (r>rad_double[0]))] = 2
factor[np.where((r<=rad_triple[1]) * (r>rad_triple[0]))] = 3
scoreboard = scoreboard * factor
scoreboard[np.where(r<= rad_single_bull)] = 25
scoreboard[np.where(r<= rad_bull_eye)] = 80
scoreboard[np.where(r> rad_double[1])] = 0


def score(params):
    x0, y0 = params
    throw = gaussian(X, Y, sigma, x0, y0)
    result = throw*scoreboard
    return(np.sum(result))

sigma = 7


opt = 0
x_opt = None
y_opt = None
for k, x0 in enumerate(x):
    print(str(k) + '/' + str(len(x)))
    for y0 in y:
        value = score([x0,y0])
        if value >= opt:
            opt = value
            x_opt = x0
            y_opt = y0
            
print('Max Score ' + str(round(opt,3)) + ' at x = ' + str(round(x_opt,3)) + ' and y = ' + str(round(y_opt,2)))

fig,ax = plt.subplots()

extent_xy = [x[0]-0.5*dx, x[-1]+0.5*dx, y[0]-0.5*dy, y[-1]+0.5*dy]

imxy = ax.imshow(scoreboard,
                    cmap='gray',
                    origin='lower',
                    aspect=1,
                    extent=extent_xy)

ax.scatter(x_opt, y_opt, color='red', marker='x', s=100)

theta = np.linspace(0, 2*np.pi, 100)
circle_x = x_opt + sigma * np.cos(theta)
circle_y = y_opt + sigma * np.sin(theta)

plt.plot(circle_x, circle_y, color='blue', linewidth=2)
circle = Circle((x0, y0), sigma, edgecolor='red', facecolor='none', linewidth=2)
plt.title('Streuung: ' + str(sigma) + ' mm')

plt.show()
