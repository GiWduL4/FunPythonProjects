# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:53:29 2025

@author: Ludwig
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def shifting(array, shift_y, shift_x):
    full_array = np.zeros(np.shape(array))
    y1, y2 = max(0, shift_y), min(array.shape[0], array.shape[0]+shift_y)
    x1, x2 = max(0, shift_x), min(array.shape[1], array.shape[1]+shift_x)
    full_array[y1:y2, x1:x2] = array[y1-shift_y:y2-shift_y, x1-shift_x:x2-shift_x] 
    return(full_array)

def gaussian(x, y, sigma, x0, y0):
    A = np.exp(-(x-x0)**2/(2*sigma**2))* np.exp(-(y-y0)**2/(2*sigma**2))
    return(A/np.sum(A))

x = np.linspace(-250, 250, 201) # position in mm
dx =  x[1] - x[0]
y = np.linspace(-250, 250, 201) # position in mm
dy = y[1] -y[0]

Y,X = np.meshgrid(x,y)

r = np.sqrt((X)**2 + (Y)**2)
phi = np.arctan2(Y, X)

seg = 2*np.pi/20
rad_bull_eye = 6.35
rad_single_bull = 15.9
rad_triple = [99,107]
rad_double = [162,170]
numbers = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]
bull2 = 50
bull1 = 25

scoreboard = np.zeros_like(r)
for i, num in enumerate(numbers):
    # i = i-10
    ind = np.where((phi+np.pi> i*seg-seg/2)*(phi+np.pi<= i*seg+seg/2))
    # print(i)
    scoreboard[ind]=numbers[i-10]
ind = np.where((phi> 10*seg-seg/2))    
scoreboard[ind]=numbers[0-10]
# scoreboard = np.swapaxes(scoreboard, 0, 1)
# scoreboard = np.flip(scoreboard)
factor = np.ones_like(r)
factor[np.where((r<=rad_double[1]) * (r>rad_double[0]))] = 2
factor[np.where((r<=rad_triple[1]) * (r>rad_triple[0]))] = 3
scoreboard = scoreboard * factor
scoreboard[np.where(r<= rad_single_bull)] = bull1
scoreboard[np.where(r<= rad_bull_eye)] = bull2
scoreboard[np.where(r> rad_double[1])] = 0


def score(params):
    x0, y0 = params
    shift_x_px = int(x0/dx)
    shift_y_px = int(y0/dy)
    throw = shifting(throw0, shift_y_px, shift_x_px)#gaussian(X, Y, sigma, x0, y0)
    result = throw*scoreboard
    return(np.sum(result))

sigma_list = np.linspace(1e-9,120,25)

p = 0.91
sigma = rad_single_bull / (np.sqrt(-2*np.log(1-p)))
# sigma_list = [sigma]

for sigma in sigma_list:
    throw0 = gaussian(X, Y, sigma, 0, 0)
    opt = 0
    x_opt = None
    y_opt = None
    step = 0.1
    start_time = time.time()
    for k, x0 in enumerate(tqdm(x, desc="Processing")):
        # print(str(k) + '/' + str(len(x)))
        for y0 in y:
            value = score([x0,y0])
            if value >= opt:
                opt = value
                x_opt = x0
                y_opt = y0
        # progress = (k+1)/len(x)
        # if progress >= step:
        #      print('Progress: ' + str(round(progress*100,3)) + ' %')
        #      step += 0.1
    print('Calculation Time: ' + str(round(time.time()-start_time,3)) + ' s')         
    print('Streuung: ' + str(round(sigma,2)) + ' mm')      
    print('Max Score: ' + str(round(opt,3)) + ' at x = ' + str(round(x_opt,3)) + ' and y = ' + str(round(y_opt,2)))
    
    fig,ax = plt.subplots()
    
    extent_xy = [x[0]-0.5*dx, x[-1]+0.5*dx, y[0]-0.5*dy, y[-1]+0.5*dy]
    
    imxy = ax.imshow(scoreboard,
                        cmap='gray',
                        origin='lower',
                        aspect=1,
                        extent=extent_xy)
    
    ax.scatter(x_opt, y_opt, color='red', marker='x', s=100)
    
    theta = np.linspace(0, 2*np.pi, 50)
    circle_x = x_opt + sigma * np.cos(theta)
    circle_y = y_opt + sigma * np.sin(theta)
    
    plt.plot(circle_x, circle_y, color='red', linewidth=2)
    plt.title('Streuung: ' + str(round(sigma,2)) + ' mm')
    
    plt.show()
