# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:15:34 2020

@author: Matteo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def bezier(curve, tgel):
    # =============================================================================
    # # Experimental data
    # =============================================================================
    db = pd.read_excel('Bezier input2.xlsx', header=[0,1,2])
    t4000 = db[curve]['EXP4000']['t[min]']
    v4000 = db[curve]['EXP4000']['v']
    t100 = db[curve]['EXP100']['t[min]']
    v100 = db[curve]['EXP100']['v']
    mask =  ~ np.isnan(np.array(t100))
    t100 = t100[mask]
    v100 = v100[mask]

    mask4000 =  ~ np.isnan(np.array(t4000))
    t4000 = t4000[mask4000]
    v4000 = v4000[mask4000]

    # =============================================================================
    # # Reference points
    # =============================================================================
    class point():
        def __init__(self,x,y):
            self.x = x
            self.y = y

 
    A = point(0,0)
    C = point(tgel, 137)  #456

    x = np.linspace(1,tgel,100)

    # =============================================================================
    # # Minimize Bezier
    # =============================================================================
    def funy(w,A,C):
        B = point(C.x, 0)
        h = (A.x-B.x+np.sqrt(x*A.x-2*x*B.x+B.x**2+x*C.x-A.x*C.x))/(A.x-2*B.x+C.x)
        y = ((1-h)**2*A.y*w[0]+2*(1-h)*h*B.y*w[1]+h**2*C.y*w[2])/((1-h)**2*w[0]+2*(1-h)*h*w[1]+h**2*w[2])
        return y

    def offset(w,x,A,C,exp):
        y = funy(w,A,C)
        new_y = np.interp(exp.x, x, y)
        # mask=[exp.x>C.x*2/3]
        return new_y-exp.y

    experiment100 = point(t100.values,v100.values)
    experiment4000 = point(t4000.values,v4000.values)
    # y = funy([1,1,1],t,A,B,C)
    wopt = optimize.least_squares(offset, [1,1,1], args = (x,A,C,experiment100), bounds=([0,0,0],[np.inf,np.inf,np.inf]), verbose=1)

    # =============================================================================
    # # Minimize Exponential
    # =============================================================================

    def offset_exponential(a, exp_x, exp_y):
        y=a[0]*np.exp(a[1]*exp_x)+a[2]
        return y - exp_y

    wothers = optimize.least_squares(offset_exponential, [8.45733E-05,.06,0], args = (experiment100.x, experiment100.y), verbose=1)

    # =============================================================================
    # # Plot
    # =============================================================================

    yy = y=wothers.x[0]*np.exp(wothers.x[1]*x)+wothers.x[2]
    y_exponential100=yy-yy[0]

    yy4000 = y=wothers.x[0]*np.exp(wothers.x[1]*experiment4000.x)+wothers.x[2]
    y_exponential4000=yy4000-yy4000[0]


    fig,ax = plt.subplots()
    ax.scatter(t4000,v4000, label='experiment 4000', edgecolor='grey', c='none', s=10)
    ax.scatter(experiment100.x[1:-5],experiment100.y[1:-5], label='experiment 100',edgecolor='r', c='none', s=10)
    ax.plot(x,funy(wopt.x,A,C), label='Bezier', c='g')
    # ax.plot(x, y_exponential100, label='exponential', c='m')
    plt.legend()
    ax.set_xlabel('time [min]')
    ax.set_ylabel('viscosity [mPa.s')
    ax.set_title(label=curve)

    # Zoom in
    ax2 = plt.axes([.2, .2, .3, .4])
    ax2.plot(x,funy(wopt.x,A,C), label='Bezier', c='g')
    ax2.scatter(experiment100.x,experiment100.y, label='experiment 100', c='none', edgecolor='r', s=10)
    ax2.plot(x, y_exponential100, label='exponential', c='m')
    # ax2.set_xscale('log')
    ax2.set_ylim([0,40])
    ax2.set_xlim([10,tgel])
    plt.legend()
    return wopt.x, np.vstack((x,funy(wopt.x,A,C),y_exponential100)).T

curves = ['1.3M','1.4M','1.5M','1.6M','1.7M','1.8M']

# tgel from normalization
tgels=[297.0288177, 214.2193559, 128.0344619, 105.2325209, 74.02077031, 58.67911323]


result = []
for i in range(len(curves)):
    result.append(bezier(curves[i],tgels[i]))
