"""
SOQplot.py - sdbonin (work in progress)
read _plot array from txt and plot them
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import time
from numpy import mod as mod
'''import matplotlib.animation as animation'''

S_plot = np.loadtxt('S_plot.txt',delimiter=',')
q_plot = np.loadtxt('q_plot.txt',delimiter=',')
p_plot = np.loadtxt('p_plot.txt',delimiter=',')
time = np.loadtxt('time.txt',delimiter=',')

#S_plot = np.zeros((S_1x.size,8))
S_1x = S_plot[:,0] #= S_1x
S_1y = S_plot[:,1] #= S_1y
S_1z = S_plot[:,2] #= S_1z
S_2x = S_plot[:,3] #= S_2x
S_2y = S_plot[:,4] #= S_2y
S_2z = S_plot[:,5] #= S_2z
S_1r = S_plot[:,6] #= S_1r
S_2r = S_plot[:,7] #= S_2r

#q_plot = np.zeros((q_1x.size,8))
q_1x = q_plot[:,0] #= q_1x
q_1y = q_plot[:,1] #= q_1y
q_1z = q_plot[:,2] #= q_1z
q_2x = q_plot[:,3] #= q_2x
q_2y = q_plot[:,4] #= q_2y
q_2z = q_plot[:,5] #= q_2z
q_1r = q_plot[:,6] #= q_1r
q_2r = q_plot[:,7] #= q_2r

#p_plot = np.zeros((p_1x.size,8))
p_1x = p_plot[:,0] #= p_1x
p_1y = p_plot[:,1] #= p_1y
p_1z = p_plot[:,2] #= p_1z
p_2x = p_plot[:,3] #= p_2x
p_2y = p_plot[:,4] #= p_2y
p_2z = p_plot[:,5] #= p_2z
p_1r = p_plot[:,6] #= p_1r
p_2r = p_plot[:,7] #= p_2r

plt.figure()

plt.subplot(121)
plt.plot(time,S_1x,label='S_1i',color='red')
plt.plot(time,S_1y,label='S_1j',color='blue')
plt.plot(time,S_1z,label='S_1k',color='green')
plt.xlabel('S_1')
plt.ylabel('time')
plt.legend(loc='best')

plt.subplot(122)
plt.plot(time,S_2x,label='S_2i',color='red')
plt.plot(time,S_2y,label='S_2j',color='blue')
plt.plot(time,S_2z,label='S_2k',color='green')
plt.xlabel('S_2')
plt.ylabel('time')
plt.legend(loc='best')

plt.show()
