"""
SOQplot.py - sdbonin (work in progress)
read _plot array from txt and plot them
"""

import numpy as np
import matplotlib.pyplot as plt

S_plot = np.loadtxt('S_plot.txt',delimiter=',')
q_plot = np.loadtxt('q_plot.txt',delimiter=',')
p_plot = np.loadtxt('p_plot.txt',delimiter=',')
time = np.loadtxt('time.txt',delimiter=',')

S_1r = S_plot[:,0] #= S_1r
S_1x = S_plot[:,1] #= S_1x
S_1y = S_plot[:,2] #= S_1y
S_1z = S_plot[:,3] #= S_1z
S_2r = S_plot[:,4] #= S_2r
S_2x = S_plot[:,5] #= S_2x
S_2y = S_plot[:,6] #= S_2y
S_2z = S_plot[:,7] #= S_2z

q_1x = q_plot[:,1] #= q_1x
q_1y = q_plot[:,2] #= q_1y
q_1z = q_plot[:,3] #= q_1z
q_2x = q_plot[:,5] #= q_2x
q_2y = q_plot[:,6] #= q_2y
q_2z = q_plot[:,7] #= q_2z
q_1r = q_plot[:,0] #= q_1r
q_2r = q_plot[:,4] #= q_2r

p_1x = p_plot[:,1] #= p_1x
p_1y = p_plot[:,2] #= p_1y
p_1z = p_plot[:,3] #= p_1z
p_2x = p_plot[:,5] #= p_2x
p_2y = p_plot[:,6] #= p_2y
p_2z = p_plot[:,7] #= p_2z
p_1r = p_plot[:,0] #= p_1r
p_2r = p_plot[:,4] #= p_2r

plt.figure()

plt.subplot(221)
#plt.semilogy(time,np.abs(S_1r),label='S_1r',color='purple')
plt.plot(time,np.abs(S_1r),label='S_1r',color='purple')

plt.plot(time,S_1x,label='S_1i',color='red')
plt.plot(time,S_1y,label='S_1j',color='blue')
plt.plot(time,S_1z,label='S_1k',color='green')
plt.xlabel('time')
plt.ylabel('S_1')
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1,1])

plt.subplot(222)
#plt.semilogy(time,np.abs(S_2r),label='S_2r',color='purple')
plt.plot(time,S_2r,label='S_2r',color='purple')

plt.plot(time,S_2x,label='S_2i',color='red')
plt.plot(time,S_2y,label='S_2j',color='blue')
plt.plot(time,S_2z,label='S_2k',color='green')
plt.xlabel('time')
plt.ylabel('S_1')
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1,1])

plt.subplot(223)
#plt.semilogy(time,np.abs(S_2r),label='S_2r',color='purple')
plt.plot(time,S_1r,label='S_1r',color='purple')

plt.plot(time,S_1x,label='S_1i',color='red')
plt.plot(time,S_1y,label='S_1j',color='blue')
plt.plot(time,S_1z,label='S_1k',color='green')
plt.xlabel('time')
plt.ylabel('S_1')
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1,1])
axes.set_xlim([500,512.35])

plt.subplot(224)
#plt.semilogy(time,np.abs(S_2r),label='S_2r',color='purple')
plt.plot(time,S_2r,label='S_2r',color='purple')

plt.plot(time,S_2x,label='S_2i',color='red')
plt.plot(time,S_2y,label='S_2j',color='blue')
plt.plot(time,S_2z,label='S_2k',color='green')
plt.xlabel('time')
plt.ylabel('S_1')
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1,1])
axes.set_xlim([0,12.35])

plt.show()
