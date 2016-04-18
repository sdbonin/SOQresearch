"""
SOQfind.py - sdbonin (work in progress)
find a specific value in a loaded array
"""

import numpy as np

S_plot = np.loadtxt('S_plot.txt',delimiter=',')
q_plot = np.loadtxt('q_plot.txt',delimiter=',')
p_plot = np.loadtxt('p_plot.txt',delimiter=',')
time = np.loadtxt('time.txt',delimiter=',')

S_1x = S_plot[:,1] #= S_1x
S_1y = S_plot[:,2] #= S_1y
S_1z = S_plot[:,3] #= S_1z
S_2x = S_plot[:,5] #= S_2x
S_2y = S_plot[:,6] #= S_2y
S_2z = S_plot[:,7] #= S_2z
S_1r = S_plot[:,0] #= S_1r
S_2r = S_plot[:,4] #= S_2r

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

print('S_plot[0,0:5] = ')
print(S_plot[0,0:4])

print('S_plot[0,4:8] = ')
print(S_plot[0,4:8])

print('S_plot[1,0:5] = ')
print(S_plot[1,0:4])

print('S_plot[1,4:8] = ')
print(S_plot[1,4:8])

'''print('S_plot[-112:,0:5] = ')
print(S_plot[-112:,0:4])

print('S_plot[-112:,4:8] = ')
print(S_plot[-112:,4:8])'''

print('S_plot[49900:50000,0:5] = ')
print(S_plot[49900:50000,0:4])

print('S_plot[49900:50000,4:8] = ')
print(S_plot[49900:50000,4:8])

print('time[0] = ')
print(time[0])

print('time[-1] = ')
print(time[-1])

print('root swap time')

rswt = time[-1]/2

print('rswt = ')
print(rswt)

'''index = np.floor(rswt/.1)

print('index = ')
print(index)

print('S_plot[index,0:5] = ')
print(S_plot[index,0:4])

print('S_plot[index,4:8] = ')
print(S_plot[index,4:8])'''