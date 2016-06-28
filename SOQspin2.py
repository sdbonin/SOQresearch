"""
SOQspin2.py - sdbonin 
"""

import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numpy import mod as mod
import math

omega_0 = 1
alpha = .001
dt = .1
totaltime = 5000

diff = 5000
t0 = 0
tolerance = .01
magtol = 1
realtol = 1e-8
watchthis = 1e-17

arguments = np.array([[omega_0, alpha]])

#np.random.seed(1268)

def quatreal(q):
    """
    Turn a 4-vector quaternion into a real matrix
    https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
    """
    a = q[0,0]
    b = q[0,1]
    c = q[0,2]
    d = q[0,3]
    amat = a*np.identity(4)
    bmat = b*np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,1,0]])
    cmat = c*np.array([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
    dmat = d*np.array([[0,0,0,1],[0,0,-1,0],[0,1,0,0],[-1,0,0,0]])
    return amat+bmat+cmat+dmat
    
def conj(q):
    """
    Takes a 4x4 real quaternion matrix and makes the complex conjugate, 
    and uses quatreal to repackage it as a 4x4 real matrix
    """
    q = np.array([q[0]])
    q[0,1]=-q[0,1]
    q[0,2]=-q[0,2]
    q[0,3]=-q[0,3]
    complexconjugate = quatreal(q)
    return complexconjugate

def normalize(q):
    """
    Takes a 4x4 quaternion vector and normalizes it
    """
    quaternion = q[0]
    norm = 1/np.sqrt((q[0,0]**2)+(q[0,1]**2)+(q[0,2]**2)+(q[0,3]**2))
    #q = norm*q
    #print('norm =', norm)
    q[0,0] = norm*q[0,0]
    q[0,1] = norm*q[0,1]
    q[0,2] = norm*q[0,2]
    q[0,3] = norm*q[0,3]
    normalizedq = q
    return normalizedq
    
def mag(q):
    """
    calculate magnitude of 4x4 real quaternion
    """
    magnitude = np.sqrt((q[0,0]**2)+(q[0,1]**2)+(q[0,2]**2)+(q[0,3]**2))
    return magnitude
    
def randq():
    """
    From "Generating a random element of SO(3), Steven M. LaValle"
    """
    u = np.random.random((1,3))
    #print("u =",u)
    h = np.zeros((1,4))
    h[0,0] = np.sin(2*math.pi*u[0,1])*np.sqrt(1-u[0,0])
    h[0,1] = np.cos(2*math.pi*u[0,1])*np.sqrt(1-u[0,0])
    h[0,2] = np.sin(2*math.pi*u[0,2])*np.sqrt(u[0,0])
    h[0,3] = np.cos(2*math.pi*u[0,2])*np.sqrt(u[0,0])
    return h
    
def randImS():
    """
    From "Generating a random element of SO(3), Steven M. LaValle"
    """
    u = 2*np.random.random()-1
    theta = 2*math.pi*np.random.random()
    h = np.zeros((1,4))
    h[0,1] = np.cos(theta)*np.sqrt(1-u**2)
    h[0,2] = np.sin(theta)*np.sqrt(1-u**2)
    h[0,3] = u
    return h

def quatdot(q_1,q_2):
    """dot product of 2 quaternions"""
    dot = np.zeros((1,4))
    dot = q_1[0,0]*q_2[0,0] + q_1[0,1]*q_2[0,1] + q_1[0,2]*q_2[0,2] + q_1[0,3]*q_2[0,3]
    return dot
    
def EOM(q_1,q_2,p_1,p_2):
    """equations of motion"""
    alpha = 0.001
    qdot_1 = p_1
    qdot_2 = p_2
    pdot_1 = -q_1 + alpha * (q_2)
    pdot_2 = -q_2 + alpha * (q_1)
    results = np.append(qdot_1[0],[qdot_2[0],pdot_1[0],pdot_2[0]])
    return results

def SOQsys(input,t):
    """function for integrator"""
    q_1 = quatreal(np.array([input[0:4]]))
    q_2 = quatreal(np.array([input[4:8]]))
    p_1 = quatreal(np.array([input[8:12]]))
    p_2 = quatreal(np.array([input[12:16]]))
    #
    output = EOM(q_1,q_2,p_1,p_2)
    #print('time = ',time)
    return output
    
"""initial conditions"""

S_1 = normalize(quatreal(np.array([[0,1,0,0]])))
S_2 = normalize(quatreal(np.array([[0,1,1,0]])))

q_1 = quatreal(randq())
q_2 = quatreal(randq())

'''c = normalize(S_1 - S_2)
q_1 = quatreal(randq())
q_2 = np.dot(q_1,c)'''

qdot_1 = np.dot(q_1,conj(S_1))
qdot_2 = np.dot(q_2,conj(S_2))

p_1 = qdot_1
p_2 = qdot_2

S_1initial = S_1
S_2initial = S_2

Sreal_1_initial = S_1[0,0]
Sreal_2_initial = S_2[0,0]

L_1_initial = 0.5*(mag(qdot_1)**2 - mag(q_1)**2)
L_2_initial = 0.5*(mag(qdot_2)**2 - mag(q_2)**2)
#
cons_1_initial = np.sqrt((L_1_initial**2) + Sreal_1_initial**2)
cons_2_initial = np.sqrt((L_2_initial**2) + Sreal_2_initial**2)
#
L_int_initial = alpha * ((mag(q_1+q_2))**2)
#
L_tot_initial = L_1_initial + L_2_initial + L_int_initial


"""
repackage initial values into a numpy array for scipy.integrate
"""

initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])

print("Initial conditions valid...")
 
"""
run scipy.integrate ODE solver
"""

"""
the following is adapted from
http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html#scipy.integrate.odeint
"""

print("Running ODE solver...")

t = np.linspace(t0,totaltime,totaltime/dt)
sol = odeint(SOQsys,initialvalues,t,rtol=1e-10,atol=1e-10)
print('np.shape(sol) = ',np.shape(sol))

solsize = np.int(totaltime/dt)
print('solsize = ',solsize)
i = 1

"""initialize plotable matrices"""

print("Initializing plotable matrices...")

S_1 = np.zeros((totaltime/dt,4))
S_2 = np.zeros((totaltime/dt,4))
S_error = np.zeros((totaltime/dt,4))
S_erVec = np.zeros((totaltime/dt,4))
cons_1 = np.zeros((totaltime/dt,1))
cons_2 = np.zeros((totaltime/dt,1))
S_diff = np.zeros((totaltime/dt,4))
S_diff_mag = np.zeros((totaltime/dt,1))
cons_1 = np.zeros((totaltime/dt,1))
cons_2 = np.zeros((totaltime/dt,1))
L_tot = np.zeros((totaltime/dt,1))
L_1 = np.zeros((totaltime/dt,1))
L_2 = np.zeros((totaltime/dt,1))
L_int = np.zeros((totaltime/dt,1))
S_1mag = np.zeros((totaltime/dt,1))
S_2mag = np.zeros((totaltime/dt,1))

S_1[0] = S_1initial[0]
S_2[0] = S_2initial[0]
S_error[0] = S_1initial[0]-S_2initial[0]
S_erVec[0] = mag(quatreal(np.array([S_error[0]])))
cons_1[0] = 0
cons_2[0] = 0
S_diff[0] = S_1initial[0]-S_2initial[0]
S_diff_mag[0] = mag(quatreal(np.array([S_diff[0]])))
L_1[i] = L_1_initial
L_2[i] = L_2_initial
L_int[i] = L_int_initial
cons_1[0] = cons_1_initial
cons_2[0] = cons_2_initial
L_tot[0] = L_tot_initial
S_1mag[0] = mag(S_1initial)
S_2mag[0] = mag(S_2initial)

print("Creating plottable matrices...")

while i < solsize:
    q_1 = np.array([sol[i,0:4]])
    p_1 = np.array([sol[i,8:12]])
    q_1 = quatreal(q_1)
    p_1 = quatreal(p_1)
    #
    q_2 = np.array([sol[i,4:8]])
    p_2 = np.array([sol[i,12:16]])
    q_2 = quatreal(q_2)
    p_2 = quatreal(p_2)
    #
    PsAndQs = np.array([EOM(q_1,q_2,p_1,p_2)])
    #
    qdot_1 = quatreal(np.array([PsAndQs[0,0:4]]))
    qdot_2 = quatreal(np.array([PsAndQs[0,4:8]]))
    #
    S_1val = np.dot(conj(qdot_1),q_1)
    S_1[i] = S_1val[0]
    #
    S_2val = np.dot(conj(qdot_2),q_2)
    S_2[i] = S_2val[0]
    #
    Sreal_1 = S_1val[0,0]
    Sreal_2 = S_2val[0,0]
    #
    L_1[i] = 0.5*(mag(qdot_1)**2 - mag(q_1)**2)
    L_2[i] = 0.5*(mag(qdot_2)**2 - mag(q_2)**2)
    #
    L_int[i] = 0
    #
    L_tot[i] = L_1[i] + L_2[i] + L_int[i]
    #
    cons_1[i] = np.sqrt((L_1[i]**2) + Sreal_1**2)
    cons_2[i] = np.sqrt((L_2[i]**2) + Sreal_2**2)
    #
    S_error[i] = S_1[i] - S_2[0]
    S_erVec[i] = mag(quatreal(np.array([S_error[i]])))
    #
    i = i + 1
    #print(i)

print("Plotting...")

print('S_1[-1,:] =',S_1[-1,:])
print('S_2[-1,:] =',S_2[-1,:])

print('S_1[0,:] =',S_1[0,:])
print('S_2[0,:] =',S_2[0,:])

plt.subplot(221)
plt.plot(t, L_tot[:, 0], label='L_tot')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
axes.set_ylim([-1.5*np.max(np.abs(L_tot[:, :])),1.5*np.max(np.abs(L_tot[:, :]))])
plt.grid()


'''plt.subplot(222)
plt.plot(t, L_int[:, 0], label='L_int')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
axes.set_ylim([-1.5*np.max(np.abs(L_int[:, :])),1.5*np.max(np.abs(L_int[:, :]))])
plt.grid()'''


plt.subplot(223)
plt.plot(t, L_1[:, 0], label='L_1')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
axes.set_ylim([-1.5*np.max(np.abs(L_1[:, :])),1.5*np.max(np.abs(L_1[:, :]))])
plt.grid()



plt.subplot(224)
plt.plot(t, L_2[:, 0], label='L_2')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
axes.set_ylim([-1.5*np.max(np.abs(L_2[:, :])),1.5*np.max(np.abs(L_2[:, :]))])
plt.grid()
plt.show()




plt.subplot(221)
plt.plot(t, S_error[:, 0], label='Serr_r')
plt.plot(t, S_error[:, 1], label='Serr_x')
plt.plot(t, S_error[:, 2], label='Serr_y')
plt.plot(t, S_error[:, 3], label='Serr_z')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
plt.grid()



plt.subplot(222)
plt.plot(t, S_erVec[:, 0], label='mag')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_xlim([0,diff])
plt.grid()



plt.subplot(223)
plt.plot(t, cons_1[:], label='cons_1')
plt.plot(t, S_1[:, 0], label='S_1r')
plt.plot(t, S_1[:, 1], label='S_1x')
plt.plot(t, S_1[:, 2], label='S_1y')
plt.plot(t, S_1[:, 3], label='S_1z')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_ylim([-2,2])
axes.set_xlim([totaltime-diff,totaltime])
plt.grid()



plt.subplot(224)
plt.plot(t, cons_2[:], label='cons_2')
plt.plot(t, S_2[:, 0], label='S_2r')
plt.plot(t, S_2[:, 1], label='S_2x')
plt.plot(t, S_2[:, 2], label='S_2y')
plt.plot(t, S_2[:, 3], label='S_2z')
plt.legend(loc='best')
plt.xlabel('t')
axes = plt.gca()
axes.set_ylim([-2,2])
axes.set_xlim([0,diff])
plt.grid()
plt.show()
    

    

"""
create plotable matrices
"""
'''S_plot = np.zeros((S_1x.size,8))
S_plot[:,0] = S_1r
S_plot[:,1] = S_1x
S_plot[:,2] = S_1y
S_plot[:,3] = S_1z
S_plot[:,4] = S_2r
S_plot[:,5] = S_2x
S_plot[:,6] = S_2y
S_plot[:,7] = S_2z

q_plot = np.zeros((q_1x.size,8))
q_plot[:,0] = q_1r
q_plot[:,1] = q_1x
q_plot[:,2] = q_1y
q_plot[:,3] = q_1z
q_plot[:,4] = q_2r
q_plot[:,5] = q_2x
q_plot[:,6] = q_2y
q_plot[:,7] = q_2z

p_plot = np.zeros((p_1x.size,8))
p_plot[:,0] = p_1r
p_plot[:,1] = p_1x
p_plot[:,2] = p_1y
p_plot[:,3] = p_1z
p_plot[:,4] = p_2r
p_plot[:,5] = p_2x
p_plot[:,6] = p_2y
p_plot[:,7] = p_2z

"""
saves files to github directoy, but they're ignored by .gitignore
"""

np.savetxt('S_plot.txt',S_plot,delimiter=',')
np.savetxt('q_plot.txt',q_plot,delimiter=',')
np.savetxt('p_plot.txt',p_plot,delimiter=',')
np.savetxt('time.txt',t_mat,delimiter=',')

print('done')'''