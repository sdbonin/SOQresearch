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
totaltime = 1.5*(math.pi/2)*200

diff = 1.5*(math.pi/2)*200
t0 = 0

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
    '''q[0,0] = norm*q[0,0]
    q[0,1] = norm*q[0,1]
    q[0,2] = norm*q[0,2]
    q[0,3] = norm*q[0,3]'''
    normalizedq = norm*q
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
    #
    b = 0.01*normalize(quatreal(np.array([[0,0,1,0]])))
    #
    qdot_1 = p_1 - np.dot(q_1,b)
    qdot_2 = p_2 - np.dot(q_2,b)
    #
    pdot_1 = -q_1 - np.dot(p_1,b)
    pdot_2 = -q_2 - np.dot(p_2,b)
    #
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

S_1 = normalize(quatreal(np.array([[0,0,0,1]])))
S_2 = normalize(quatreal(np.array([[0,-1,1,0]])))

q_1 = quatreal(randq())
q_2 = quatreal(randq())

#print("c = ",c)

p_1 = np.dot(q_1,conj(S_1))
p_2 = np.dot(q_2,conj(S_2))


S_1initial = S_1
S_2initial = S_2

Sreal_1_initial = S_1[0,0]
Sreal_2_initial = S_2[0,0]

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
    S_1val = np.dot(conj(p_1),q_1)
    S_1[i] = S_1val[0]
    #
    S_2val = np.dot(conj(p_2),q_2)
    S_2[i] = S_2val[0]
    #
    Sreal_1 = S_1val[0,0]
    Sreal_2 = S_2val[0,0]
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

plt.subplot(121)
#plt.plot(t, cons_1[:], label='cons_1')
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


plt.subplot(122)
#plt.plot(t, cons_2[:], label='cons_2')
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