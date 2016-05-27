# -*- coding: utf-8 -*-
"""
This code uses a loop along with our set of coupled differential equations and
matrix math to create arrays of 4-vector quaternions.

The old plotting functions need to be updated and incorperated into the end of
this code or a better visualization solution needs to be found.
"""


#------------------------------------------------------------------------------
#               Importing modules and copying functions
#                   AKA "Setting stuff up"
#------------------------------------------------------------------------------


import numpy as np
from time import time as checktime


# a set of init quaternions and the identity matrix for building general q-matrices
rm = np.identity(2)
im = np.array([[-1j,0],[0,1j]])
jm = np.array([[0,1],[-1,0]])
km = np.array([[0,-1j],[-1j,0]])

def vec_mat(v):
    '''
    Converts a quaternion vector into the 2x2 imaginary matrix representation
    '''
    return v[0]*rm + v[1]*im + v[2]*jm + v[3]*km

def mat_vec(M):
    '''
    Converts a 2x2 imaginary matrix quaternion into its vector representation
    '''
    return np.array([ M[1,1].real , M[1,1].imag , M[0,1].real , -M[0,1].imag ])

def qvecmult(vec1,vec2):
    '''
    Multiplies two 4-vector quaternions via matrix math
    '''
    return mat_vec(np.dot(vec_mat(vec1),vec_mat(vec2)))

def qmatcon(M):
    '''
    conjugates a 2x2 imaginary matrix quaternion
    '''
    return vec_mat(mat_vec(M)*np.array([1,-1,-1,-1]))

def qveccon(vec):
    '''
    conjugates 4-vector quaternion
    '''
    return vec*np.array([1,-1,-1,-1])

def qvecnorm(vec):
    '''
    normalizes a 4-vector quaternion
    '''
    return vec/np.sqrt(qvecmult(qveccon(vec),vec)[0])

def qmatnorm(M):
    '''
    piggy-backs off the previous function to normalize 2x2 imaginary matrices
    '''
    return vec_mat(qvecnorm(mat_vec(M)))

def qvecmagsqr(vec):
    '''
    returns the magnitude squared of a 4-vector quaternion
    '''
    return qvecmult(qveccon(vec),vec)[0]

def qmatmagsqr(M):
    '''
    piggy-backs off the previous function to give the magnitude squared of 2x2 imaginary matrix
    quaternions
    '''
    return qvecmagsqr(mat_vec(M))



#------------------------------------------------------------------------------
#                   Defining the differential equations
#              AKA "Bringing (first) order to the universe"
#------------------------------------------------------------------------------

def q1_dot(q1,q2,p1,p2,a):
    '''
    takes the current value of things that we know and calculates derivatives
    Function assumes 2x2 complex matrices as inputs for q1,q2,p1,p2
    a is the coupling constant
    '''
    return (p1 - a*np.dot(q1,np.dot(qmatcon(q2),p2)))   \
           #/(1. - qmatmagsqr(q1)*qmatmagsqr(q2)*a**2)

def p1_dot(q1,q2,q1dot,q2dot,a,w):
    '''
    takes the current values of things we know and the hopefully recently
    calculated derivatives of q1,q2 and uses them to find other derivatives
    '''
    return a*np.dot(q1dot,np.dot(qmatcon(q2dot),q2)) - q1*w**2

#------------------------------------------------------------------------------
#          Defining necessary constants and initial conditions
#                   AKA "on the first day..."
#------------------------------------------------------------------------------

w = 1. # \omega_0 in our notation
a = 0.01 # coupling constant. \alpha in our notation
print 'alpha =',a

seed = 42
np.random.seed(seed)
print 'seed =',seed

q1 = vec_mat([1,0,0,0])
q2 = vec_mat([1,0,0,0])
p1 = np.random.rand(4)
p2 = np.random.rand(4)
p1[0] = 0
p2[0] = 0


p1 = vec_mat(p1)
p2 = vec_mat(p2)


q1 = qmatnorm(q1)
q2 = qmatnorm(q2)
p1 = qmatnorm(p1)
p2 = qmatnorm(p2)

#------------------------------------------------------------------------------
#                     Defining loop parameters
#            AKA "Configuring the space-time continuum"
#------------------------------------------------------------------------------

dt = 0.01 #time step
t  = 0
print 'dt = ',dt



q1a = [mat_vec(q1)]
p1a = [mat_vec(p1)]
s1a = [mat_vec(np.dot(qmatcon(p1),q1))]
q2a = [mat_vec(q2)]
p2a = [mat_vec(p2)]
s2a = [mat_vec(np.dot(qmatcon(p2),q2))]
time = [t]


swaptime = 0.8785/a #determined 'experimentally'



#------------------------------------------------------------------------------
#                   Checking conserved quantity
#                     AKA "might as well..."
#------------------------------------------------------------------------------

con = [] #checking to see if our conserved quantity is actually conserved

def conserved(q1,q2,p1,p2):
    return np.dot(qmatcon(p1),q1) + np.dot(qmatcon(p2),q2)

#------------------------------------------------------------------------------
#                   Creating the time loop
#                     AKA "Let 'er rip"
#------------------------------------------------------------------------------


runtime = checktime()
while t<swaptime:
    '''
    This integrator works on an RK4 algorithm.
    For a good explaination, see wikipedia
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    note that the algorithm is modified slightly to fit our function
    '''
    q1k1 = q1_dot(q1,q2,p1,p2,a)
    q2k1 = q1_dot(q2,q1,p2,p1,a)
    p1k1 = p1_dot(q1,q2,q1k1,q2k1,a,w)
    p2k1 = p1_dot(q2,q1,q2k1,q1k1,a,w)
    q1k2 = q1_dot(q1+q1k1*dt/2.,q2+q2k1*dt/2.,p1+p1k1*dt/2.,p2+p2k1*dt/2.,a)
    q2k2 = q1_dot(q2+q2k1*dt/2.,q1+q1k1*dt/2.,p2+p2k1*dt/2.,p1+p1k1*dt/2.,a)
    p1k2 = p1_dot(q1+q1k1*dt/2.,q2+q2k1*dt/2.,q1k1,q2k1,a,w)
    p2k2 = p1_dot(q2+q2k1*dt/2.,q1+q1k1*dt/2.,q2k1,q1k1,a,w)
    q1k3 = q1_dot(q1+q1k2*dt/2.,q2+q2k2*dt/2.,p1+p1k2*dt/2.,p2+p2k2*dt/2.,a)
    q2k3 = q1_dot(q2+q2k2*dt/2.,q1+q1k2*dt/2.,p2+p2k2*dt/2.,p1+p1k2*dt/2.,a)
    p1k3 = p1_dot(q1+q1k2*dt/2.,q2+q2k2*dt/2.,q1k1,q2k1,a,w)
    p2k3 = p1_dot(q2+q2k2*dt/2.,q1+q1k2*dt/2.,q2k1,q1k1,a,w)
    q1k4 = q1_dot(q1+q1k3*dt,q2+q2k3*dt,p1+p1k3*dt,p2+p2k3*dt,a)
    q2k4 = q1_dot(q2+q2k3*dt,q1+q1k3*dt,p2+p2k3*dt,p1+p1k3*dt,a)
    p1k4 = p1_dot(q1+q1k3*dt,q2+q2k3*dt,q1k1,q2k1,a,w)
    p2k4 = p1_dot(q2+q2k3*dt,q1+q1k3*dt,q2k1,q1k1,a,w)
    q1 += (q1k1 + 2*q1k2 + 2*q1k3 + q1k4)*dt/6.
    q2 += (q2k1 + 2*q2k2 + 2*q2k3 + q2k4)*dt/6.
    p1 += (p1k1 + 2*p1k2 + 2*p1k3 + p1k4)*dt/6.
    p2 += (p2k1 + 2*p2k2 + 2*p2k3 + p2k4)*dt/6.
    t  += dt
    q1a.append(mat_vec(q1))
    p1a.append(mat_vec(p1))
    s1a.append(mat_vec(np.dot(qmatcon(p1),q1)))
    q2a.append(mat_vec(q2))
    p2a.append(mat_vec(p2))
    s2a.append(mat_vec(np.dot(qmatcon(p2),q2)))
    time.append(t)

runtime = checktime() - runtime

q1a = np.array(q1a)
q2a = np.array(q2a)
p1a = np.array(p1a)
p2a = np.array(p2a)
s1a = np.array(s1a)
s2a = np.array(s2a)
time = np.array(time)

#------------------------------------------------------------------------------
#                       Plotting things
#                   AKA "Can we see it now?"
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

def vecplot(thing,time,name):
    plt.clf()
    plt.title(name)
    plt.plot(time,thing[:,0],label='Real', color = 'black')
    plt.plot(time,thing[:,1],label='i', color = 'red')
    plt.plot(time,thing[:,2],label='j', color = 'green')
    plt.plot(time,thing[:,3],label='k', color = 'blue')
    plt.legend(loc='best')
    plt.xlim([time[0], time[-1]])
    plt.grid()
    plt.show()

def scalarplot(thing,time,name):
    plt.clf()
    plt.title(name)
    plt.plot(time,thing,color = 'black')
    plt.grid()    
    plt.xlim([time[0], time[-1]])
    plt.show()


vecplot(q1a,time,'$q_1$')
vecplot(q2a,time,'$q_2$')
vecplot(p1a,time,'$p_1$')
vecplot(p2a,time,'$p_2$')
vecplot(s1a,time,'$p_1^{\dagger}q_1$')
vecplot(s2a,time,'$p_2^{\dagger}q_2$')


print 'Initial:'
print 'q1 = ', q1a[0]
print 'q2 = ', q2a[0]
print 'p1 = ', p1a[0]
print 'p2 = ', p2a[0]
print 's1 = ', s1a[0]
print 's2 = ', s2a[0]
print 'Final:'
print 'q1 = ', q1a[-1]
print 'q2 = ', q2a[-1]
print 'p1 = ', p1a[-1]
print 'p2 = ', p2a[-1]
print 's1 = ', s1a[-1]
print 's2 = ', s2a[-1]
print 'runtime is',runtime, 'seconds'