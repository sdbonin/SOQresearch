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
           /(1. - qmatmagsqr(q1)*qmatmagsqr(q2)*a**2)

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
a = 0.001 # coupling constant. \alpha in our notation
print 'alpha = ',a

seed = 8675309
np.random.seed(seed)
print 'seed =',seed

q1 = vec_mat(np.random.rand(4))
q2 = vec_mat(np.random.rand(4))
p1 = np.random.rand(4)
p2 = np.random.rand(4)

# this definition of p1 and p2 ensures the condition that the real part of 
# p_dagger*q is zero at t=0
p1[0] = -qvecmult(qveccon(p1),mat_vec(q1))[0]
p2[0] = -qvecmult(qveccon(p2),mat_vec(q2))[0]

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

dt = 0.001 #time step
t  = 0
print 'dt =',dt

q1a = []
q2a = []
p1a = []
p2a = []

s1a = []
s2a = []

time = []

L1 = [0,0]
L2 = [0,0]

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

while t<2000:
    q1a.append(mat_vec(q1))
    q2a.append(mat_vec(q2))
    p1a.append(mat_vec(p1))
    p2a.append(mat_vec(p2))
    s1a.append(mat_vec(np.dot(qmatcon(p1),q1)))
    s2a.append(mat_vec(np.dot(qmatcon(p2),q2)))
    time.append(t)
    con.append(mat_vec(conserved(q1,q2,p1,p2)))
    q1d = q1_dot(q1,q2,p1,p2,a)
    q2d = q1_dot(q2,q1,p2,p1,a)
    p1d = p1_dot(q1,q2,q1d,q2d,a,w)
    p2d = p1_dot(q2,q1,q2d,q1d,a,w)
    q1 += q1d*dt
    q2 += q2d*dt
    p1 += p1d*dt
    p2 += p2d*dt
    L1[0] = L1[1]
    L1[1] = np.sqrt(qmatmagsqr(p1)) - np.sqrt(qmatmagsqr(q1))
    L2[0] = L2[1]
    L2[1] = np.sqrt(qmatmagsqr(p2)) - np.sqrt(qmatmagsqr(q2))
    if L1[0]*L1[1]<0:
        q1 = qmatnorm(q1)
        p1 = qmatnorm(p1)
    if L2[0]*L2[1]<0:
        q2 = qmatnorm(q2)
        p2 = qmatnorm(p2)
    t  += dt
    
q1a = np.array(q1a)
q2a = np.array(q2a)
p1a = np.array(p1a)
p2a = np.array(p2a)
s1a = np.array(s1a)
s2a = np.array(s2a)
time = np.array(time)
con = np.array(con)

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
    plt.show()

vecplot(con,time,'$p_1^{\dagger}q_1 + p_2^{\dagger}q_2$')
vecplot(q1a,time,'$q_1$')
vecplot(q2a,time,'$q_2$')
vecplot(p1a,time,'$p_1$')
vecplot(p2a,time,'$p_2$')
vecplot(s1a,time,'$p_1^{\dagger}q_1$')
vecplot(s2a,time,'$p_2^{\dagger}q_2$')
