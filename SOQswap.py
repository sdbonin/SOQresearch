# -*- coding: utf-8 -*-
"""
These functions should be much more efficient than those of the previous
version, however this code doesn't actually have any of the plotting nor does
it have a solution to plot. These should be a nice starting point for working
with quaternions as they simplify common operations including conversion
between matrix and vector representations.

That said, these can almost certainly be optimized a bit if we need to make the
integration run faster once we get to that point.

To multiply two matrix quaternions together, use np.dot(M1,M2)

This code was written in Python 2.7, however most if not all of it should work
in Python 3.5

I'm not actually sure how Python modules work, but it would probably be a good
idea to format this all as a module so that these functions could be imported
easily
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
    returns the magnitude of a 4-vector quaternion
    '''
    return qvecmult(qveccon(vec),vec)[0]

def qmatmagsqr(M):
    '''
    piggy-backs off the previous function to give the magnitude of 2x2 imaginary matrix
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

def p1_dot(q1,q2,q2dot,a,w):
    '''
    takes the current values of things we know and the hopefully recently
    calculated derivatives of q1,q2 and uses them to find other derivatives
    '''
    return a*np.dot(qmatcon(q2dot),q2) - q1*w**2

#------------------------------------------------------------------------------
#          Defining necessary constants and initial conditions
#                   AKA "on the seventh day..."
#------------------------------------------------------------------------------

w = 1. # \omega_0 in our notation
a = 0.001 # coupling constant. \alpha in our notation

q1 = vec_mat(np.random.rand(4))
q2 = vec_mat(np.random.rand(4))
p1 = vec_mat(np.random.rand(4))
p2 = vec_mat(np.random.rand(4))

#------------------------------------------------------------------------------
#                     Defining loop parameters
#            AKA "Configuring the space-time continuum"
#------------------------------------------------------------------------------

dt = 0.001 #time step. setting to 0.0001 results in 2000 iterations
t  = 0

q1a = []
q2a = []
p1a = []
p2a = []
time = []

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

while t<1.5:
    q1a.append(mat_vec(q1))
    q2a.append(mat_vec(q2))
    p1a.append(mat_vec(p1))
    p2a.append(mat_vec(p2))
    time.append(t)
    con.append(mat_vec(conserved(q1,q2,p1,p2)))
    q1d = q1_dot(q1,q2,p1,p2,a)
    q2d = q1_dot(q2,q1,p2,p1,a)
    p1d = p1_dot(q1,q2,q2d,a,w)
    p2d = p1_dot(q2,q1,q1d,a,w)
    q1 += q1d*dt
    q2 += q2d*dt
    p1 += p1d*dt
    p2 += p2d*dt
    t  += dt
    
q1a = np.array(q1a)
q2a = np.array(q2a)
p1a = np.array(p1a)
p2a = np.array(p2a)
time = np.array(time)
con = np.array(con)

print con # looks pretty conserved to me, even the real part
