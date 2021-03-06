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