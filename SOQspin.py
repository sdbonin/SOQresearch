"""
Work in progess - sdbonin
"""

import numpy as np

omega_0 = 1
alpha = 0.001
dt = 0.01

def quatreal(q):
    a = q[0,0]
    b = q[0,1]
    c = q[0,2]
    d = q[0,3]
    amat = a*np.identity(4)
    bmat = b*np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,1,0]])
    cmat = c*np.array([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
    dmat = d*np.array([[0,0,0,1],[0,0,-1,0],[0,1,0,0],[-1,0,0,0]])
    return amat+bmat+cmat+dmat

qvec_1 = np.random.randn(1,4)
qvec_2 = np.random.randn(1,4)

q_1 = quatreal(qvec_1)
q_2 = quatreal(qvec_2)

qdot_1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
qdot_2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

p_1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
p_2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

pdot_1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
pdot_2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

# next add scipy integration