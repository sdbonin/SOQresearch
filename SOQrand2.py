"""
SOQrand2.py - sdbonin 
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
totaltime = (math.pi/4)*1000

diff = (math.pi/4)*1000

'''totaltime = (.4)*1000

diff = (.4)*1000'''
t0 = 0
tolerance = .01
magtol = 1
realtol = 1e-8
watchthis = 1e-17

arguments = np.array([[omega_0, alpha]])

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
    '''qdot_1 = p_1
    qdot_2 = p_2
    pdot_1 = -q_1 + alpha * (q_2)
    pdot_2 = -q_2 + alpha * (q_1)'''
    #
    denominator = 1 + 2*alpha
    #
    numerator = p_1 + alpha * (p_1 - p_2)
    qdot_1 = numerator*(1/denominator)
    #
    numerator = p_2 + alpha * (p_2 - p_1)
    qdot_2 = numerator*(1/denominator)
    #
    pdot_1 = -q_1 + alpha * (q_1 + q_2)
    #
    pdot_2 = -q_2 + alpha * (q_2 + q_1)
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

n = 0

"""find out which c's go to k-> 0.353"""

"""try other lagrangians"""

"""c changing with time, c(t) = q_1(t)*q_2(t) along with S_1 and S_2"""

while n < 1: 
    """initial conditions"""

    S_1 = normalize(quatreal(np.array([[0,1,0,0]])))
    S_2 = normalize(quatreal(np.array([[0,-1,1,0]])))

    '''print("S_1 = ", S_1)
    print("S_2 = ", S_2)'''

    '''q_1 = quatreal(randq())
    q_2 = quatreal(randq())'''

    '''c = np.dot(conj(q_1),q_2)'''

    c = normalize(S_1-S_2)
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)

    '''c = np.dot(normalize(S_1+S_2),quatreal(randImS()))
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    '''c = np.dot(normalize(S_1+S_2),quatreal(randImS()))
    q_1 = quatreal(randq())
    q_2 = conj(q_1)'''
    
    '''c = normalize(quatreal(np.array([[0,1,0,0]])))
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    
    '''q_1 = quatreal(randq())
    q_2 = q_1'''
    
    '''q_1 = S_1
    q_2 = S_2'''
    
    '''c = np.dot(S_1,conj(S_2))
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    '''angle = 2*math.pi*np.random.random()  
    Sdot = S_1-S_2
    Sdot = np.array([Sdot[0]])
    Sdot[0,0] = 0
    print("Sdot = ",Sdot)
    Sdot = quatreal(Sdot)
    c = quatreal(np.array([[np.cos(angle),0,0,0]])) + np.sin(angle)*normalize(Sdot)
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    '''c = quatreal(np.array([[1,0,0,0]]))
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    '''c = quatreal(np.array([[1,0,0,0]]))
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    '''c = np.dot(conj(S_2),S_1)
    q_1 = quatreal(randq())
    q_2 = np.dot(q_1,c)'''
    
    #print("c = ",c)

    qdot_1 = np.dot(q_1,conj(S_1))
    qdot_2 = np.dot(q_2,conj(S_2))

    p_1 = qdot_1 + alpha*(qdot_1 + qdot_2)
    p_2 = qdot_2 + alpha*(qdot_2 + qdot_1)
    
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

    '''print("Creating plottable matrices...")

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
        L_tot[i] = L_1[i] + L_2[i]
        #
        cons_1[i] = np.sqrt((L_1[i]**2) + Sreal_1**2)
        cons_2[i] = np.sqrt((L_2[i]**2) + Sreal_2**2)
        #
        S_error[i] = S_1[i] - S_2[0]
        S_erVec[i] = mag(quatreal(np.array([S_error[i]])))
        #
        i = i + 1
        #print(i)'''
        
    q_1 = np.array([sol[-1,0:4]])
    p_1 = np.array([sol[-1,8:12]])
    q_1 = quatreal(q_1)
    p_1 = quatreal(p_1)
    #
    q_2 = np.array([sol[-1,4:8]])
    p_2 = np.array([sol[-1,12:16]])
    q_2 = quatreal(q_2)
    p_2 = quatreal(p_2)
    #
    qdot_1 = p_1
    #
    qdot_2 = p_2
    #
    #
    #
    PsAndQs = np.array([EOM(q_1,q_2,p_1,p_2)])
    #
    qdot_1 = quatreal(np.array([PsAndQs[0,0:4]]))
    qdot_2 = quatreal(np.array([PsAndQs[0,4:8]]))
    #
    pdot_1 = quatreal(np.array([PsAndQs[0,8:12]]))
    pdot_1 = quatreal(np.array([PsAndQs[0,12:16]]))
    '''pdot_1 = -q_1 + alpha * (q_1 + q_2)
    #
    pdot_2 = -q_2 + alpha * (q_2 + q_1)'''
    #        
    S_1mat = np.dot(conj(qdot_1),q_1)
    #print('S_1mat =',S_1mat)
    #
    S_1 = S_1mat[0]
    #
    S_2mat = np.dot(conj(qdot_2),q_2)
    #
    S_2 = S_2mat[0]
    #
    Sreal_1 = S_1mat[0,0]
    Sreal_2 = S_2mat[0,0]
    #print('Sreal_1 =',Sreal_1)
    #
        
    i = i + 1
        
    qmat = np.zeros((1,8))
    qmat[0,0:4] = q_1[0,:]
    qmat[0,4:8] = q_2[0,:]
    #print("qmat = ",qmat)
    print("q_1[0] = ",q_1[0])
    print("q_2[0] = ",q_2[0])

    Smat = np.zeros((1,8))
    Smat[0,0:4] = S_1mat[0,:]
    Smat[0,4:8] = S_2mat[0,:]
    #print("qmat = ",qmat)
    print("S_1mat[0] = ",S_1mat[0])
    print("S_2mat[0] = ",S_2mat[0])

    if n == 0:
        qswaps = qmat
    else:
        qswaps = np.append(qswaps,qmat,axis = 0)
        
    if n == 0:
        Sswaps = Smat
    else:
        Sswaps = np.append(Sswaps,Smat,axis = 0)

    print("Sswaps = ",Sswaps)
    n = n + 1


plt.subplot(221)        
plt.scatter(Sswaps[:,0],Sswaps[:,4],label = "S_r")
plt.legend(loc='best')
print("Swaps[:,0] = ",Sswaps[:,0])
print("Swaps[:,4] = ",Sswaps[:,4])
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
axes.set_xlim([-1.5,1.5])


plt.subplot(222)
plt.scatter(Sswaps[:,1],Sswaps[:,5],label = "S_i")
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
axes.set_xlim([-1.5,1.5])


plt.subplot(223)
plt.scatter(Sswaps[:,2],Sswaps[:,6],label = "S_j")
plt.legend(loc='best')
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
axes.set_xlim([-1.5,1.5])


plt.subplot(224)
plt.scatter(Sswaps[:,3],Sswaps[:,7],label = "S_k")
plt.legend(loc='best')
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plt.show()
