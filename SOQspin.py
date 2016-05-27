"""
SOQspin.py - sdbonin (comments are work in progress, cleaning up the code for efficiency and checking physics)
****************************
***How to run on Windows:***
****************************
1) Download Python 3.5, I recommend here:
    https://www.continuum.io/downloads#_windows

2) Chances are you found this on Github anyway, but in case you haven't the repository is here:
    https://github.com/sdbonin/SOQresearch
    
    Feel free to email me at sdbonin **at** gmail **DOT** com and I'll add you as a collaborator.
    
    There are many options, but for our purposes I recommend using the Github app which can be downloaded here:
    https://desktop.github.com/
    
3) Open up a command prompt and cd to the SOQresearch Github directory. For example:
    "cd C:\sdbonin\GitHub\SOQresearch" without quotes (or wherever you have Github sync your files.)
    
4) now run the command:
    "python SOQspin.py"

Several variables are editable, should be located at the top of the code. I intend to make this more interactive from the command prompt.

If you edit code or comments, please make sure to make a quick note of what you do when you sync it to Github.

****Work List 5/10/2016****

continue to update commentary

remove unnecessary editables

clean up "dot" variables

"""

import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from numpy import mod as mod
import math

"""
************************
***Editable Variables***
************************
omega_0, alpha <- editable constants
dt <- max time step for integrator
totaltime <- total time integrator will run
t0 <- initial time
"""
omega_0 = 1
alpha = .001
dt = .1
totaltime = 5000

"""For a normal plot set totaltime and diff equal to eachother"""

diff = 5000
t0 = 0
tolerance = .01
magtol = 1
realtol = 1e-8
watchthis = 1e-17

"""
arguments packages the omega_0 and alpha into a numpy array for use in integrable function SOQsys.
"""
arguments = np.array([[omega_0, alpha]])

#np.random.seed(4949)

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
    print('norm =', norm)
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
    print("u =",u)
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
    print("u =",u)
    theta = 2*math.pi*np.random.random()
    print("theta =",theta)
    h = np.zeros((1,4))
    #h[0,0] = np.sin(2*math.pi*u[0,1])*np.sqrt(1-u[0,0])
    h[0,1] = np.cos(theta)*np.sqrt(1-u**2)
    h[0,2] = np.sin(theta)*np.sqrt(1-u**2)
    h[0,3] = u
    return h

def quatdot(q_1,q_2):
    """dot product of 2 quaternions"""
    dot = np.zeros((1,4))
    dot = q_1[0,0]*q_2[0,0] + q_1[0,1]*q_2[0,1] + q_1[0,2]*q_2[0,2] + q_1[0,3]*q_2[0,3]
    return dot

"""
initialize random q_1 and q_2
"""

q_1 = quatreal(randq())
q_2 = quatreal(randq())

"""
initialize random S_1 and S_1
"""
S_1 = quatreal(randq())
print("S_1 = ",S_1)
S_2 = quatreal(randq())
print("S_2 = ",S_2)


"""find a,b,c"""

'''rotAxis = np.zeros((1,4))
Sdots = np.dot(S_1,S_2)
rotAxis[0,1:4] = Sdots[0,1:4]

v = normalize(quatreal(rotAxis))

theta = 2*np.arccos(quatdot(S_1,S_2))
print("theta = ", theta)

c = np.zeros((1,4))
c[0,0] = np.cos(theta/2)
c[0,1] = -v[0,1]*np.sin(theta/2)
c[0,2] = -v[0,2]*np.sin(theta/2)
c[0,3] = -v[0,3]*np.sin(theta/2)
c = quatreal(c)
print("c = ",c)

dot = np.dot(c,S_2)
check = np.dot(dot,conj(c))
print("check = ",check)
print("-S_1 = ",-S_1)

a = randImS()
b = np.dot(a,c)

q_1 = a
q_2 = b'''

"""
initialize p_1 and p_2
"""

qdot_1 = np.dot(q_1,conj(S_1))
qdot_2 = np.dot(q_2,conj(S_2))

p_1 = qdot_1 + 2*alpha*(mag(q_1+q_2)**2)*(qdot_1-qdot_2)

p_2 = qdot_2 + 2*alpha*(mag(q_1+q_2)**2)*(qdot_2-qdot_1)

S_1initial = S_1
S_2initial = S_2
"""
repackage initial values into a numpy array for scipy.integrate
"""

initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])


def SOQsys(input,t,omega_0,alpha):
    """
    This is the system of first order ODEs we're solving
    """
    """
    initialize real matrices from input
    """
    #print('time = ', t)
    #
    q_1 = quatreal(np.array([input[0:4]]))
    q_2 = quatreal(np.array([input[4:8]]))
    p_1 = quatreal(np.array([input[8:12]]))
    p_2 = quatreal(np.array([input[12:16]]))
    #
    omega_0 = arguments[0,0]
    alpha = arguments[0,1]
    """
    matrix operations
    """
    #denominator = 1
    denominator = 1+(4*alpha)*(mag(q_1+q_2)**2)
    #print("denominator =")
    #print(denominator)
    if denominator == 0:
        output = "failure"
    else:
        #
        numerator = p_1 + 2*alpha*(mag(q_1+q_2)**2)*(p_1+p_2)
        q_1dot = numerator*(1/denominator)
        #
        numerator = p_2 + 2*alpha*(mag(q_1+q_2)**2)*(p_2+p_1)
        q_2dot = numerator*(1/denominator)
        #
        p_1dot = -q_1+2*alpha*(mag(q_1dot-q_2dot)**2)*(q_1+q_2)
        #
        p_2dot = -q_2+2*alpha*(mag(q_2dot-q_1dot)**2)*(q_1+q_2)
        #
        output = np.append(q_1dot[0],[q_2dot[0],p_1dot[0],p_2dot[0]])
        #print('time = ',time)
    return output
    
"""
Let me know if the way I'm interpreting how this integrator works is incorrect
"""
i = 0

"""
initialize spin matrices for plotting
"""

#S_1 = np.dot(conj(qdot_1),q_1)
#S_2 = np.dot(conj(qdot_2),q_2)
S_1init=S_1
S_2init=S_2

'''S_1r = S_1[0,0]
S_1x = S_1[0,1]
S_1y = S_1[0,2]
S_1z = S_1[0,3]
S_2r = S_2[0,0]
S_2x = S_2[0,1]
S_2y = S_2[0,2]
S_2z = S_2[0,3]
    #
q_1r = q_1[0,0]
q_1x = q_1[0,1]
q_1y = q_1[0,2]
q_1z = q_1[0,3]
q_2r = q_2[0,0]
q_2x = q_2[0,1]
q_2y = q_2[0,2]
q_2z = q_2[0,3]
    #
p_1r = p_1[0,0]
p_1x = p_1[0,1]
p_1y = p_1[0,2]
p_1z = p_1[0,3]
p_2r = p_2[0,0]
p_2x = p_2[0,1]
p_2y = p_2[0,2]
p_2z = p_2[0,3]
t_mat = np.array([[0]])'''

"""
run scipy.integrate ODE solver
"""
"""
this can probably be greatly sped up, and now that the plot function has been moved to SOQplot, 
this needs to be less hard coded
"""

"""
the following is adapted from
http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html#scipy.integrate.odeint
"""

t = np.linspace(t0,totaltime,totaltime/dt)
sol = odeint(SOQsys,initialvalues,t,args=(omega_0,alpha),rtol=1e-10,atol=1e-10)
print('np.shape(sol) = ',np.shape(sol))

'''plt.subplot(221)
plt.plot(t, sol[:, 0], label='q_1r')
plt.plot(t, sol[:, 1], label='q_1x')
plt.plot(t, sol[:, 2], label='q_1y')
plt.plot(t, sol[:, 3], label='q_1z')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()

plt.subplot(222)
plt.plot(t, sol[:, 4], label='p_1r')
plt.plot(t, sol[:, 5], label='p_1x')
plt.plot(t, sol[:, 6], label='p_1y')
plt.plot(t, sol[:, 7], label='p_1z')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()'''

solsize = np.int(totaltime/dt)
print('solsize = ',solsize)
i = 1

print('q_1 = ',q_1)
S_1init = S_1
S_2init = S_2

S_1 = np.zeros((totaltime/dt,4))
S_2 = np.zeros((totaltime/dt,4))
S_error = np.zeros((totaltime/dt,4))
S_erVec = np.zeros((totaltime/dt,4))
cons_1 = np.zeros((totaltime/dt,1))
cons_2 = np.zeros((totaltime/dt,1))

S_1[0] = S_1initial[0]
S_2[0] = S_2initial[0]
S_error[0] = S_1initial[0]-S_2initial[0]
print("S_error[0] =",S_error[0])
S_erVec[0] = mag(quatreal(np.array([S_error[0]])))
cons_1[0] = 0
cons_2[0] = 0

'''dot1 = np.dot(q_1,conj(q_2))
dot2 = np.dot(dot1,qdot_2)
p_1 = qdot_1 - alpha*dot2

dot1 = np.dot(q_2,conj(q_1))
dot2 = np.dot(dot1,qdot_1)
p_2 = qdot_2 - alpha*dot2'''

while i < solsize:
    q_1i = np.array([sol[i,0:4]])
    p_1i = np.array([sol[i,8:12]])
    q_1i = quatreal(q_1i)
    p_1i = quatreal(p_1i)
    #
    q_2i = np.array([sol[i,4:8]])
    p_2i = np.array([sol[i,12:16]])
    q_2i = quatreal(q_2i)
    p_2i = quatreal(p_2i)
    #
    denominator = 1+(4*alpha)*(mag(q_1i+q_2i)**2)
    #
    numerator = p_1i + 2*alpha*(mag(q_1i+q_2i)**2)*(p_1i+p_2i)
    q_1idot = numerator*(1/denominator)
    #
    numerator = p_2i + 2*alpha*(mag(q_1i+q_2i)**2)*(p_2i+p_1i)
    q_2idot = numerator*(1/denominator)
    #
    S_1val = np.dot(conj(q_1idot),q_1i)
    #print('S_1val =',S_1val)
    #
    S_1[i] = S_1val[0]
    #
    S_2val = np.dot(conj(q_2idot),q_2i)
    #
    S_2[i] = S_2val[0]
    #
    Sreal_1 = S_1val[0,0]
    Sreal_2 = S_2val[0,0]
    #print('Sreal_1 =',Sreal_1)
    #
    L_1 = 0.5*(mag(q_1idot)**2 - mag(q_1i)**2)
    L_2 = 0.5*(mag(q_2idot)**2 - mag(q_2i)**2)
    #print('L_1 =',L_1)
    #
    cons_1[i] = np.sqrt((L_1**2) + Sreal_1**2)
    cons_2[i] = np.sqrt((L_2**2) + Sreal_2**2)
    #
    cons_2[i]
    #
    S_error[i] = S_1[i] - S_2[i]
    S_erVec[i] = mag(quatreal(np.array([S_error[i]])))
    #
    i = i + 1
    #print(i)

print('S_1[-1,:] =',S_1[-1,:])
print('S_2[-1,:] =',S_2[-1,:])

print('S_1[0,:] =',S_1[0,:])
print('S_2[0,:] =',S_2[0,:])

print('S_1.dtype = ',S_1.dtype)
print('S_2.dtype = ',S_2.dtype)

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