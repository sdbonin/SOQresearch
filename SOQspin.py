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
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time
from numpy import mod as mod

"""
************************
***Editable Variables***
************************
omega_0, alpha <- editable constants
dt <- max time step for integrator
totaltime <- total time integrator will run
t0 <- initial time
total plots <- the total number of plots, evenly distributed between t0 and totaltime
"""
omega_0 = 1
alpha = .001
dt = .1
totaltime = 511.2
t0 = 0
tolerance = .01
magtol = 1

"""
arguments packages the omega_0 and alpha into a numpy array for use in integrable function SOQsys.
"""
arguments = np.array([[omega_0, alpha]])

np.random.seed(42)

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
    q = np.array([q[0]])
    #print('q =')
    #print(q)
    norm = 1/np.sqrt(q[0,0]**2+q[0,1]**2+q[0,2]**2+q[0,3]**2)
    #print('norm =')
    #print(norm)
    q = norm*q
    #print('norm*q =')
    #print(q)
    normalizedq = quatreal(q)
    return normalizedq

"""
initialize random q_1 and q_2
"""

'''qvec_1 = 1*np.random.randn(1,4)
qvec_2 = 1*np.random.randn(1,4)
q_1 = quatreal(qvec_1)
q_2 = quatreal(qvec_2)
'''

"""
initialize variables q_1 and q_2
"""

q_1 = np.array([[1,0,0,0]])
q_2 = np.array([[1,0,0,0]])

q_1 = quatreal(q_1)
q_2 = quatreal(q_2)


"""
normalize q
"""

q_1 = normalize(q_1)
q_2 = normalize(q_2)


"""
initialize qdot_1 and qdot_2
"""

qdot_1 = np.zeros((4,4))
qdot_2 = np.zeros((4,4))

"""
initialize p_1 and p_2
p_# = [[0,real,real,real]]
"""

p_1 = np.array([[0,-0.4,-0.5,-np.sqrt(1-(.4**2)-(.5**2))]])
p_2 = np.array([[0,0.7,-0.7,-np.sqrt(1-(.7**2)-(.7**2))]])

p_1 = quatreal(p_1)
p_2 = quatreal(p_2)

p_1 = normalize(p_1)
p_2 = normalize(p_2)

"""
initialize pdots
"""

pdot_1 = np.zeros((4,4))
pdot_2 = np.zeros((4,4))

"""
repackage initial values into a numpy array for scipy.integrate
"""
initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])


def mag(q):
    """
    calculate magnitude of 4x4 real quaternion
    """
    magnitude = np.sqrt((q[0,0]**2)+(q[0,1]**2)+(q[0,2]**2)+(q[0,3]**2))
    return magnitude

def SOQsys(time,input,arguments):
    """
    This is the system of first order ODEs we're solving
    """
    """
    initialize real matrices from input
    """
    q_1 = quatreal(np.array([input[0:4]]))
    q_2 = quatreal(np.array([input[4:8]]))
    p_1 = quatreal(np.array([input[8:12]]))
    p_2 = quatreal(np.array([input[12:16]]))
    
    """
    pull out omega_0 and alpha from arguments
    """
    omega_0 = arguments[0,0]
    alpha = arguments[0,1]
    """
    matrix operations
    """
    #denominator = 1
    denominator = 1-(alpha**2)*((mag(q_1)**2)*(mag(q_2)**2))
    #print("denominator =")
    #print(denominator)
    if denominator == 0:
        output = "failure"
    else:
        dot1 = np.dot(q_1,conj(q_2))
        dot2 = np.dot(dot1,p_2)
        top1 = p_1 - alpha*dot2
        q_1_dt = top1*(1/denominator)
        #
        dot1 = np.dot(q_2,conj(q_1))
        dot2 = np.dot(dot1,p_1)
        top2 = p_2 - alpha*dot2
        q_2_dt = top2*(1/denominator)
        #
        dot1 = np.dot(q_1_dt,conj(q_2_dt))
        dot2 = np.dot(dot1,q_2)
        p_1_dt = -(omega_0**2)*q_1 + alpha*dot2
        #
        dot1 = np.dot(q_2_dt,conj(q_1_dt))
        dot2 = np.dot(dot1,q_1)
        p_2_dt = -(omega_0**2)*q_2 + alpha*dot2
        #
        output = np.append(q_1_dt[0],[q_2_dt[0],p_1_dt[0],p_2_dt[0]])
    return output
    
"""
Let me know if the way I'm interpreting how this integrator works is incorrect
"""

"""
define our integrator 
"vode" for a real system of ODE'set'
"bdf" instead of "adams" for stiff functions
no jacobian
max number of steps between each dt set to 100
"""


i = 0

"""
initialize spin matrices for plotting
"""

S_1 = np.dot(p_1,q_1)
S_2 = np.dot(p_2,q_2)
S_1init=S_1
S_2init=S_2

S_1r = S_1[0,0]
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
t_mat = np.array([[0]])

"""
run scipy.integrate ODE solver
"""
"""
this can probably be greatly sped up, and now that the plot function has been moved to SOQplot, 
this needs to be less hard coded
"""

runODE = ode(SOQsys).set_integrator('vode',method='bdf',with_jacobian=False,max_step=dt/100,first_step=dt/1000,atol=1e-10,rtol=1e-10)
runODE.set_initial_value(initialvalues,t0).set_f_params(arguments)

while runODE.successful() and runODE.t<totaltime:
    #print("runODE.t = ",runODE.t)
    check = runODE.integrate(runODE.t+dt)
    #print("runODE.integrate(runODE.t+dt) = ")
    #print("check = ")
    #print(check)
    results = np.array([check])
    
    q_1 = quatreal(np.array([results[0,0:4]]))
    q_2 = quatreal(np.array([results[0,4:8]]))
    p_2 = quatreal(np.array([results[0,12:16]]))
    p_1 = quatreal(np.array([results[0,8:12]]))
    S_1 = np.dot(p_1,q_1)
    S_2 = np.dot(p_2,q_2)
    #test = np.abs(S_1[0,1]-S_2[0,1])
    test1 = np.abs(S_1[0,0]-S_2init[0,0])
    test2 = np.abs(S_1[0,1]-S_2init[0,1])
    test3 = np.abs(S_1[0,2]-S_2init[0,2])
    test4 = np.abs(S_1[0,3]-S_2init[0,3])
    print('test1 = ',test1)
    print('test2 = ',test2)
    print('test3 = ',test3)
    print('test4 = ',test4)
    #print('test =',test)
    '''print('mag(q_1) =',mag(q_1))
    print('mag(p_1) =',mag(p_1))
    print('mag(S_1) =',mag(S_1))
    print('mag(q_2) =',mag(q_2))
    print('mag(p_2) =',mag(p_2))
    print('mag(S_2) =',mag(S_2))'''
    print("runODE.t = ",runODE.t)
    #if test < tolerance*2:
    if test2<magtol or test3<magtol or test4<magtol:
        S_1r = np.append(S_1r,S_1[0,0])
        S_1x = np.append(S_1x,S_1[0,1])
        S_1y = np.append(S_1y,S_1[0,2])
        S_1z = np.append(S_1z,S_1[0,3])
        S_2r = np.append(S_2r,S_2[0,0])
        S_2x = np.append(S_2x,S_2[0,1])
        S_2y = np.append(S_2y,S_2[0,2])
        S_2z = np.append(S_2z,S_2[0,3])
        #
        q_1r = np.append(q_1r,q_1[0,0])
        q_1x = np.append(q_1x,q_1[0,1])
        q_1y = np.append(q_1y,q_1[0,2])
        q_1z = np.append(q_1z,q_1[0,3])
        q_2r = np.append(q_2r,q_2[0,0])
        q_2x = np.append(q_2x,q_2[0,1])
        q_2y = np.append(q_2y,q_2[0,2])
        q_2z = np.append(q_2z,q_2[0,3])
        #
        p_1r = np.append(p_1r,p_1[0,0])
        p_1x = np.append(p_1x,p_1[0,1])
        p_1y = np.append(p_1y,p_1[0,2])
        p_1z = np.append(p_1z,p_1[0,3])
        p_2r = np.append(p_2r,p_2[0,0])
        p_2x = np.append(p_2x,p_2[0,1])
        p_2y = np.append(p_2y,p_2[0,2])
        p_2z = np.append(p_2z,p_2[0,3])
        t_mat = np.append(t_mat,runODE.t)
        #
        test1 = np.abs(S_1r[-1]-S_2init[0,0])
        test2 = np.abs(S_1x[-1]-S_2init[0,1])
        test3 = np.abs(S_1y[-1]-S_2init[0,2])
        test4 = np.abs(S_1z[-1]-S_2init[0,3])
        #testvalue = max(S_1r)
        '''print('max(S_1r) = ',testvalue)
        print('S_1r[-1] = ',S_1r[-1])
        i=i+1'''
       
        #time.sleep(.5)
        if test1<tolerance and test2<tolerance and test3<tolerance and test4<tolerance:
            totaltime=runODE.t

    
"""
create plotable matrices
"""
S_plot = np.zeros((S_1x.size,8))
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

print('done')
