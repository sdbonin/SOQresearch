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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import time
from numpy import mod as mod
'''import matplotlib.animation as animation'''

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
alpha = 0.001
dt = .01
totaltime = 200
t0 = 0
totalplots = 100

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
    print('q =')
    print(q)
    norm = 1/np.sqrt(q[0,0]**2+q[0,1]**2+q[0,2]**2+q[0,3]**2)
    print('norm =')
    print(norm)
    q = norm*q
    print('norm*q =')
    print(q)
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

'8-11,12-15'
qdot_1 = np.zeros((4,4))
qdot_2 = np.zeros((4,4))

"""
initialize p_1 and p_2
p_# = [[0,real,real,real]]
"""

'16-19,20-23'
'''p_1 = np.random.randn(1,4)
p_2 = np.random.randn(1,4)
p_1[0,0] = 0
p_2[0,0] = 0'''

p_1 = np.array([[0,-0.4,-0.5,-np.sqrt(1-(.4**2)-(.5**2))]])
p_2 = np.array([[0,0.7,-0.7,-np.sqrt(1-(.7**2)-(.7**2))]])

"""
ignore the following for now
"""
'''pmat_1 = quatreal(p_1)
pmat_2 = quatreal(p_2)
condition_1 = np.dot(conj(pmat_1),q_1)
condition_2 = np.dot(conj(pmat_2),q_2)
"""condition should be purely imaginary"""
p_1[0] = condition_1[0,0]
p_2[0] = condition_2[0,0]'''

p_1 = quatreal(p_1)
p_2 = quatreal(p_2)

p_1 = normalize(p_1)
p_2 = normalize(p_2)

"""
initialize pdots
"""

'24-27,28-31'
pdot_1 = np.zeros((4,4))
pdot_2 = np.zeros((4,4))

"""
repackage initial values into a numpy array for scipy.integrate
"""
#initialvalues = np.append(q_1[0],[q_2[0],qdot_1[0],qdot_2[0],p_1[0],p_2[0],pdot_1[0],pdot_2[0]])
initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])


def mag(q):
    """
    calculate magnitude of 4x4 real quaternion
    """
    magnitude = np.sqrt(q[0,0]**2+q[0,1]**2+q[0,2]**2+q[0,3]**2)
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
    divisor = 1-(alpha**2)*((mag(q_1)**2)*(mag(q_2)**2))
    #print("divisor =")
    #print(divisor)
    if divisor == 0:
        output = "failure"
    else:
        dot1 = np.dot(q_1,conj(q_2))
        dot2 = np.dot(dot1,p_2)
        top1 = p_1 - alpha*dot2
        q_1_dt = top1*(1/divisor)
        #q_1_dt = quatreal(q_1_dt)
        #
        dot1 = np.dot(q_2,conj(q_1))
        dot2 = np.dot(dot1,p_1)
        top2 = p_2 - alpha*dot2
        q_2_dt = top2*(1/divisor)
        #q_2_dt = quatreal(q_2_dt)
        #
        dot1 = np.dot(q_1_dt,conj(q_2_dt))
        dot2 = np.dot(dot1,q_2)
        p_1_dt = -(omega_0**2)*q_1 + alpha*dot2
        #p_1_dt = quatreal(p_1_dt)
        #
        dot1 = np.dot(q_2_dt,conj(q_1_dt))
        dot2 = np.dot(dot1,q_1)
        p_2_dt = -(omega_0**2)*q_2 + alpha*dot2
        #p_2_dt = quatreal(p_2_dt)
        #
        output = np.append(q_1_dt[0],[q_2_dt[0],p_1_dt[0],p_2_dt[0]])
    #print("    time = ",time)
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
runODE = ode(SOQsys).set_integrator('vode',method='adams',with_jacobian=False,max_step=dt/100)
runODE.set_initial_value(initialvalues,t0).set_f_params(arguments)

i = 0
"""
initialize spin matrices for plotting
"""
#S_1x = np.zeros((1,1))
#S_2x = np.zeros((1,1))
#time = np.zeros((1,1))

S_1 = np.dot(conj(p_1),q_1)
S_2 = np.dot(conj(p_2),q_2)

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
time = np.array([[0]])

"""
run scipy.integrate ODE solver
"""
"""
this can probably be greatly sped up, and now that the plot function has been moved to SOQplot, 
this needs to be less hard coded
"""

while runODE.successful() and runODE.t<totaltime:
    #print("runODE.t = ",runODE.t)
    check = runODE.integrate(runODE.t+dt)
    #print("runODE.integrate(runODE.t+dt) = ")
    #print(check)
    #print("check = ")
    #print(check)
    results = np.array([check])
    q_1 = quatreal(np.array([results[0,0:4]]))
    q_2 = quatreal(np.array([results[0,4:8]]))
    p_1 = quatreal(np.array([results[0,8:12]]))
    p_2 = quatreal(np.array([results[0,12:16]]))
    S_1 = np.dot(conj(p_1),q_1)
    S_2 = np.dot(conj(p_2),q_2)
    '''if mod(i,10)==0:
        print("q_1[0] = ")
        print(q_1[0])
        print("q_2[0] = ")
        print(q_2[0])
        print("p_1[0] = ")
        print(p_1[0])
        print("conj(p_1)[0] = ")
        print(conj(p_1)[0])
        print("p_2[0] = ")
        print(p_2[0])
        print("conj(p_2)[0] = ")
        print(conj(p_2)[0])
        print("S_1[0] = ")
        print(S_1[0])
        print("S_2[0] = ")
        print(S_2[0])'''
    #print("size =",S_1.size,"S_1x =",S_1[0,1])
    #print("S_1y =",S_1[0,2],"S_1z =",S_1[0,3])
    #S_1_i = S_1[0,1:4]
    #S_2_i = S_2[0,1:4]
    '''if i==0:
        S_1r = S_1[0,0]
        S_1x = S_1[0,1]
        S_1y = S_1[0,2]
        S_1z = S_1[0,3]
        S_2r = S_2[0,0]
        S_2x = S_2[0,1]
        S_2y = S_2[0,2]
        S_2z = S_2[0,3]
        #S_1x = S_1[0,1]
        #S_1y = S_1[0,2]
        #S_1z = S_1[0,3]
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
        time[0,0] = runODE.t
    elif i>0:'''
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
    time = np.append(time,runODE.t)
    #print('time =')
    #print(time)
    i = i + 1
    #print("S_1x.size = ",S_1x.size)
    #time.sleep(1)
    
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
np.savetxt('time.txt',time,delimiter=',')

print('done')

"""
ignore the following code for now
"""

'''
"""
plot figure
"""
plt.figure()

plt.subplot(121)
plt.plot(time,S_1x,label='S_1i',color='red')
plt.plot(time,S_1y,label='S_1j',color='blue')
plt.plot(time,S_1z,label='S_1k',color='green')
plt.xlabel('S_1')
plt.ylabel('time')
plt.legend(loc='best')

plt.subplot(122)
plt.plot(time,S_2x,label='S_2i',color='red')
plt.plot(time,S_2y,label='S_2j',color='blue')
plt.plot(time,S_2z,label='S_2k',color='green')
plt.xlabel('S_2')
plt.ylabel('time')
plt.legend(loc='best')

plt.show()

plots = 0'''

"""
ignore the following code for now
"""

#print('S_plot =',S_plot[:,0:10])
#print('S_plot[plots,0:4] =',S_plot[plots,0:3])
#S_plot = S_plot.transpose()







#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim([-1,1])
#ax.set_ylim([-1,1])
#ax.set_zlim([-1,1])

'''display = np.ceil(totaltime/dt/totalplots)

plt.ion()
plt.figure(figsize=[24,12])
maximum_S = np.sqrt([3*(np.amax(np.absolute(S_plot[:,0:6]))**2)])
print("np.amax(S_plot) = ")
print(np.amax(S_plot))
maximum_q = np.sqrt([3*(np.amax(np.absolute(q_plot[:,0:6]))**2)])
print("np.amax(q_plot) = ")
print(np.amax(q_plot))
print("maximum_q = ")
print(maximum_q)
maximum_p = np.sqrt([3*(np.amax(np.absolute(p_plot[:,0:6]))**2)])
print("maximum_p = ")
print(maximum_p)
print("np.amax(p_plot) = ")
print(np.amax(p_plot))

x = np.zeros((1,2))
y = np.zeros((1,2))
z = np.zeros((1,2))
zvec = np.zeros((1,3))'''
#print("qvec_1 =")
#print(qvec_1)
#print("qvec_2 =")
#print(qvec_2)


'''while plots<totaltime/dt:
    if mod(plots,display) == 0:
        plt.cla()
        ax1 = plt.subplot(131, projection='3d',aspect='equal')
        plt.tight_layout()
        ax1.set_xlim([-maximum_q,maximum_q])
        ax1.set_ylim([-maximum_q,maximum_q])
        ax1.set_zlim([-maximum_q,maximum_q])
        #plt.axis('equal')
        
        #x[0] = np.array(S_plot[plots,0:2])
        #y[0] = np.array(S_plot[plots,2:4])
        #z[0] = np.array(S_plot[plots,4:6])
        #u = np.array([S_plot[plots,0:2]])
        #v = np.array([S_plot[plots,2:4]])
        #w = np.array([S_plot[plots,4:6]])
        vecq_1 = np.append(zvec,q_plot[plots,0:3])
        #print('******')
        print('q_plot[plots,0:3] =')
        print(q_plot[plots,0:3])
        vecq_2 = np.append(zvec,q_plot[plots,3:6])
        print('q_plot[plots,3:6] =')
        print(q_plot[plots,3:6])
        #print('******')
        #vectors = np.array([vecS_1,vecS_2])
        #x,y,z,u,v,w = zip(*vectors)
        vectors_1 = np.array([vecq_1])
        vectors_2 = np.array([vecq_2])
        x_1,y_1,z_1,u_1,v_1,w_1 = zip(*vectors_1)
        x_2,y_2,z_2,u_2,v_2,w_2 = zip(*vectors_2)
        length_1 = np.sqrt((vectors_1[0,3]**2)+(vectors_1[0,4]**2)+(vectors_1[0,5]**2))
        print('length_1 = ')
        print(length_1)
        length_2 = np.sqrt((vectors_2[0,3]**2)+(vectors_2[0,4]**2)+(vectors_2[0,5]**2))
        print('length_2 = ')
        print(length_2)
        #print("x,y,z,u,v,w = ")
        #print(x,y,z,u,v,w)
        #u = np.array([np.sin(plots),np.cos(plots)])
        #v = np.array([np.cos(plots),np.sin(plots)])
        #w = 1
        ax1.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax1.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        ''''''
        ax2 = plt.subplot(132, projection='3d',aspect='equal')
        ax2.set_xlim([-maximum_p,maximum_p])
        ax2.set_ylim([-maximum_p,maximum_p])
        ax2.set_zlim([-maximum_p,maximum_p])
        vecp_1 = np.append(zvec,p_plot[plots,0:3])
        print('p_plot[plots,0:3] =')
        print(p_plot[plots,0:3])
        print('p_plot[plots,3:6] =')
        print(p_plot[plots,3:6])
        vecp_2 = np.append(zvec,p_plot[plots,3:6])
        vectors_1 = np.array([vecp_1])
        vectors_2 = np.array([vecp_2])
        x_1,y_1,z_1,u_1,v_1,w_1 = zip(*vectors_1)
        x_2,y_2,z_2,u_2,v_2,w_2 = zip(*vectors_2)
        length_1 = np.sqrt((vectors_1[0,3]**2)+(vectors_1[0,4]**2)+(vectors_1[0,5]**2))
        length_2 = np.sqrt((vectors_2[0,3]**2)+(vectors_2[0,4]**2)+(vectors_2[0,5]**2))
        print('length_1 = ')
        print(length_1)
        ax2.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax2.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        print('length_2 = ')
        print(length_2)
        #plt.axis('equal')
        ''''''
        ax3 = plt.subplot(133, projection='3d',aspect='equal')
        ax3.set_xlim([-maximum_S,maximum_S])
        ax3.set_ylim([-maximum_S,maximum_S])
        ax3.set_zlim([-maximum_S,maximum_S])
        vecS_1 = np.append(zvec,[S_plot[plots,0],S_plot[plots,1],S_plot[plots,2]])
        print('S_plot[plots,0:3] =')
        print(S_plot[plots,0:3])
        print('S_plot[plots,3:6] =')
        print(S_plot[plots,3:6])
        vecS_2 = np.append(zvec,[S_plot[plots,3],S_plot[plots,4],S_plot[plots,5]])
        vectors_1 = np.array([vecS_1])
        vectors_2 = np.array([vecS_2])
        x_1,y_1,z_1,u_1,v_1,w_1 = zip(*vectors_1)
        x_2,y_2,z_2,u_2,v_2,w_2 = zip(*vectors_2)
        #magnitude_1 = np.sqrt(S_plot[plots,0]**2 + S_plot[plots,1]**2 +S_plot[plots,2]**2 + S_plot[plots,6]**2)
        #print("magnitude_1 = ")
        #print(magnitude_1)
        q_1 = np.append(q_plot[plots,0:3],q_plot[plots,6])
        q_2 = np.append(q_plot[plots,3:6],q_plot[plots,7])
        p_1 = np.append(p_plot[plots,0:3],p_plot[plots,6])
        p_2 = np.append(p_plot[plots,3:6],p_plot[plots,7])
        length_1 = np.sqrt((vectors_1[0,3]**2)+(vectors_1[0,4]**2)+(vectors_1[0,5]**2))
        length_2 = np.sqrt((vectors_2[0,3]**2)+(vectors_2[0,4]**2)+(vectors_2[0,5]**2))
        print('length_1 = ')
        print(length_1)
        print('length_2 = ')
        print(length_2)
        ax3.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax3.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        #plt.axis('equal')
        #print('plots =',plots)
        plt.pause(.0001)
        #time.sleep(0.05)
        plt.draw()
    plots = plots+1
print("stopped")
plt.ioff()  
plt.show()'''

    
#test = SOQsys(t,input,omega_0,alpha)