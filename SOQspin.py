"""
Work in progess - sdbonin
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import time

omega_0 = 1
alpha = 0.001
arguments = np.array([[omega_0, alpha]])
dt = .01
totaltime = 5
t0 = 0

def quatreal(q):
    '''
    Turn a 4-vector quaternion into a real matrix
    https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
    '''
    a = q[0,0]
    b = q[0,1]
    c = q[0,2]
    d = q[0,3]
    amat = a*np.identity(4)
    bmat = b*np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,1,0]])
    cmat = c*np.array([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
    dmat = d*np.array([[0,0,0,1],[0,0,-1,0],[0,1,0,0],[-1,0,0,0]])
    return amat+bmat+cmat+dmat
    

qvec_1 = 1*np.random.randn(1,4)
qvec_2 = 1*np.random.randn(1,4)
'''qvec_1 = np.array([[.1,.2,.3,.4]])
qvec_2 = np.array([[.3,-.3,-.3,.3]])'''

print("qvec_1 =")
print(qvec_1)
print("qvec_2 =")
print(qvec_2)
'''qvec_1 = -0.1*np.ones((1,4))
qvec_2 = 0.1*np.ones((1,4))'''

'0-3,4-7'
q_1 = quatreal(qvec_1)
q_2 = quatreal(qvec_2)

'8-11,12-15'
qdot_1 = np.zeros((4,4))
qdot_2 = np.zeros((4,4))

'16-19,20-23'
p_1 = np.zeros((4,4))
p_2 = np.zeros((4,4))

'24-27,28-31'
pdot_1 = np.zeros((4,4))
pdot_2 = np.zeros((4,4))

#initialvalues = np.append(q_1[0],[q_2[0],qdot_1[0],qdot_2[0],p_1[0],p_2[0],pdot_1[0],pdot_2[0]])
initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])

def mag(q):
    magnitude = 1/np.sqrt(q[0,0]**2+q[0,1]**2+q[0,2]**2+q[0,3]**2)
    return magnitude
    
def hconj(q):
    q = np.array([q[0]])
    q[0,1]=-q[0,1]
    q[0,2]=-q[0,2]
    q[0,3]=-q[0,3]
    hermitianconjugate = quatreal(q)
    return hermitianconjugate

def SOQsys(t,input,arguments):
    '''
    initialize real matrices from input
    '''
    q_1 = quatreal(np.array([input[0:4]]))
    q_2 = quatreal(np.array([input[4:8]]))
    p_1 = quatreal(np.array([input[8:12]]))
    p_2 = quatreal(np.array([input[12:16]]))
    
    omega_0 = arguments[0,0]
    alpha = arguments[0,1]
    '''
    matrix operations
    '''
    divisor = (1-(alpha**2)*(mag(q_1)**2)*(mag(q_2)**2))
    dot1 = np.dot(q_1,hconj(q_2))
    dot2 = np.dot(dot1,p_2)
    top1 = p_1 - alpha*dot2
    q_1_dt = top1*(1/divisor)
    #
    dot1 = np.dot(q_2,hconj(q_1))
    dot2 = np.dot(dot1,p_1)
    top2 = p_2 - alpha*dot2
    q_2_dt = top2*(1/divisor)
    #
    dot1 = np.dot(q_1_dt,hconj(q_2_dt))
    dot2 = np.dot(dot1,q_2)
    p_1_dt = -(omega_0**2)*q_1 + alpha*dot2
    #
    dot1 = np.dot(q_2_dt,hconj(q_1_dt))
    dot2 = np.dot(dot1,q_1)
    p_2_dt = -(omega_0**2)*q_2 + alpha*dot2
    output = np.append(q_1_dt[0],[q_2_dt[0],p_1_dt[0],p_2_dt[0]])
    return output

runODE = ode(SOQsys).set_integrator('vode',method='adams',with_jacobian=False)
runODE.set_initial_value(initialvalues,t0).set_f_params(arguments)

i = 0
S_1x = np.zeros((1,1))
S_2x = np.zeros((1,1))

while runODE.successful() and runODE.t<totaltime:
    check = runODE.integrate(runODE.t+dt)
    #print("check = ")
    #print(check)
    results = np.array([check])
    q_1 = quatreal(np.array([results[0,0:4]]))
    q_2 = quatreal(np.array([results[0,4:8]]))
    p_1 = quatreal(np.array([results[0,8:12]]))
    p_2 = quatreal(np.array([results[0,12:16]]))
    S_1 = np.dot(hconj(p_1),q_1)
    S_2 = np.dot(hconj(p_2),q_2)
    print("size =",S_1.size,"S_1x =",S_1[0,1])
    print("S_1y =",S_1[0,2],"S_1z =",S_1[0,3])
    #S_1_i = S_1[0,1:4]
    #S_2_i = S_2[0,1:4]
    if i==0:
        S_1x = S_1[0,1]
        S_1y = S_1[0,2]
        S_1z = S_1[0,3]
        S_2x = S_2[0,1]
        S_2y = S_2[0,2]
        S_2z = S_2[0,3]
        #S_1x = S_1[0,1]
        #S_1y = S_1[0,2]
        #S_1z = S_1[0,3]
    elif i>0:
        S_1x = np.append(S_1x,S_1[0,1])
        S_1y = np.append(S_1y,S_1[0,2])
        S_1z = np.append(S_1z,S_1[0,3])
        S_2x = np.append(S_2x,S_2[0,1])
        S_2y = np.append(S_2y,S_2[0,2])
        S_2z = np.append(S_2z,S_2[0,3])
    i = i + 1
    print(i)

S_plot = np.zeros((S_1x.size,6))
S_plot[:,0] = S_1x
S_plot[:,1] = S_1y
S_plot[:,2] = S_1z
S_plot[:,3] = S_2x
S_plot[:,4] = S_2y
S_plot[:,5] = S_2z

plots = 0

print('S_plot =',S_plot[:,0:10])
print('S_plot[plots,0:4] =',S_plot[plots,0:3])
#S_plot = S_plot.transpose()







'''ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])'''



plt.ion()
plt.show()

x = np.zeros((1,2))
y = np.zeros((1,2))
z = np.zeros((1,2))
zvec = np.zeros((1,3))
print("qvec_1 =")
print(qvec_1)
print("qvec_2 =")
print(qvec_2)

while plots<=totaltime/dt:
    plt.clf()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    '''x[0] = np.array(S_plot[plots,0:2])
    y[0] = np.array(S_plot[plots,2:4])
    z[0] = np.array(S_plot[plots,4:6])'''
    '''u = np.array([S_plot[plots,0:2]])
    v = np.array([S_plot[plots,2:4]])
    w = np.array([S_plot[plots,4:6]])'''
    vecS_1 = np.append(zvec,S_plot[plots,0:3])
    print('******')
    print('S_plot[plots,0:3] =',S_plot[plots,0:3])
    vecS_2 = np.append(zvec,S_plot[plots,3:6])
    print('S_plot[plots,3:6] =',S_plot[plots,3:6])
    print('******')
    vectors = np.array([vecS_1,vecS_2])
    x,y,z,u,v,w = zip(*vectors)
    #u = np.array([np.sin(plots),np.cos(plots)])
    #v = np.array([np.cos(plots),np.sin(plots)])
    #w = 1
    ax.quiver(0,0,0,u,v,w,pivot='tail',length=1.5)      
    plots = plots+1
    print('plots =',plots)
    plt.draw()
    plt.pause(1)
  
plt.ioff()  
plt.show()

    
#test = SOQsys(t,input,omega_0,alpha)