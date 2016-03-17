"""
Work in progess - sdbonin
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, ion, show
import time
from numpy import mod as mod
import matplotlib.animation as animation

omega_0 = 1
alpha = 0.001
arguments = np.array([[omega_0, alpha]])
dt = .01
totaltime = 20
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
    
def hconj(q):
    q = np.array([q[0]])
    q[0,1]=-q[0,1]
    q[0,2]=-q[0,2]
    q[0,3]=-q[0,3]
    hermitianconjugate = quatreal(q)
    return hermitianconjugate    

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
p_1 = np.random.randn(4,4)
p_2 = np.random.randn(4,4)
pmat_1 = quatreal(p_1)
pmat_2 = quatreal(p_2)

condition_1 = np.dot(hconj(pmat_1),q_1)
condition_2 = np.dot(hconj(pmat_2),q_2)

p_1[0] = condition_1[0,0]
p_2[0] = condition_2[0,0]

p_1 = quatreal(p_1)
p_2 = quatreal(p_2)

'24-27,28-31'
pdot_1 = np.zeros((4,4))
pdot_2 = np.zeros((4,4))

#initialvalues = np.append(q_1[0],[q_2[0],qdot_1[0],qdot_2[0],p_1[0],p_2[0],pdot_1[0],pdot_2[0]])
initialvalues = np.append(q_1[0],[q_2[0],p_1[0],p_2[0]])

def mag(q):
    magnitude = np.sqrt(q[0,0]**2+q[0,1]**2+q[0,2]**2+q[0,3]**2)
    return magnitude

def SOQsys(time,input,arguments):
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
    divisor = 1-(alpha**2)*((mag(q_1)**2)*(mag(q_2)**2))
    #print("divisor =")
    #print(divisor)
    if divisor == 0:
        output = "failure"
    else:
        dot1 = np.dot(q_1,hconj(q_2))
        dot2 = np.dot(dot1,p_2)
        top1 = p_1 - alpha*dot2
        q_1_dt = top1*(1/divisor)
        #q_1_dt = quatreal(q_1_dt)
        #
        dot1 = np.dot(q_2,hconj(q_1))
        dot2 = np.dot(dot1,p_1)
        top2 = p_2 - alpha*dot2
        q_2_dt = top2*(1/divisor)
        #q_2_dt = quatreal(q_2_dt)
        #
        #
        dot1 = np.dot(q_1_dt,hconj(q_2_dt))
        dot2 = np.dot(dot1,q_2)
        p_1_dt = -(omega_0**2)*q_1 + alpha*dot2
        #p_1_dt = quatreal(p_1_dt)
        #
        dot1 = np.dot(q_2_dt,hconj(q_1_dt))
        dot2 = np.dot(dot1,q_1)
        p_2_dt = -(omega_0**2)*q_2 + alpha*dot2
        #p_2_dt = quatreal(p_2_dt)
        #
        output = np.append(q_1_dt[0],[q_2_dt[0],p_1_dt[0],p_2_dt[0]])
    #print("    time = ",time)
    return output

runODE = ode(SOQsys).set_integrator('vode',method='bdf',with_jacobian=False,max_step=dt/100)
runODE.set_initial_value(initialvalues,t0).set_f_params(arguments)

i = 0
S_1x = np.zeros((1,1))
S_2x = np.zeros((1,1))

while runODE.successful() and runODE.t<totaltime:
    print("runODE.t = ",runODE.t)
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
    S_1 = np.dot(hconj(p_1),q_1)
    S_2 = np.dot(hconj(p_2),q_2)
    '''if mod(i,10)==0:
        print("q_1[0] = ")
        print(q_1[0])
        print("q_2[0] = ")
        print(q_2[0])
        print("p_1[0] = ")
        print(p_1[0])
        print("hconj(p_1)[0] = ")
        print(hconj(p_1)[0])
        print("p_2[0] = ")
        print(p_2[0])
        print("hconj(p_2)[0] = ")
        print(hconj(p_2)[0])
        print("S_1[0] = ")
        print(S_1[0])
        print("S_2[0] = ")
        print(S_2[0])'''
    #print("size =",S_1.size,"S_1x =",S_1[0,1])
    #print("S_1y =",S_1[0,2],"S_1z =",S_1[0,3])
    #S_1_i = S_1[0,1:4]
    #S_2_i = S_2[0,1:4]
    if i==0:
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
    elif i>0:
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
    i = i + 1
    #print("S_1x.size = ",S_1x.size)
    #time.sleep(1)
    

S_plot = np.zeros((S_1x.size,8))
S_plot[:,0] = S_1x
S_plot[:,1] = S_1y
S_plot[:,2] = S_1z
S_plot[:,3] = S_2x
S_plot[:,4] = S_2y
S_plot[:,5] = S_2z
S_plot[:,6] = S_1r
S_plot[:,7] = S_2r

q_plot = np.zeros((q_1x.size,8))
q_plot[:,0] = q_1x
q_plot[:,1] = q_1y
q_plot[:,2] = q_1z
q_plot[:,3] = q_2x
q_plot[:,4] = q_2y
q_plot[:,5] = q_2z
q_plot[:,6] = q_1r
q_plot[:,7] = q_2r

p_plot = np.zeros((p_1x.size,8))
p_plot[:,0] = p_1x
p_plot[:,1] = p_1y
p_plot[:,2] = p_1z
p_plot[:,3] = p_2x
p_plot[:,4] = p_2y
p_plot[:,5] = p_2z
p_plot[:,6] = p_1r
p_plot[:,7] = p_2r


plots = 0

#print('S_plot =',S_plot[:,0:10])
#print('S_plot[plots,0:4] =',S_plot[plots,0:3])
#S_plot = S_plot.transpose()







#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim([-1,1])
#ax.set_ylim([-1,1])
#ax.set_zlim([-1,1])

totalplots = 50
display = np.ceil(totaltime/dt/totalplots)

plt.ion()
plt.show()
plt.figure(figsize=[24,12])
maximum_S = np.sqrt(3*(np.amax(S_plot)**2))
maximum_q = np.sqrt(3*(np.amax(q_plot)**2))
maximum_p = np.sqrt(3*(np.amax(p_plot)**2))

'''#plt.figure(figsize=(8,6),dpi=80)
#plt.set_size_inches(18.5, 10.5)
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
'''
x = np.zeros((1,2))
y = np.zeros((1,2))
z = np.zeros((1,2))
zvec = np.zeros((1,3))
#print("qvec_1 =")
#print(qvec_1)
#print("qvec_2 =")
#print(qvec_2)

while plots<totaltime/dt:
    if mod(plots,display) == 0:
        plt.clf()
        ax = plt.subplot(131, projection='3d',aspect='equal')
        plt.tight_layout()
        #plt.axis('equal')
        ax.set_xlim([-maximum_q,maximum_q])
        ax.set_ylim([-maximum_q,maximum_q])
        ax.set_zlim([-maximum_q,maximum_q])
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
        ax.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        ''''''
        ax = plt.subplot(132, projection='3d',aspect='equal')
        ax.set_xlim([-maximum_p,maximum_p])
        ax.set_ylim([-maximum_p,maximum_p])
        ax.set_zlim([-maximum_p,maximum_p])
        vecp_1 = np.append(zvec,p_plot[plots,0:3])
        print('p_plot[plots,0:3] =')
        print(p_plot[plots,0:3])
        vecp_2 = np.append(zvec,p_plot[plots,3:6])
        vectors_1 = np.array([vecp_1])
        vectors_2 = np.array([vecp_2])
        x_1,y_1,z_1,u_1,v_1,w_1 = zip(*vectors_1)
        x_2,y_2,z_2,u_2,v_2,w_2 = zip(*vectors_2)
        length_1 = np.sqrt((vectors_1[0,3]**2)+(vectors_1[0,4]**2)+(vectors_1[0,5]**2))
        length_2 = np.sqrt((vectors_2[0,3]**2)+(vectors_2[0,4]**2)+(vectors_2[0,5]**2))
        ax.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        #plt.axis('equal')
        ''''''
        ax = plt.subplot(133, projection='3d',aspect='equal')
        ax.set_xlim([-maximum_S,maximum_S])
        ax.set_ylim([-maximum_S,maximum_S])
        ax.set_zlim([-maximum_S,maximum_S])
        vecS_1 = np.append(zvec,[S_plot[plots,0],S_plot[plots,1],S_plot[plots,2]])
        print('S_plot[plots,0:3] =')
        print(S_plot[plots,0:3])
        vecS_2 = np.append(zvec,[S_plot[plots,3],S_plot[plots,4],S_plot[plots,5]])
        vectors_1 = np.array([vecS_1])
        vectors_2 = np.array([vecS_2])
        x_1,y_1,z_1,u_1,v_1,w_1 = zip(*vectors_1)
        x_2,y_2,z_2,u_2,v_2,w_2 = zip(*vectors_2)
        length_1 = np.sqrt((vectors_1[0,3]**2)+(vectors_1[0,4]**2)+(vectors_1[0,5]**2))
        length_2 = np.sqrt((vectors_2[0,3]**2)+(vectors_2[0,4]**2)+(vectors_2[0,5]**2))
        ax.quiver(0,0,0,u_1,v_1,w_1,pivot='tail',length=length_1,colors='r')
        ax.quiver(0,0,0,u_2,v_2,w_2,pivot='tail',length=length_2,colors='b')
        #plt.axis('equal')
        #print('plots =',plots)
        plt.draw()
        plt.pause(.0001)
    plots = plots+1
print("stopped")
plt.ioff()  
plt.show()
    
#test = SOQsys(t,input,omega_0,alpha)