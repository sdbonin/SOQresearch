# -*- coding: utf-8 -*-
"""
This is out code from last semester where we were plotting an analytical
solution that we found. Throughout the code a 4-vector representation is used
for quaternions which made things a bit clunky. Some functions are not tested,
so test things before using them. The plotting is probably the most useful
portion of this code and I (Eric) would like to transfer it over to the simpler
matrix math.
"""

import matplotlib.pyplot as plt
import numpy as np


def qmag(q):
    '''
    This function outputs the magnitude of a quaternion.
    '''
    return np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)


def qnorm(q):
    '''
    This function takes a quaternion formatted as a 4-element np.array and
    normalizes it such that the output has magnitude 1
    '''
    return q/qmag(q)


def qmult(q1,q2):
    '''
    This function takes two inputs q1 and q2 and returns the product q1*q2
    q = np.array([q_I, q_n, q_m, q_l])
    '''
    A = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    B = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    C = q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3]
    D = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
    return np.array([A,B,C,D])

"""
# I was really tired when I wrote these two... They don't work. I need to make better versions.

def qexp_p(q):
    '''
    output = exp(+q)
    note: I wouldn't trust this one until I talk to Wharton about it
    '''
    if q[0]==0:
        return qnorm(q)*np.cos(qmag(q)) + qnorm(q)*np.sin(qmag(q))
    else:
        print("I don't know how to use Euler's Identity for impure quaternions. Sorry.")

def qexp_m(q):
    '''
    output = exp(-q)
    note: I wouldn't trust this one until I talk to Wharton about it
    additional note: yeah, I was really tired when I made this. It makes no sense.
    '''
    if q[0]==0:
        return qnorm(q)*np.cos(qmag(q)) - qnorm(q)*np.sin(qmag(q))
    else:
        print("I don't know how to use Euler's Identity for impure quaternions. Sorry.")

"""




def q(q1,q2,omega,alpha,t):
    """
    I'm pretty sure this does what it's supposed to do, which is to output
    the solution we found.
    q = q1(cos(t*w-)+n*sin(t*w-)) + q2(cos(t*w+)-n*sin(t*w+))
    """
    return qnorm(qmult(q1,[np.cos((omega-alpha)*t),np.sin((omega-alpha)*t),0,0]) + qmult(q2,[np.cos((omega+alpha)*t),-np.sin((omega+alpha)*t),0,0]))


def qorbitloop(q1,q2,omega,alpha,tmin,tmax,tstep):
    '''
    This function utilizes the q(t) function above to create orbit arrays
    corresponding to times governed by the time parameters. The final output
    array has 4 subarrays, each corresponding to [I,n,m,l] where each of those
    elements is itself an array with values calculated at different time steps.
    note: THis function has been changed recently and has not been fully tested
    the original version is commented out with the new stuff underneath
    I don't think I actually need this function now that I made things arrays.
    '''
    t = tmin
    i = 0
    n = round((tmax-tmin)*tstep**(-1))
    '''
    I = np.zeros(n)
    N = np.zeros(n)
    M = np.zeros(n)
    L = np.zeros(n)
    '''
    I = []
    N = []
    M = []
    L = []
    while t<tmax:
        '''
        I[i],N[i],M[i],L[i] = q(q1,q2,omega,alpha,t)
        '''
        I.append(q(q1,q2,omega,alpha,t)[0])
        N.append(q(q1,q2,omega,alpha,t)[1])
        M.append(q(q1,q2,omega,alpha,t)[2])
        L.append(q(q1,q2,omega,alpha,t)[3])
        t+=tstep
        i+=1
    return np.array([I,N,M,L])




def qplot_vect(q):
    '''
    This function plots a quaternion as a vector on one of two 3d plots
    This requires that the axes ax and bx already be defined
    '''
    #converting q into a plottable vector
    X,Y,Z,U,V,W = q[1],q[2],q[3],q[1],q[2],q[3]
    #plotting q
    if q[0]<0:
        ax.quiver(X,Y,Z,U,V,W)
    else:
        bx.quiver(X,Y,Z,U,V,W)


def qplot_orbit(I,N,M,L):
    '''
    note to self: put come comments here
    also, make this plot lines, not points
    '''
    for i in range(0,len(I)):
        if I[i]<0:
            ax.scatter(N[i],M[i],L[i])
        else:
            bx.scatter(N[i],M[i],L[i])


def qplot_orbit_color(q):
    '''
    This plots the result with the real component mapped to the color
    '''
    I,N,M,L = q
    ax.scatter(N,L,M,c=I,cmap='bwr')
 

'''
#AXES DEFINITION FOR A SUBPLOT THINGAMAGIGGY
#currently not being used in favor of a colored plot setup
#eventually all of these axis definition stuff will be put inside the plotting

#defining the axes
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
bx = fig.add_subplot(122, projection='3d')
#scaling the axes
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
bx.set_xlim([-1,1])
bx.set_ylim([-1,1])
bx.set_zlim([-1,1])
'''



#AXES DEFINITION FOR A SINGLE PLOT THINGAMAGIGGY

#defining the axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#scaling the axes
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])



# Defining initial conditions and whatnot
q1 = np.array([0,0,1,0])
q2 = np.array([0,1,1,1])
omega = 0.5
alpha = 0.1
t = np.linspace(0,500,6370)
#tmin = 0
#tmax = 500
#tstep = 0.2



orb = q(q1,q2,omega,alpha,t)#qorbitloop(q1,q2,omega,alpha,tmin,tmax,tstep)


qplot_orbit_color(orb)



#display the plot
plt.show()
#print(qnorm(q))