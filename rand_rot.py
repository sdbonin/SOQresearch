'''
Code synopsis:
This code takes some quaternion q, and rotates it (RqR_dag) with a random unit quaternion R.
These rotations are repeated with different random R's to build up a statistical 
description of 'random quaternion rotations'. Ultimitely this code will show the distribution
of angles between the original q and the rotated q's.
'''

import numpy as np
import matplotlib.pyplot as plt

N=int(1e6) #Number of random quaternions to generate
rand_min = 0
rand_max = 1 #Choosing the range of random numbers
seed = 1 #Used to keep inter-run results consistent

#Defining a function that performs quaternion multiplication
def quat_mult(q1,q2) :
  qp = np.zeros(4) #Product of q1 and q2
  Q = np.zeros((4,4)) #An intermediate matrix for keeping track of terms 
  for i in range(4):
    for j in range(4):
      Q[i,j] = q1[i]*q2[j]
  #Using 'quaternion mult.' figure from https://en.wikipedia.org/wiki/Quaternion
  qp[0] = Q[0,0] - Q[1,1] - Q[2,2] - Q[3,3]
  qp[1] = Q[0,1] + Q[1,0] + Q[2,3] - Q[3,2]
  qp[2] = Q[0,2] + Q[2,0] + Q[3,1] - Q[1,3]
  qp[3] = Q[0,3] + Q[3,0] + Q[1,2] - Q[2,1]
  return qp

#Generating array of uniform random real numbers
np.random.seed(seed)
r = np.random.uniform(rand_min,rand_max,(3,N))

#Generating array of uniform random quaternions
RR = np.zeros((4,N)) 
for i in range(N):
  aux_1 = np.sqrt(1-r[0,i])
  aux_2 = np.sqrt(r[0,i])
  aux_3 = 2*np.pi*r[1,i]
  aux_4 = 2*np.pi*r[2,i]
  RR[0,i] = aux_2*np.cos(aux_4)
  RR[1,i] = aux_1*np.sin(aux_3)
  RR[2,i] = aux_1*np.cos(aux_3)
  RR[3,i] = aux_2*np.sin(aux_4)

#Taking square root of random quaternions
R = np.zeros((4,N)) #The square root of RR
for i in range(N) :
  aux_1 = RR[0,i]**2 + RR[1,i]**2 + RR[2,i]**2 + RR[3,i]**2
  aux_2 = 0.5*(RR[0,i] + np.sqrt(aux_1))
  aux_3 = np.sqrt(aux_2)
  R[0,i] = aux_3
  R[1:,i] = RR[1:,i]/(2*aux_3)

#Rotating q for each random quaternion R
q = np.array([0,1,0,0]) #I chose this particular q arbitrarily
q_new = np.zeros((4,N))
dag = np.array([1,-1,-1,-1]) #Used to take ajoint of quaternions
for i in range(N):
	R_dag = np.multiply(R[:,i],dag)
	aux_1 = quat_mult(q,R_dag)
	q_new[:,i] = quat_mult(R[:,i],aux_1)

#Finding angles between the original quaternion (q) and the rotated quaternions (q_new)
angles = np.zeros(N)
for i in range(N):
  q_new_dag = np.multiply(q_new[:,i],dag)
  q_dif = quat_mult(q,q_new_dag)
  angles[i] = np.arccos(q_dif[0])

#Plotting resulting angles with a fit function overlaid
n,bins,batches = plt.hist(angles,bins=100)
y = 8/np.sqrt(27)*max(n)*np.sin(bins)*np.cos(bins/2)**2 #This is the fit function
plt.plot(bins,y,'--r',lw=4)
plt.text(1.7,0.9*max(n),r'$y=Asin(\theta)cos^2\left(\frac{\theta}{2}\right)$',fontsize=24)
plt.xlabel(r'$\theta$ (rad)',fontsize=24)
plt.ylabel('Bin Count',fontsize=24)
plt.savefig('fig_rand_rot.png')
plt.show()
