'''
Michael Mulanix -- San Jose State University
19 November 2016

This script contains 4 operator functions, the square root of a quaternion and bi-quaternion, and the product of two quaternions and two bi-quaternions.
'''

import numpy as np
from scipy.optimize import fsolve
#The classes/data-structures to hold the quaternions and bi-quaternions
class quat :
  def __init__(self,a,i,j,k) :
    self.a = float(a) #Real axis
    self.i = float(i) #3 independent imaginary axes
    self.j = float(j)
    self.k = float(k)
  def __str__(self) :
    return "(%f) + (%f)i + (%f)j + (%f)k" % (self.a,self.i,self.j,self.k)

class biquat :
  def __init__(self,a,i,j,k,aa,ii,jj,kk) :
    self.a = float(a) #Scalar of quaternion
    self.i = float(i) #Vector components of quaternion
    self.j = float(j)
    self.k = float(k)
    self.aa = float(aa) #Scalar of 'imaginary' quaternion
    self.ii = float(ii) #Vector components of 'imaginary' quaternion
    self.jj = float(jj)
    self.kk = float(kk)
  def __str__(self) :
    return "[(%f) + (%f)i + (%f)j + (%f)k] + i[(%f) + (%f)i + (%f)j + (%f)k]" % (self.a,self.i,self.j,self.k,self.aa,self.ii,self.jj,self.kk)

#Quaternion Operations
def quat_mult(q1,q2) :
  #Ref: from https://en.wikipedia.org/wiki/Quaternion
  a = q1.a*q2.a - q1.i*q2.i - q1.j*q2.j - q1.k*q2.k
  i = q1.a*q2.i + q1.i*q2.a + q1.j*q2.k - q1.k*q2.j
  j = q1.a*q2.j + q1.j*q2.a + q1.k*q2.i - q1.i*q2.k
  k = q1.a*q2.k + q1.k*q2.a + q1.i*q2.j - q1.j*q2.i
  return quat(a,i,j,k)

def quat_sqrt(q) :
  #Expand p^2, set equal to q, invert ==> terms below
  aux_1 = q.a**2 + q.i**2 + q.j**2 + q.k**2
  aux_2 = 0.5*(q.a + np.sqrt(aux_1))
  aux_3 = 2*np.sqrt(aux_2)
  return quat(aux_3/2,q.i/aux_3,q.j/aux_3,q.k/aux_3)


#Bi-Quaternion Operations
def biquat_mult(q1,q2) :
  #By writing q1=(q1r + i*q1i) and q2=(q2r + i*q2i) I can use my previous function quat_mult to make this process simpler
  q1r = quat(q1.a,q1.i,q1.j,q1.k) #Bi-quaternion 1 real
  q2r = quat(q2.a,q2.i,q2.j,q2.k) #Bi-quaternion 2 real
  q1i = quat(q1.aa,q1.ii,q1.jj,q1.kk) #Bi-quaternion 1 imaginary
  q2i = quat(q2.aa,q2.ii,q2.jj,q2.kk) #Bi-quaternion 2 imaginary
  aux_1 = quat_mult(q1r,q2r) #Four terms you get from expanding (q1r + i*q1i)(q2r + i*q2i)
  aux_2 = quat_mult(q1i,q2i)
  aux_3 = quat_mult(q1r,q2i)
  aux_4 = quat_mult(q1i,q2r)
                 ### qp.a ###    ### qp.i ###    ### qp.j ###    ### qp.k ###     ### qp.aa ###    ### qp.ii ###      ### qp.jj ###     ### qp.kk ###
  return biquat(aux_1.a-aux_2.a,aux_1.i-aux_2.i,aux_1.j-aux_2.j,aux_1.k-aux_2.k,aux_3.a + aux_4.a,aux_3.i + aux_4.i,aux_3.j + aux_4.j,aux_3.k + aux_4.k,)

def biquat_sqrt(q) :
  #Ref: 'Computing the square root of a bi-quaternion'
  #A bunch of intermediate variables
  mu = (q.i*q.ii) + (q.j*q.jj) + (q.k*q.kk)
  nu = (q.i**2 + q.j**2 + q.k**2) - (q.ii**2 + q.jj**2 + q.kk**2)
  theta=q.a**2 + q.aa**2
  phi = q.a**2 - q.aa**2
  gamma = nu**2 + 4*mu**2
  delta = nu**2 - 4*mu**2
  #Defining the root equation from eq. 17
  f = lambda x : 36.0*x**(16) - 256.0*theta*x**(12) + 4.0*(32.0*(np.sqrt((gamma+delta)*phi**2) + np.sqrt((theta+phi)*(theta-phi)*(gamma-delta)))/np.sqrt(2) -3.0*gamma)*x**8 - 16.0*theta*gamma*x**4 + gamma*2
  x_range = np.max([np.abs(q.a),np.abs(q.aa)])
  xr = fsolve(f,[-1.0*x_range,x_range]) #Roots of the polynomial
  print('**** Roots:',xr)
  solutions = []
  for m in range(len(xr)) :
    alpha = xr[m]
    alpha_sq = xr[m]**2
    aux_2 = 4.0*alpha_sq*(2.0*mu*q.a - (4.0*alpha_sq**2 + nu)*q.aa) #Numerator of eq. 14
    aux_3 = 4.0*mu**2 - (4.0*alpha_sq**2 + nu)*(4*alpha_sq**2 - nu) #Denominator of eq. 14
    beta = 0.5*np.arcsin(aux_2/aux_3) #Inverting eq. 14
    print('*** Beta: ',beta*180/3.1416)
    a = alpha*np.cos(beta) #eq. 11
    c = alpha*np.sin(beta)
    aux_4 = 2.0*(a**2 + c**2) #Denominator of eq. 7 & 8
    b1 = (a*q.i + c*q.ii)/aux_4 #Eq. 7
    b2 = (a*q.j + c*q.jj)/aux_4
    b3 = (a*q.k + c*q.kk)/aux_4
    d1 = (a*q.ii - c*q.i)/aux_4 #Eq. 8
    d2 = (a*q.jj - c*q.j)/aux_4
    d3 = (a*q.kk - c*q.k)/aux_4
    solutions.append(biquat(a,b1,b2,b3,c,d1,d2,d3))
  return solutions

#################################
## Checking the Implementation ##
#################################
print('Checking Quaternion Operations...')
test = quat(1,2,3,4)
root = quat_sqrt(test)
print('    Initial Q: ',test)
print('    Sqrt(Q): ',root)
print('    (Sqrt(Q))**2: ',quat_mult(root,root))
print('')
print('')
print('Checking Bi-Quaternion Operations...')
test = biquat(1,2,3,4,5,6,7,8)
print('    Initial Q: ',test)
print('')
roots = biquat_sqrt(test)
print('    There are {a} roots: '.format(a=len(roots)))
print('')
for root in roots :
  print('    Sqrt(Q): ',root)
#  print(root.a**2 - root.aa**2 + (root.i**2 + root.j**2 + root.k**2) - (root.ii**2 + root.jj**2 + root.kk**2))
#  print(2*root.a*root.aa + 2*(root.i*root.ii + root.j*root.jj + root.k*root.kk))
#  print(2*root.a*root.i - 2*root.aa*root.ii)
  print('    (Sqrt(Q))**2: ',biquat_mult(root,root))
  print('')
