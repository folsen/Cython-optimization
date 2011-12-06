from __future__ import division
from scipy.linalg import (norm, solve, cholesky, solve_triangular)
import numpy as np
cimport numpy as np

cdef extern from "math.h":
  double abs(double)

class BFGS (object):

  def __init__(self, g, linesearch='None'):
    self.g = g
    self.linesearch=linesearch

  def __call__(self, np.ndarray x):
    # Kor loopen
    return runbfgs(self.g, x, self.linesearch)

cdef np.ndarray runbfgs(g, np.ndarray x, char* linesearch):
  cdef int maxit = 100
  cdef double tol = 1e-8
  cdef double alpha 
  cdef char* ls = linesearch
  cdef char* none = "None"
  cdef char* exact = "Exact"
  #g = self.g

  cdef np.ndarray H, xold, s

  for i in range(0,maxit):
    if i == 0:
      # This function and the one below can be integrated into one
      H = initial_h(g, x)
    else:
      # Update - this function and the one above can be integrated into one
      H = update(g, H, x, xold, alpha, s)

    s = chol_solve(H, g(x))
    if ls == none:
      alpha = 1
    else:
      alpha = exact_linesearch(g,x,s,0)

    xold = x
    x = x-alpha*s
    if norm(np.array(x)-np.array(xold)) < tol:
      return x

  raise Exception("Didn't converge.")

cdef np.ndarray initial_h(g, np.ndarray x):
  cdef double h = 1e-8
  cdef np.ndarray Gb, G
  # Finite difference approximation
  Gb = np.array([(g(x + ei * h) - g(x))/h for ei in np.identity(np.size(x))])
  G = (1./2.) * (Gb + Gb.T)
  return G

cdef double exact_linesearch(g, np.ndarray x, np.ndarray s, double a_guess):
  cdef double a, ll, lu, tol, tmp

  a = a_guess
  ll = 0
  lu = 1
  tol = 1e-5
  while not cfprime(g,x,-s,lu) > 0:
    lu = 2*lu

  while True:  
    a = (ll+lu)/2
    tmp = cfprime(g,x,-s,a)
    if tmp > 0:
      lu = a
    elif tmp < 0:
      ll = a
    else:
      break

    if abs(ll-lu) < tol:
      break
  
  return a

cdef double cfprime(g, np.ndarray x, np.ndarray s, double a):
  return np.dot(g(x+a*s),s)

cdef np.ndarray chol_solve(np.ndarray H, np.ndarray g):
    return solve(H,g)  

cdef np.ndarray update(g, np.ndarray H, np.ndarray x, np.ndarray xold, double alpha, np.ndarray s):

  cdef np.ndarray delta, gamma, p1, p2, dH, Hd

  delta = x - xold # alpha*s
  gamma = g(x) - g(xold)

  p1 = np.outer(gamma,gamma)/np.inner(gamma,delta)

  Hd = H.dot(delta)
  dH = delta.dot(H)
  p2 = np.outer(Hd, dH) / np.inner(dH, delta)

  return H + p1 - p2