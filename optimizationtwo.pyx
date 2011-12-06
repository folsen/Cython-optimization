from __future__ import division
from scipy.linalg import (norm, solve, cholesky, solve_triangular)
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double abs(double)

DTYPE = np.int
ctypedef np.int_t DTYPE_T

class OptimizationProblem (object):
  def __init__(self, f, grad=0):
    self.f = f
    self.g = grad

class OptimizationMethod (object):

  def __init__(self, problem, linesearch='None', verbose=False):
    self.prob = problem
    self.linesearch=linesearch
    self.verbose = verbose

  def __call__(self, np.ndarray x):
    # Kor loopen
    cdef int maxit = 100
    cdef double tol = 1e-8
    cdef double alpha 
    cdef char* ls = self.linesearch
    cdef char* none = "None"
    cdef char* exact = "Exact"
    g = self.prob.g

    cdef np.ndarray H, xold, s

    for i in range(0,maxit):
      if i == 0:
        # This function and the one below can be integrated into one
        H = self.initial_h(x)
      else:
        # Update - this function and the one above can be integrated into one
        H = update(g, H, x, xold, alpha, s)

      s = chol_solve(H, g(x))
      if ls == none:
        alpha = 1
      elif ls == exact:
        alpha = self.exact_linesearch(x,s,0)

      xold = x
      x = x-alpha*s

      if norm(np.array(x)-np.array(xold)) < tol:
        return x

    raise Exception("Didn't converge.")

  def initial_h(self, np.ndarray x):
    cdef double h = 1e-8
    g = self.prob.g
    # Finite difference approximation
    Gb = np.array([(g(x + ei * h) - g(x))/h for ei in np.identity(np.size(x))])
    G = (1./2.) * (Gb + Gb.T)
    return G

  def exact_linesearch(self, np.ndarray x, np.ndarray s, double a_guess):
    #return fmin(self.freeze_function(self.prob.f, x, -s),0,disp=False)
    cdef double a, ll, lu, tol, tmp

    a = a_guess
    ll = 0
    lu = 1
    tol = 1e-5
    #fprime = self.get_fprime_alpha(self.prob.g,x,-s)
    while not cfprime(self.prob.g,x,-s,lu) > 0:
      lu = 2*lu

    while True:  
      a = (ll+lu)/2
      tmp = cfprime(self.prob.g,x,-s,a)
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
  dH = (delta.T).dot(H)
  p2 = Hd.dot(dH) / np.inner(dH, delta)

  #p1 = (1 + (gamma.T).dot(H).dot(gamma)/inner(delta, gamma))*(outer(delta, delta)/inner(delta, gamma))

  #p2 = (outer(delta, gamma).dot(H) + H.dot(outer(gamma, delta)))/inner(delta, gamma)
  return H + p1 - p2