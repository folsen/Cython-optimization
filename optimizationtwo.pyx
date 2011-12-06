from __future__ import division
from scipy.linalg import *
from numpy import *

class OptimizationProblem (object):
  def __init__(self, f, grad=0):
    self.f = f
    self.g = grad

class OptimizationMethod (object):

  def __init__(self, problem, linesearch='None', verbose=False):
    self.prob = problem
    self.linesearch=linesearch
    self.verbose = verbose

  def __call__(self, x):
    # Kor loopen
    cdef int maxit = 100
    cdef double tol = 1e-8
    cdef double alpha 

    g = self.prob.g

    for i in range(0,maxit):
      if i == 0:
        # This function and the one below can be integrated into one
        H = self.initial_h(x)
      else:
        # Update - this function and the one above can be integrated into one
        H = self.update(H, x, xold, alpha, s)

      s = self.chol_solve(H, g(x))
      if self.linesearch == 'None':
        alpha = 1
      elif self.linesearch == 'Exact':
        alpha = self.exact_linesearch(x,s,0)
      elif self.linesearch == 'Inexact':
        alpha = self.exact_linesearch(x,s)
      elif self.linesearch == 'Armijo':
        alpha = self.armijo_linesearch(x,s)
      xold = x
      x = x-alpha*s

      if norm(array(x)-array(xold)) < tol:
        if self.verbose:
          print "Found answer in " + str(i) + " iterations."
        return x

    raise Exception("Didn't converge.")

  def chol_solve(self,H,g):
    try:
      L = cholesky(H)
      lp = solve_triangular(L.T, g, lower=True)
      Hi = solve_triangular(L, lp)
      return Hi
    except:
      raise Exception("G not pos. def. please pick a guess closer to the target.")

  def initial_h(self, x):
    cdef double h = 1e-8
    g = self.prob.g
    # Finite difference approximation
    Gb = array([(g(x + ei * h) - g(x))/h for ei in identity(size(x))])
    G = (1./2.) * (Gb + Gb.T)
    return G

  def exact_linesearch(self, x, s, double a_guess):
    #return fmin(self.freeze_function(self.prob.f, x, -s),0,disp=False)
    cdef double a, ll, lu, tol, tmp

    a = a_guess
    ll = 0
    lu = 1
    tol = 1e-5
    fprime = self.get_fprime_alpha(self.prob.g,x,-s)
    while not fprime(lu) > 0:
      lu = 2*lu

    while True:  
      a = (ll+lu)/2
      tmp = fprime(a)
      if tmp > 0:
        lu = a
      elif tmp < 0:
        ll = a
      else:
        break

      if abs(ll-lu) < tol:
        break
    
    return a
  
  def freeze_function(self, f, x, s):
    return lambda ap:f(x + ap*s)

  def get_fprime_alpha(self, g, x, s):
    return lambda a: dot(g(x+a*s),s)

class QuasiNewton (OptimizationMethod):

  # def initial_h(self,x):
  #   return identity(size(x))
  
  def chol_solve(self,H,g):
    return solve(H,g)  

  def update(self, H, x, xold, double alpha, s):
    print "Not implemented"
  
class BFGS (QuasiNewton):
  def update(self, H, x, xold, double alpha, s):

    g = self.prob.g
    delta = x - xold # alpha*s
    gamma = g(x) - g(xold)

    p1 = outer(gamma,gamma)/inner(gamma,delta)

    Hd = H.dot(delta)
    dH = (delta.T).dot(H)
    p2 = Hd.dot(dH) / inner(dH, delta)

    #p1 = (1 + (gamma.T).dot(H).dot(gamma)/inner(delta, gamma))*(outer(delta, delta)/inner(delta, gamma))

    #p2 = (outer(delta, gamma).dot(H) + H.dot(outer(gamma, delta)))/inner(delta, gamma)
    return H + p1 - p2