from optimizationtwo import *
from numpy import *
from scipy import *

execfile('chebyquad_problem_.py')

x=linspace(0,1,5)

def runtest():
	bfgs = BFGS(chebyquad,gradchebyquad,linesearch='Exact')
	return bfgs(x)

if __name__ == '__main__':
	from timeit import Timer
	t = Timer("runtest()", "from __main__ import runtest")
	times = map(lambda x: x/100, t.repeat(repeat=5, number=100))
	print "Mean runtime in seconds:"
	print times
