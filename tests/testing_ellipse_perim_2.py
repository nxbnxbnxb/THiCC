

from sympy import *
import numpy as np

a, b, w = symbols('a b w')

x = a/2 * cos(w)
y = b/2 * sin(w)

dx = diff(x, w)
dy = diff(y, w)

ds = sqrt(dx**2 + dy**2)

def perim(majr, minr):
    return Integral(ds.subs([(a,majr),(b,minr)]), (w, 0, 2*pi)).evalf().doit()

"""
print('test1: a, b = 1 gives dia = 1 circle,  perim/pi = ',
      perim(1, 1)/pi.evalf())
"""

ellipses=np.array([ [10., 10.],
                    [10.,  5.],
                    [10.,  3.],
                    [10.,  1.]]).astype('float64')
for a,b in ellipses:
  print("Major: {0}  \nMinor: {1}  \nPerim: {2}".format(a,b,perim(a,b)))

