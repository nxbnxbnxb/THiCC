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

print('perim: {0}'.format(perim(20,20)))
print('perim: {0}'.format(perim(20,10)))
print('perim: {0}'.format(perim(20, 6)))
print('perim: {0}'.format(perim(20, 2)))
