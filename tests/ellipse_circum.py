import sympy as s
import math
import sys

#=======================================================================================================================================
def ellipse_circum(a, b):
  # https://en.wikipedia.org/wiki/Ellipse
  # Major and Minor axes:
  #   (Major = a and Minor = b)
  # TODO: finish this function and integrate it into chest_circum().

  # https://stackoverflow.com/questions/22560342/calculate-an-integral-in-python
  e=math.sqrt(1-(b**2)/(a**2))
  theta=s.symbols('theta')
  return 4*a*\
  s.integrate(
    s.sqrt(
      1-\
      e**2*s.sin(theta)**2),
    (theta, 0, math.pi/2.))
  # rename oval_perim()?  slightly shorter and less intimidating
#================================================= ellipse_cirum() =====================================================================
if __name__=="__main__":
  print(ellipse_circum(float(sys.argv[1]), float(sys.argv[2])))
