# TODO: kill code fragments I was using to unit test.
import numpy as np
from math import pi
import sys
from utils import pn

def pn(n):
  print('\n'*n)
#=======================================================================================================================================
def perim_e(a, b, precision=6):
  """Get perimeter (circumference) of ellipse

  Parameters
  ----------
  %(input)s
  a : scalar
    Length of one of the SEMImajor axes (like radius, not diameter)
  b : scalar
    Length of the other semimajor axis
  precision : int
    How close to the actual circumference we want to get.  Note: this function will always underestimate the ellipse's circumference.

  Returns
  -------
  float
    Perimeter
  gaussian_filter : ndarray
      Returned array of same shape as `input`.

  Notes
  -----
  As far as I can tell from ~5 hours working on this, the precise circumference of an ellipse is an open problem in mathematics.  If you see the integral on Wikipedia and have some idea how to solve it, please let the mathematics community (and me: [nathanbendich@gmail.com]) know!
  The real ellipse's circumference will always be more than what this infinite series predicts because the infinite series is a sum of a positive sequence.

  According to https://www.mathsisfun.com/geometry/ellipse-perimeter.html, Ramanujan's approximation that perimeter ~= pi*(a+b)*(1+(3h/(10+sqrt(4-3*h)))) seems to be higher than this series approximation.
  I think Sympy uses Approx. 3 from that ellipse page (https://www.mathsisfun.com/geometry/ellipse-perimeter.html).  I only tested for a = 10 and values of b on the page https://www.mathsisfun.com/geometry/ellipse-perimeter.html (I really hope this link doesn't end up breaking)
    But sympy is slowwww.  Maybe for refactoring, use Ramanujan's approx 3 but turn it into pure python or numpy or something fast?  Or take the mean of this series and Ramanujan's approx. 3?  There's no analytical reason for that; I'm just basing it off the fact that this series undershoots and Ramanujan's approximation overshoots.
  Another potential improvement is to actually take the time to understand the "Binomial Coefficient with half-integer factorials" mentioned on https://www.mathsisfun.com/geometry/ellipse-perimeter.html.  I attempted this for awhile, but got caught on some bug I didn't understand, opting for hard-coding 6 "levels of precision" instead.  So long as there are no overflows, extending this to arbitrary precision should be doable; it just takes a little more mathematical rigor and care than I'm giving right now.  Anyway, I've spent too long writing this documentation, but it was very personally satisfying to take the time to do this right.  I really need to be practical about churning out code faster though, haha.  While just prototyping, this level of detail might not be practical.

  Examples
  --------
  """
  # This function uses approximations mentioned in these sources:
  #   1.(https://www.mathsisfun.com/geometry/ellipse-perimeter.html) and 
  #   2. https://en.wikipedia.org/wiki/Ellipse
  #   3. https://stackoverflow.com/questions/42310956/how-to-calculate-perimeter-of-ellipse
  #   4. 
  # For more reading while refactoring, please consult the wikipedia page, https://math.stackexchange.com/, or wherever else.
  # func perim_e():
  funcname=  sys._getframe().f_code.co_name
  if b > a:
    tmp=a; a=b; b=tmp # swap such that a is always the semi-major axis (bigger than b)
  if precision <= 0:
    pn(2); print("In function ",funcname); print("WARNING:  precision cannot be that low"); pn(2)
  if precision >  6:
    # precision higher than 6 not supported as of Tue Feb 26 12:06:35 EST 2019
    pn(2); print("In function ",funcname); print("WARNING:  precision that high is not yet supported"); pn(2)
  # To understand what each symbol (h, seq, a, b) means, please see our sponsor at 
  #   https://www.mathsisfun.com/geometry/ellipse-perimeter.html.
  h=((a-b)**2)/((a+b)**2)
  seq=[  1/          h**0,
         1/        4*h**1,
         1/       64*h**2,
         1/      256*h**3,
        25/    16384*h**4, 
        49/    65536*h**5, 
       441/  1048576*h**6]  # only up to 7 terms
  perim=pi*(a+b)*sum(seq[:precision])
  return perim
#============================================= perim_e() ===============================================================================

if __name__=="__main__":

  ellipses=np.array([ [10, 10],
                      [10,  5],
                      [10,  3],
                      [10,  1]]).astype('float64')
  for a,b in ellipses:
    print("a = {0}, b = {1} ".format(a,b))
    #print("Major: {0}  \nMinor: {1}  \nPerim: {2}".format(a,b,perim(a,b)))
    for i in range(1,7):
      print('  i is {0} and perim is {1}'.format(i,perim_e(a,b,precision=i)))
    #print(       "  2*pi              = ",2*pi)
