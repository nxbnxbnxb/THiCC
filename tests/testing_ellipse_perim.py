from math import pi


def perim(a,b,precision=4):
  # ellipse
  if b > a:
    tmp=a; a=b; b=tmp # swap such that a is always the semi-major axis (bigger than b)
  h=((a-b)**2)/((a+b)**2)
  return pi*(a+b)*\
    (1+\
      (1/4)*h+\
      (1/64)*h**2+\
      (1/256)*h**3)



if __name__=="__main__":
  print(perim(10,10))






























































