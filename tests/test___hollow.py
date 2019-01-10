from hollow import *
from viz import *

if __name__=="__main__":
  cube=np.zeros((8,8,8)).astype('bool')
  cube[1:7,1:7,1:7]=True
  cube[0,1,1]=True
  shell=hollow(cube)
  show_all_cross_sections(shell,freq=1)
