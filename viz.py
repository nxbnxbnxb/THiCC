import matplotlib.pyplot as plt
import numpy as np

#######################################################################################################
################################# visualization functions #############################################
#######################################################################################################
#=========================================================================
def pltshow(x):
  plt.imshow(x); plt.show(); plt.close()
#=========================================================================
def cross_sections_biggest(m):
  '''
      @precondition m is a cube (a 3-d np.array)
  '''
  # NOTE:  
  # NOTE:  this isn't working right.  not sure why.  Dec. 12, 2018
  # NOTE:  
  i_3=i_2=i_1=0
  max_vol_3=max_vol_2=max_vol_1=0
  for i in range(m.shape[0]):
      vol_1=np.count_nonzero(m[i,:,:])
      if vol_1 > max_vol_1: max_vol_1 = vol_1; i_1=i
      vol_2=np.count_nonzero(m[:,i,:])
      if vol_2 > max_vol_2: max_vol_2 = vol_2; i_2=i
      vol_3=np.count_nonzero(m[:,:,i])
      if vol_3 > max_vol_3: max_vol_3 = vol_3; i_3=i
  pltshow(m[i_1,:,:])
  pltshow(m[:,i_2,:])
  pltshow(m[:,:,i_3]); return
#=========================================================================
def show_cross_sections(model_3d, axis='z', show_every=1):
  if axis.lower()=='z':
      for i in range(model_3d.shape[2]):
          if i%show_every==0:
              if np.any(model_3d[:,:,i]):
                  pltshow(model_3d[:,:,i])
                  print 'height is {0}   out of {1}'.format(i, model_3d.shape[2])
  elif axis.lower()=='y':
      for i in range(model_3d.shape[1]):
          if i%show_every==0:
              if np.any(model_3d[:,i,:]):
                  pltshow(model_3d[:,i,:])
                  print 'loc is {0}   out of {1}'.format(i, model_3d.shape[1])
  elif axis.lower()=='x':
      for i in range(model_3d.shape[0]):
          if i%show_every==0:
              if np.any(model_3d[i,:,:]):
                  pltshow(model_3d[i,:,:])
                  print 'loc is {0}   out of {1}'.format(i, model_3d.shape[0])
  else:
      print "Usage: please input axis x, y, or z in format:\n\n  show_cross_sections([model_name], axis='z')"
  return
#=========================================================================
def show_all_cross_sections(model_3d, how_often=1):
  print '\n'*3; print "x: \n\n"
  show_cross_sections(model_3d, axis='x', show_every=how_often); print '\n'*3; print "y: \n\n"
  show_cross_sections(model_3d, axis='y', show_every=how_often); print '\n'*3; print "z: \n\n"
  show_cross_sections(model_3d, axis='z', show_every=how_often); return
#=========================================================================














































































