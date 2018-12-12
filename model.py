import  numpy as np
from    copy import deepcopy
import imageio as ii
from visualization import *
from d import debug 

# TODO:  scipy.ndimage.rotate() and fill in the 1 or 2-gaps
#   NOTE:  z is "up" ("up" as in, think about a human being "upright"; the head is "up")

#===================================================================================================================================
def pif(s):
    if debug: print s
#===================================================================================================================================
class BadInputException(RuntimeError):
    pass
#===================================================================================================================================
class DimensionMismatchException(RuntimeError):
    pass
#===================================================================================================================================
def mask(model, mask, axis='x'):
  '''
    masks the 3-d np array "model" all the way through using the mask
  '''
  model_copy = deepcopy(model)
  if not type(mask ) == type(np.ones((2,2))):
    raise BadInputException("the mask parameter was supposed to be a numpy array")
  if not type(model) == type(np.ones((2,2))):
    raise BadInputException("the model parameter was supposed to be a (3-D) numpy array")
  if not len(model.shape)==3:
    raise BadInputException("the model np array parameter was supposed to have 3 dimensions")
  if not len(mask.shape )==2:
    raise BadInputException("the mask  np array parameter was supposed to have 2 dimensions")
  models_height = model.shape[2]; masks_height=mask.shape[1]; models_width  = model.shape[0]; masks_width =mask.shape[0]
  if not models_width == masks_width and models_height==masks_height:
    raise DimensionMismatchException(" the models width and masks width have to be equal.   So do the heights")

  '''
  mask_3d = 
  '''
  # NOTE: actual code starts HERE:
  # TODO:  vectorize this instead of using a for loop
  model_depth=model.shape[1]
  pif('model_depth is {0}'.format(model_depth))
  for i in range(model_depth):
    model_copy[:,i,:] = np.logical_and(model[:,i,:], mask)
  print 
  return model_copy
#====================================  end func def mask(model, mask, axis='x'):  =======================================================


#===================================================================================================================================
def test_mask_func(mask_fname):
  mask_2d   = np.asarray(ii.imread(mask_fname)).astype('bool')
  #pltshow(mask_2d)     # this mask was fine
  depth     = mask_2d.shape[0]
  model     = np.ones((mask_2d.shape[0], depth, mask_2d.shape[0])).astype('bool')
  if debug:
    cross_sections_biggest(model)

  model     = mask(model, mask_2d)
  if debug:
    cross_sections_biggest(model)
    show_cross_sections(model, axis='y')

  return model
#====================================   end func def test_mask_func(mask_fname):   =======================================================

#===================================================================================================================================
if __name__=='__main__':
  test_mask_func('0.0.png')
#===================================================================================================================================
