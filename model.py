import  numpy as np
from    copy import deepcopy
import imageio as ii
import scipy; from scipy import ndimage, misc

from viz import *
from d import debug 
from utils import pad_all

# TODO: integrate seg.py into the beginning of this
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
    returns a new 3-D np array
      (the result of masks the 3-d np array "model" all the way through using the mask)
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
  #       do vstack or array_copy or just use clever broadcasting
  model_depth=model.shape[1]
  pif('model_depth is {0}'.format(model_depth))
  for i in range(model_depth):
    model_copy[:,i,:] = np.logical_and(model[:,i,:], mask)
  print 
  return model_copy
#====================================  end func def mask(model, mask, axis='x'):  =======================================================
def rot8(model, angle):
  '''
    rotates counterclockwise (by default around the positive z-axis; if you were looking down at the model from the head, it would rotate counterclockwise)
      TODO:  make sure the direction is right
      TODO:  extend to rotations around other axes

    output: 
      rotated 3-D numpy array
    parameters:
      model is a 3-D numpy array
      angle in degrees (float: floating point)
    Notes:
      mode : not 100% sure what this parameter does, but here are:
        Notes from scipy.ndimage.rotate():
          The given matrix and offset are used to find for each point in the
          output the corresponding coordinates in the input by an affine
          transformation. The value of the input at those coordinates is
          determined by spline interpolation of the requested order. Points
          outside the boundaries of the input are filled according to the given mode.

          According to these notes, possible values of mode are:
            'constant'
            'nearest'
            'reflect'
            'wrap'

      reshape=False will cut off the corners when we rotate.  So we NEED to make sure the human body is centered within the numpy array before proceeding.  I used this so the dimensions don't get fucked up later
  '''
  z_up=(1,0); return scipy.ndimage.rotate(model, angle, axes=z_up, reshape=False, mode='nearest')
#===================================================================================================================================
def test_human():
  # mask 1
  import seg; mask_2d = seg.main('http://columbia.edu/~nxb2101/180.0.png'); max_dim   = max(mask_2d.shape); shape=(max_dim, max_dim, max_dim); shape_2d=(max_dim, max_dim)
  #mask_1___fname = "/home/u/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks___human_front_and_side/180.0.jpg"
  mask_2d   = pad_all(mask_2d, shape_2d)
  model     = np.ones(shape).astype('bool')
  model     = mask(model, mask_2d)
  print "before rot8();   \n\n"
  show_cross_sections(model, axis='y', show_every=250) # NOTE: good.  it worked this time
  model     = rot8(model, 90.0)
  print "right after rot8();   \n\n"
  show_all_cross_sections(model, how_often=20)

  # mask 2
  #mask_2___fname = "/home/u/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks___human_front_and_side/90.0.png"
  mask_2d = seg.main('http://columbia.edu/~nxb2101/90.0.png'); mask_2d   = pad_all(mask_2d, shape_2d)
  model     = mask(model, mask_2d)
  print "after 2nd masking:    \n\n"
  show_all_cross_sections(model, how_often=20)
  if debug:
    show_all_cross_sections(model)
  return model
#====================================   end func def test_human():   =======================================================

#===================================================================================================================================
if __name__=='__main__':
  test_human()
#===================================================================================================================================







































