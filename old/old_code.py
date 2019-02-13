import numpy as np
np.seterr(all='raise')

#from smpl/smpl_webuser_hello_world/hello_smpl.py
    '''
    if isinstance(flabbiness, Number):
      m.betas[2]          = flabbiness
    else:
      couch_potato        = m.betas[2]  # I honestly don't know how we can measure this variable.  Prob by asking the user.  But will they tell the truth?  Will they know?    (negative values for this mean a well-muscled guy)
    '''

# part of model.py's test_human():
"""
on_locs=np.nonzero(model)
model_2=deepcopy(model)
model_2[on_locs[0],on_locs[1],on_locs[2]]=MID

print "on_locs[0].shape is {0}".format(on_locs[0].shape)
  # on_locs[0].shape is (3817528,)

on_locs=np.nonzero(model)
model_2=deepcopy(model)
model_2[on_locs[0],on_locs[1],on_locs[2]]=MID
assert np.array_equal(model, model_2)
print "assertion passed!"
  value was True.  !!!
"""

"""
  desired final result
ons = on_locs_nonzero(model)
print ons.shape
model_2=np.zeros(model.shape).astype('bool'); model_2[ons]=1
"""
def find_crotch(mask):
  '''
    As of Thu Jan 17 09:15:12 EST 2019,
      This func assumes 0 noise in the mask, which is an unrealistic assumption
  '''
  locs=np.nonzero(mask)
  toe_height=np.amax(locs[0])
  both_feet_height=float('inf')
  #both_feet_height=toe_height+30 # TODO: fiddle with this until it's right
  crotch_x_loc=float('inf')
  for height in range(toe_height,0,-1):
    if both_feet_height==float('inf'): # both_feet_height will change values once we find the pt where we see both legs separately
      in_1st=False
      looking_for_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st and mask[height,x]:
          in_1st=True
        elif in_1st and not mask[height,x]:
          looking_for_2nd=True
        elif looking_for_2nd and mask[height,x]:
          both_feet_height=height
          break # out of for loop
    else: # found height where both legs are present in one row
      in_1st=False
      looking_for_2nd=False
      found_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st        and     mask[height,x]:
          in_1st=True
        elif in_1st          and not mask[height,x]:
          looking_for_2nd=True
        elif looking_for_2nd and     mask[height,x]:
          crotch_x_loc=x-2
          found_2nd=True
      if looking_for_2nd and not found_2nd:
        crotch_height=height
        return (crotch_x_loc, crotch_height)
  raise CrotchNotFoundException
  return

#===================================================================================================================================
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
  xy=(1,0); return uint_mids(
                              scipy.ndimage.rotate(model, angle, axes=xy, reshape=False, mode='constant')
                              )
#=============================================  end func def of rot8(model, angle):  ===============================================
def uint_mids(arr):
  # NOTE:  do we want this to run on floats?  TODO:   try multiple ways (first on uint8, then float, etc.)
  UINT8_MAX=np.iinfo('uint8').max; MID=int(round(UINT8_MAX/2.))
  arr[np.greater(arr, 0)]=MID; return arr
#===================================================================================================================================
def binarize(mask_3_colors):
  RED=0; CERTAIN=256.0; probable = np.array(int(CERTAIN / 2)-1) # default color is magenta, but the red part shows 
  mask_binary = deepcopy(mask_3_colors[:,:,RED])
  return np.greater(mask_binary, probable).astype('bool')
#===========================================  end func def of binarize(mask_3_colors):  ============================================
def test_mask_func(mask_fname):
  mask_2d   = np.asarray(ii.imread(mask_fname)).astype('bool')
  #pltshow(mask_2d)     # this mask was fine
  depth     = mask_2d.shape[0]
  model     = np.ones((mask_2d.shape[0], depth, mask_2d.shape[1])).astype('bool')
  if debug:
    cross_sections_biggest(model)

  model     = mask(model, mask_2d)
  if debug:
    cross_sections_biggest(model)
    show_cross_sections(model, axis='y')

  return model
#====================================   end func def test_mask_func(mask_fname):   =======================================================

