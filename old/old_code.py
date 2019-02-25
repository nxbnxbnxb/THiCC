import numpy as np
np.seterr(all='raise')

#from smpl/smpl_webuser_hello_world/hello_smpl.py
    '''
    if isinstance(flabbiness, Number):
      m.betas[2]          = flabbiness
    else:
      couch_potato        = m.betas[2]  # I honestly don't know how we can measure this variable.  Prob by asking the user.  But will they tell the truth?  Will they know?    (negative values for this mean a well-muscled guy)
    '''

#seg.py's  overlay_imgs()   (oughta be called  "superimpose_imgs()")
    #ii.imwrite(tmpfname,cutout2) # write specialized funcs that convert to/from PIL/np.
    #cutout2=np_img(tmpfname)

    #ii.imwrite(tmpfname,cutout1)
    #cutout1=np_img(tmpfname)
    #sp.call(['rm',tmpfname])
    #ii.imwrite(tmpfname,cutout2)
    #cutout2=np_img(tmpfname)
    #sp.call(['rm',tmpfname])
    #pltshow(cutout2)

    #sp.call(['rm',tmpfname])
    # TODO: finish!
    pass
    '''
    # `pltshow(cutout1[x_min_1:x_max_1,y_min_1:y_max_1])` is my version of "crop()" function for numpy representations of images
    # this ugly assignment is much shorter than the alternative (all fits on one line)
    edge_x_1, edge_y_1  = min(locs1[0]), min(locs1[1])  
    edge_x_2, edge_y_2  = min(locs2[0]), min(locs2[1])
    print("edge_x_1: \n",edge_x_1)
    print("edge_x_2: \n",edge_x_2)
    print("edge_y_1: \n",edge_y_1)
    print("edge_y_2: \n",edge_y_2)
    mask1_shape = (max(locs1[0])-min(locs1[0]),  max(locs1[1])-min(locs1[0]))
    mask2_shape = (max(locs2[0])-min(locs2[0]),  max(locs2[1])-min(locs2[0]))
    # "0" b/c no color shift
    pltshow(shift(cutout1, (-edge_x_1, -edge_y_1, 0)))
    pltshow(shift(cutout2, (-edge_x_2, -edge_y_2, 0)))
    # I'm doing this imwrite() b/c I don't know how to resize an image in np.ndarray();  I only know how to with PIL.Image.resize((x,y))
    tmpfname='tmp.png'
    ii.imwrite(tmpfname,cutout1)
    cutout1=Image.open(tmpfname)
    cutout1=cutout1.resize((mask2_shape), Image.ANTIALIAS)
    pltshow(cutout1)
    # TODO: scale both images s.t. they overlay
    # TODO: crop imgs b4 overlaying
    # TODO: finish this overlay_imgs() function!

    cutout2=np.array(cutout2)
    pltshow(cutout2+cutout1)  # looks funky.
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

#===================================================================================================================================
# This code is a half-formed thought.  Probably will never use it again.
def find_nip(mask):
  find_nip_by_tracing()
def find_nip_by_tracing():
  pass
def armpit(mask):
  '''
    Finds the customer's armpit given a segmented-out mask of their body
    Customer is in "Jesus pose" as of Mon Feb 25 13:34:31 EST 2019
  '''
  return armpit_by_tracing(mask)
def armpit_by_tracing(mask):
  # Finds the armpit by locating the extended arms of the customer and then going "down" the mask the proper corresponding amount.
  # "Jesus Pose" required
  # TODO: update the comment above this line after I've actually written the function.
  counts=np.count_nonzero(mask, axis=1)
  fingertips_y_idx=np.argmax(counts)
  CONST=10
  pltshow(mask)
  pltshow(mask[fingertips_y_idx-CONST:fingertips_y_idx+CONST])
  pltshow(mask)
  # TODO: paste this code into the function "chest_circum(json_fname, front_fname, side_fname)"
#===================================================================================================================================

