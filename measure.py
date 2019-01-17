import numpy as np
import imageio as ii
import glob
import os

import viz
from utils import pif
from d import debug
from save import save 

def find_crotch(mask___face_on):
  '''
    As of Thu Jan 17 09:15:12 EST 2019,
      This func assumes 0 noise in the mask, which is an unrealistic assumption
      Also assumes we never get any "hooks" in the left leg, ie. 
                                                              /
                                                             /                               \   <--- right leg
                                                             ^                                \ 
                                                            / \ <---- this is the hook I mean  \ 
                                                           /
                                                          /
  '''
  mask=mask___face_on
  locs=np.nonzero(mask)
  toe_height=np.amax(locs[0])
  both_feet_height=float('inf')
  crotch={};crotch['x_loc']=float("inf");crotch['height']=float("inf")
  left_leg_inner_x=0 # assume the person is facing away from us;   then their left is our left
  for height in range(toe_height,0,-1):
    if both_feet_height!=float('inf'):
      break  # we've found the starting point from the left leg
    else: # both_feet_height will change values once we find the pt where we see both legs separately
      in_1st=False
      looking_for_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st and mask[height,x]:
          in_1st=True
        elif in_1st and not mask[height,x] and not looking_for_2nd:
          left_leg_inner_x=x
          pif("within for loops, left_leg_inner_x is {0}".format(left_leg_inner_x))
          looking_for_2nd=True
        elif looking_for_2nd and mask[height,x]:
          both_feet_height=height
          break # out of inner (for x in range...) for loop, not the "for height in range(...)" loop
  pif("left_leg_inner_x is {0}".format(left_leg_inner_x))
  low=max(np.nonzero(mask[:,left_leg_inner_x])[0])
  should_be_lower=max(np.nonzero(mask[:,left_leg_inner_x+1])[0])
  pif(low)
  pif(should_be_lower)
  while float('inf') in crotch.values():
    # NOTE: could do an infinite loop
    low=max(np.nonzero(mask[:,left_leg_inner_x])[0])
    should_be_lower=max(np.nonzero(mask[:,left_leg_inner_x+1])[0])
    if should_be_lower > low:
      crotch['x_loc']=left_leg_inner_x
      crotch['height']=low
      # will cause the break
    else:
      left_leg_inner_x+=1
  return crotch
#==============================================================
if __name__=="__main__":
  # TODO:  esp. in the future, be wary of how important it is to use np.greater()  (segmentation doesn't just return simple "true-false")
  # whole directory of masks:
  folder='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/'
  mask_fnames= sorted(glob.glob(folder+'*'),key=os.path.getmtime)
  print(mask_fnames)
  fronts_ctr=0
  for mask_fname in mask_fnames:
    if mask_fname.endswith("jpg") or mask_fname.endswith("png") or mask_fname.endswith("jpeg"):
      print(find_crotch(np.greater(np.asarray(ii.imread(mask_fname)),127)))
    if fronts_ctr == 60: # 60 is experimentally derived from this particular set of masks
      break
    fronts_ctr+=1




  # single image:
  """
  mask_fname  = "/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg"
  mask        = np.asarray(ii.imread(mask_fname))
  unique, counts = np.unique(mask, return_counts=True)

  viz.pltshow(np.greater(mask,128))
  mask=np.greater(mask,127)
  """
  












































































