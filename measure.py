import numpy as np
import imageio as ii
import glob
import os
import sys

import viz
from utils import pif
from d import debug
from save import save 










# TODO: extend these functions to get ALL measurements of the person (ie. wingspan, leg length, waist)




#==============================================================
def find_toes(mask___face_on):
  '''
    Doesn't do this perfectly; just an estimation (Thu Jan 17 11:22:14 EST 2019)

    Ideas to improve it if the result isn't precise enough:
      Add dist_from_crotch() as a negative weight 
        ie. 
          1. Try to find the toe that is both closest to the left corner and farthest from the crotch, and
          2. Find        the toe that is closest to the right corner and farthest from the crotch)
  '''
  mask        = mask___face_on
  bot_left    = np.array([mask.shape[0]-1,0               ])
  bot_right   = np.array([mask.shape[0]-1,mask.shape[1]-1 ])
  locs        = np.array(np.nonzero(mask)).T
  distances   = np.sum(np.sqrt(np.abs(locs-bot_left )),axis=1)
  lefts_idx   = np.argmin(distances)
  distances   = np.sum(np.sqrt(np.abs(locs-bot_right)),axis=1)
  rights_idx  = np.argmin(distances)
  return {'left_toe':locs[lefts_idx].reshape(1,2), 'right_toe':locs[rights_idx].reshape(1,2)}

#==============================================================
def find_crotch(mask___portrait_view):
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
  mask=mask___portrait_view
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
          looking_for_2nd=True
        elif looking_for_2nd and mask[height,x]:
          both_feet_height=height
          break # out of inner (for x in range...) for loop, not the "for height in range(...)" loop
  while float('inf') in crotch.values():
    # NOTE: could do an infinite loop on the wrong input
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
def pixel_height(mask):
  locs=np.nonzero(mask)
  return np.amax(locs[0])-np.amin(locs[0])
#==============================================================
#==============================================================
def measure_leg(crotch,toes):
  TWO_LEGS=2
  return np.sum(np.sqrt(np.sum(np.square(crotch-toes),axis=1)))/TWO_LEGS
#==============================================================
#==============================================================
def leg_len(mask,customers_height):
  pix_height=pixel_height(mask)
  #print("pix_height:\n{0}".format(pix_height))  # not the problem

  crotch=find_crotch(mask)
  crotch=np.array([crotch['height'], crotch['x_loc']]).astype('int64')
  toes  =find_toes  (mask)
  toes  =np.concatenate((toes['left_toe'],toes['right_toe']),axis=0)
  '''
  #print("toes:\n{0}".format(toes))
  #print("crotch:\n{0}".format(crotch))  # NOTE: toes and crotch were fine.  the issue was somewhere in between these lines and the end

    pix_height:
    330

    toes:
    [[423 120]
     [409 168]]

     crotch:
     [288 137]

     leg_len_pixels:
     16.154910013534966
  '''
  pif("crotch-toes:\n{0}".format(crotch-toes))
  leg_len_pixels=measure_leg(crotch,toes)
  pif("leg_len_pixels:\n{0}".format(leg_len_pixels))
  pif("\n"*3)
  return leg_len_pixels/pix_height*customers_height
#==============================================================
#==============================================================
if __name__=="__main__":
  # NOTE:  esp. in the future, be wary of how important it is to use np.greater()  (segmentation doesn't just return simple "true-false")
  # whole directory of masks:
  """
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



  mask_fname  = sys.argv[1]   #"/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg"
  # 32, 38, 39, and 43 find the crotch to be higher  than the "nearby" images
  mask        = np.asarray(ii.imread(mask_fname))
  mask=np.greater(mask,127)
  NATHAN_HEIGHT=75 # inches
  print("inseam estimation (length in inches):   {0}".format(leg_len(mask,NATHAN_HEIGHT)))
  # NOTE:  there are no real units here;  it's all just a ratio that is normalized to Nathan's height and pants length
  #"""
#==============================================================
  # for Nathan segmentation (/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg),
  #
  #                       29.65615073829731     **** short
  #
  #
  # for file "./masks/2019_01_17____17\:57_PM_______inseam_36___male__/000000000.jpg":  leg_len() was 30.77904443651943
  #
  #   See prefix above.  It ain't just 000000000.jpg
  #
  # 000000000.jpg         30.77904443651943
  # 000000001.jpg         30.384982686788234    **** short
  # 000000002.jpg         30.428061480054698
  # 000000003.jpg          7.355339015509475      crossed legs
  # 000000004.jpg         32.046654629942765    **** big
  # 000000005.jpg         31.32609217992018
  # 000000006.jpg         31.86657296827472
  # 000000007.jpg         30.800175451288585
  # 000000008.jpg         34.446067858908876    **** big
  # 000000009.jpg          5.712278750981283      crossed legs
  # 000000010.jpg         32.53770678534222
  # 000000011.jpg          7.482261612732713      crossed legs
  # 000000012.jpg          7.630327382130935      crossed legs
  # 000000013.jpg         31.989151381902385
  # 000000014.jpg          6.805627029964921      crossed legs
  # 000000015.jpg         32.964500571537876
  # 000000016.jpg         28.52023566021052     **** short because the guy's turning to the side so his inseam looks lower than it actually is
  # 000000017.jpg         29.261865975820115
  # 000000018.jpg         33.05316375925231
  # 000000019.jpg          5.640038864533751      crossed legs
  # 000000020.jpg         34.87838629196392
  # 000000021.jpg          5.653397472957198      crossed legs
  # 000000022.jpg          6.675191520628813      crossed legs
  # 000000023.jpg         34.87838629196392
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  












































































