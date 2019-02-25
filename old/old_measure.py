import numpy as np
import imageio as ii
import glob
import os
import sys
import json

import viz
from viz import pltshow
from utils import pif
from d import debug
from save import save 


  """
    Glossary as of Mon Feb 25 09:07:23 EST 2019:
      def find_toes(mask___face_on):
      def find_crotch(mask___portrait_view):
      def get_waist(mask,customers_height):
      def pixel_height(mask):
      def measure_leg(crotch,toes):
      def leg_len(mask,customers_height):
  """
  """
    Point is, we have height and weight (from customer) and inseam
      TODO:
        Chest
        Waist
        Hips

        exercise???  (if we get this at all, it should just be by askign.  Althgouh people will not necessarily know the answer off the top of their heads, they will prooooooooooooobably be able to estimate REASONabblllly well.
  """









# TODO: extend these functions to get ALL measurements of the person (ie. wingspan, leg length, waist)
#     TODO: using iPhone measure() app in conjunction with the code in this module, automate height-retrieval in inches and size customers accordingly



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
  #     minimum distance from bottom right corner (one toe)
  # and minimum distance from bottom left corner  (the other toe)
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
  # Given a photo with the customer's legs spread, trace the inside of the left leg up until you find the crotch.
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
  # Given a photo with the customer's legs spread, trace the inside of the left leg up until you find the crotch.
  mask=mask___portrait_view
  locs=np.nonzero(mask)
  toe_height=np.amax(locs[0])
  both_feet_height=float('inf')
  # return "crotch" as dict at the end
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
  # NOTE:  should be at (280,150).  It was.  Jan 29, 2019.
#==============================================================
def get_waist(mask,customers_height):
  '''
    This function assumes there are no arms down at the waist height.  ie. Jesus pose, mTailor pose, one of those.
    Waist, not inseam, for men's jeans

    customers_height is in inches
  '''
  pix_height=pixel_height(mask)
  crotch=find_crotch(mask)
  crotch_height=crotch['height']
  crotch_x=crotch['x_loc']
  crotch=np.array([crotch['height'], crotch['x_loc']]).astype('int64')
  pix_btwn_waist_and_crotch=24 # TODO: fiddle with.
  waist_height=crotch_height-pix_btwn_waist_and_crotch
  waist_in_pixels=int(np.count_nonzero(mask[waist_height])) # NOTE: can modify this to get only the middle section (strip)'s length if both arms go down and are just as high as the waist
  print("waist_in_pixels is {0}".format(waist_in_pixels))
  print("crotch is {0}".format(crotch)) # fine
  print("pix_height is {0}".format(pix_height))
  print("customers_height is {0}".format(customers_height))
  pltshow(mask)
  pltshow(mask[waist_height-10:waist_height+10,:])
  if debug:
    pltshow(mask[crotch_height-10:crotch_height+10, crotch_x-10:crotch_x+10])
    pltshow(mask[waist_height-5:waist_height+5,:])
  return waist_in_pixels/pix_height*customers_height  # NOTE NOTE NOTE NOTE NOTE not right!
  # TODO: get the belly-waist from the side view and approximate the waist measurement as an ellipse.
  #       but how do we locate the waist given the side view?  Openpose could do it, but that's another dependency.  We could take the midpoint of head and toe, but again, pretty fragile.
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
  '''
    customers_height is in inches
  '''
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
  folder='/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/'
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



  mask_fname  = sys.argv[1]   #"/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg"
  # 32, 38, 39, and 43 find the crotch to be higher  than the "nearby" images
  mask        = np.asarray(ii.imread(mask_fname))
  mask=np.greater(mask,127)
  NATHAN_HEIGHT=75 # inches
  print("inseam estimation (length in inches):   {0}".format(leg_len(mask,NATHAN_HEIGHT)))
  # NOTE:  there are no real units here;  it's all just a ratio that is normalized to Nathan's height and pants length
  #"""
#==============================================================
  # for Nathan segmentation (/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg),
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
  #  NOTE: this doesn't use openpose; will probably be more dependent on the customer posing a certain way, dependent on the segmentation coming out right, etc.
  #    slightly less robust to 
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
 










































































