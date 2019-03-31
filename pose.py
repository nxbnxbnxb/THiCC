from measure import measure
from seg import segment_black_background
from viz import pltshow
from utils import pe,np_img

import numpy as np
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
import skimage as skimg

#=================================================================================================================================
def openpose_2_deepercut(openpose_fname, img_fname):
  no_background, mask=segment_black_background(img_fname)

  img=np_img(img_fname)
  OP_kps=measure(openpose_fname)
  #pltshow(no_background)
  mask=mask.astype('uint8')
  mask[np.nonzero(mask)]=255
  #pltshow(mask)

  pltshow(no_background)
  mask=skimg.transform.resize( mask, img.shape, anti_aliasing=True )
  marked=mark_img(img, OP_kps, mask)
  pltshow(marked)
  return DC_json
#================================================= end func openpose_2_deepercut() ===============================================
def mark_img(img, OP_kps, mask):
  '''
    Marks img w/ pertinent openpose keypoints OP_kps   and also calculates top of the head.
  '''
  neck=OP_kps['Neck']
  nose=OP_kps['Nose']
  neck=np.array([neck['x'],neck['y']]).astype('float64')
  nose=np.array([nose['x'],nose['y']]).astype('float64')
  print(neck)
  print(nose)
  plt.imshow(img)
  plt.scatter(neck[0], neck[1])
  plt.scatter(nose[0], nose[1])
  plt.show();plt.close()
  top_of_head=top_head(neck, nose, mask)
  print("top_of_head: ",top_of_head)

  plt.imshow(img)
  plt.scatter(top_of_head[0], top_of_head[1])
  # make big enough 2 see
  plt.show()
  plt.close()
  #top_of_head=dir_vert intersect mask edge
  #DC_json=OP_2_DC_dict(OP_kps)
#===================================================================================================================================

#===================================================================================================================================
def top_head(neck, nose, mask):
  '''
    calculates top of head within mask's coordinates (plt coords)
    x+ from left to right,
    y+ from top to bottom
  '''
  RGB=3
  if len(mask.shape)==RGB: # RGB
    mask01=np.logical_and(mask[:,:,0],mask[:,:,1])
    mask=np.logical_and(mask01,mask[:,:,0])
    mask=mask.reshape(mask.shape[0],mask.shape[1])

  funcname=sys._getframe().f_code.co_name
  dir_vect=nose-neck
  # unit vector
  magnitude=np.sqrt(np.sum(np.square(dir_vect)))
  dir_vect/=magnitude
  # unit in "y"
  X=0;Y=1
  dir_vect/=dir_vect[Y]
  x=nose[X];del_x=dir_vect[X]
  nose_y=int(nose[int(round(Y))])
  # for each y step "up," change delta_x.  Then when we're off the head, return the prev "in-the-mask" pt.
  mask2=deepcopy(mask)
  mask2[nose_y-3:nose_y+3,int(round(x))-3:int(round(x))+3]=0
  print("neck:")
  print(neck)
  print("mask2:")
  pltshow(mask2)
  for y in range(nose_y,0,-1): # "up" is -y
    if not mask[y,int(round(x))]: #mask[y,x] because  x and y are "reversed" in numpy
      pltshow(mask)
      mask[y-2:y+2,:]=0
      pltshow(mask)
      return (x, y+1)
    else:
      x+=del_x 
  raise Exception("Top of head not found in function named '{0}.'  Please doublecheck your code".format(funcname))
#============================================ end func top_head() ==================================================================
#===================================================================================================================================
if __name__=="__main__":
  OP_kps_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006____VHMR_keypoints.json'
  img_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006_ORIG____VHMR.png'
  openpose_2_deepercut(OP_kps_fname, img_fname)
#===================================================================================================================================





































































