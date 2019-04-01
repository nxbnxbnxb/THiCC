from measure import measure
from seg import segment_black_background
from viz import pltshow
from utils import pe,np_img

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy
import skimage as skimg
from pprint import pprint as p

pr=print
#=================================================================================================================================
def openpose_2_deepercut(openpose_fname, img_fname):
  no_background, mask=segment_black_background(img_fname)

  img=np_img(img_fname)
  OP_kps=measure(openpose_fname)
  mask=mask.astype('uint8')
  mask[np.nonzero(mask)]=255
  #pltshow(mask)
  #pltshow(no_background)
  mask=skimg.transform.resize( mask, img.shape, anti_aliasing=True )
  marked=mark_img(img, OP_kps, mask)
  pltshow(marked)
  return marked
#============================================ end func openpose_2_deepercut() ====================================================
#=================================================================================================================================
def mark_img(img, OP_kps, mask):
  '''
    Marks img w/ pertinent openpose keypoints OP_kps   and also calculates top of the head.
  '''
#====================================== TODO =====================================
  neck=OP_kps['Neck']
  nose=OP_kps['Nose']
  neck=np.array([neck['x'],neck['y']]).astype('float64')
  nose=np.array([nose['x'],nose['y']]).astype('float64')
  top_of_head=top_head(neck, nose, mask)
  OP_kps=deepcopy(OP_kps)
  OP_kps['TopHead']={
    'x':top_of_head[0],
    'y':top_of_head[1],
    'c':np.mean([ OP_kps['Nose']['c'],
                  OP_kps['Neck']['c']])}
  print("OP_kps: "); p(OP_kps)
  #xs,ys=OP_2_DC_kps(OP_kps)
  print("top_of_head: ",top_of_head)
  body_parts=[ "TopHead", "Neck",
  "LWrist", "LShoulder", "LElbow", "LHip", "LKnee", "LAnkle",
  "RWrist", "RShoulder", "RElbow", "RHip", "RKnee", "RAnkle",]
  red     = 'red'#np.array([[1,0,0]])
  green   = '#00FF00'#'green'#np.array([[0,1,0]])
  blue    = 'blue'#np.array([[0,0,1]])
  cyan    = '#00FFFF'#np.array([[0,1,1]])
  yellow  = '#FFFF00'#np.array([[1,1,0]])
  magenta = '#FF88FF'#np.array([[1,0.5,1]]) # TODO: match DC's shade of pink
  black   = '#000000'#np.array([[0,0,0]])
  white   = '#FFFFFF'#np.array([[1,1,1]])
  fig,ax = plt.subplots(1)
  ax.set_aspect('equal')
  ax.imshow(img)
  #plt.imshow(img)
  RAD=7.6 # worked for /home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006_ORIG____VHMR.png
  for part in body_parts:
    # case:   color='cyan', vs. color='blue' vs. ... etc.
    if part=='LWrist' or part=='LAnkle':
      color=yellow
      print("if part=='LWrist' or part=='LKnee':")
      print(part+" (x,y):")
      print(OP_kps[part]['x'],OP_kps[part]['y']);pe()
    elif part=='LElbow' or part=='LKnee':
      color=magenta
      print("elif part=='LElbow' or part=='LAnkle':")
      print(part+" (x,y):")
      print(OP_kps[part]['x'],OP_kps[part]['y']);pe()
    elif part=='LShoulder' or part=='LHip':
      color=cyan
    elif part=='RWrist' or part=='RAnkle':
      color=red
    elif part=='RElbow' or part=='RKnee':
      color=green
    elif part=='RShoulder' or part=='RHip':
      color=blue
    elif part=='TopHead':
      color=white
    elif part=='Neck':
      color=black
    # TODO: make circles ('o') bigger
    circ = Circle((OP_kps[part]['x'],OP_kps[part]['y']),  RAD,  color=color)
    ax.add_patch(circ)
    #ax=fig.add_subplot(111, aspect='equal')
    #plt.scatter(OP_kps[part]['x'], OP_kps[part]['y'], marker='o', c=color)
  marked_fname="marked.png"
  plt.axis('off')
  extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  #fig.axes.get_xaxis().set_visible(False)
  #fig.axes.get_yaxis().set_visible(False)
  #plt.imsave(marked_fname, ) # old answer, but worth a shot
  plt.savefig(marked_fname,bbox_inches=extent) #pad_inches=0,bbox_inches=extent)
  '''
  plt.show()
  plt.clf() # or cla() or plt.close()
  pr('after savefig()')
  pr('before plt.show().')
  #plt.show();plt.close()
  # TODO: plt.imsave() but do it properly.
  '''
  return np_img(marked_fname)
  # make big enough 2 see
  #top_of_head=dir_vert intersect mask edge
  #DC_json=OP_2_DC_dict(OP_kps)
#=============================================end func mark_img(params):==================================================

#===================================================================================================================================
def top_head(neck, nose, mask):
  '''
    calculates top of head within mask's coordinates (plt coords)
    x+ from left to right,
    y+ from top to bottom

    Debugged on 1 image.  We should double-check with more images later.
  '''
  RGB=3;X=0;Y=1
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
  dir_vect/=dir_vect[Y]
  x=nose[X]
  del_x=-dir_vect[X] # We're going "UP" the picture in y, meaning we're going "minus."  So the del_x has to be negative.
  nose_y=int(nose[int(round(Y))])
  # for each y step "up," change delta_x.  Then when we're off the head, return the prev "in-the-mask" pt.
  for y in range(nose_y,0,-1): # "up" is -y
    if not mask[y,int(round(x))]: #mask[y,x] because  x and y are "reversed" in numpy
      return (x, y+1)
    else:
      x+=del_x
  raise Exception("Top of head not found in function named '{0}.'  Please doublecheck your code".format(funcname))
#======================================= end func top_head(params) ========================================================
#===================================================================================================================================
if __name__=="__main__":
  OP_kps_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006____VHMR_keypoints.json'
  img_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006_ORIG____VHMR.png'
  MuVS_img=openpose_2_deepercut(OP_kps_fname, img_fname) # TODO: test on multiple images; try to get MuVS working on your own video.
  pltshow(MuVS_img)
#===================================================================================================================================





































































