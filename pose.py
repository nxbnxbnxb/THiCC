from measure import measure
from seg import segment_black_background
from viz import pltshow
import viz
from utils import pe,np_img

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy
import skimage as skimg
from pprint import pprint as p
import imageio as ii

pr=print
#=================================================================================================================================
def openpose_2_deepercut(openpose_fname, img_fname):
  no_background, mask=segment_black_background(img_fname)
  pltshow(no_background)

  img=np_img(img_fname)
  OP_kps=measure(openpose_fname)
  mask=mask.astype('uint8')
  mask[np.nonzero(mask)]=255
  mask=skimg.transform.resize( mask, img.shape, anti_aliasing=True )
  marked=mark_img(img, OP_kps, mask)
  return marked
#============================================ end func openpose_2_deepercut() ====================================================
#=================================================================================================================================
def mark_img(img, OP_kps, mask):
  '''
    Marks img w/ pertinent openpose keypoints OP_kps   and also calculates top of the head.
  '''
  # TODO: appropriate resizes to get DCut/MuVS working
  funcname=sys._getframe().f_code.co_name
  marked=deepcopy(img)
  neck=OP_kps['Neck']
  nose=OP_kps['Nose']
  neck=np.array([neck['x'],neck['y']]).astype('float64')
  nose=np.array([nose['x'],nose['y']]).astype('float64')
  top_of_head=top_head(neck, nose, mask)
  OP_kps=deepcopy(OP_kps)#; print("OP_kps: "); p(OP_kps)
  OP_kps['TopHead']={
    'x':top_of_head[0],
    'y':top_of_head[1],
    'c':np.mean([ OP_kps['Nose']['c'],
                  OP_kps['Neck']['c']])}
  body_parts=[ "TopHead", "Neck",
  "LWrist", "LShoulder", "LElbow", "LHip", "LKnee", "LAnkle",
  "RWrist", "RShoulder", "RElbow", "RHip", "RKnee", "RAnkle",]
  red     = [255.0,  0.0,  0.0] #[255,0,0]           #'red'#np.array([[1,0,0]])
  green   = [  0.0,255.0,  0.0] #[255,255.0,0]       #'#00FF00'#'green'#np.array([[0,1,0]])
  blue    = [  0.0,  0.0,255.0] #[255,0,255.0]       #'blue'#np.array([[0,0,1]])
  cyan    = [  0.0,255.0,255.0] #[255,255.0,255.0]   #'#00FFFF'#np.array([[0,1,1]])
  yellow  = [255.0,255.0,  0.0] #[255,255.0,0]       #'#FFFF00'#np.array([[1,1,0]])
  magenta = [255.0,128.0,255.0] #[255,0,255.0]       #'#FF88FF'#np.array([[1,0.5,1]]) # Todo: match DC's shade of pink.   Mostly done.
  black   = [  0.0,  0.0,  0.0] #'#000000'#np.array([[0,0,0]])
  white   = [255.0,255.0,255.0] #'#FFFFFF'#np.array([[1,1,1]])
  DRAW_CIRC_RAD= 8.5  # sizing; match DCut img sizes
  for part in body_parts:
    # switch statement:
    if   part=='LWrist'     or part=='LAnkle':  color=yellow
    elif part=='LElbow'     or part=='LKnee':   color=magenta
    elif part=='LShoulder'  or part=='LHip':    color=cyan
    elif part=='RWrist'     or part=='RAnkle':  color=red
    elif part=='RElbow'     or part=='RKnee':   color=green
    elif part=='RShoulder'  or part=='RHip':    color=blue
    elif part=='TopHead':                       color=white
    elif part=='Neck':                          color=black
    marked=viz.draw_circ(marked, int(round(OP_kps[part]['x'])), int(round(OP_kps[part]['y'])), DRAW_CIRC_RAD, color)
  return marked
#=============================================end func mark_img(params):==================================================
  """
  # TODO: I think the problem is we have to raw the circles on the numpy image itself to avoid resizing to the plt window.  -nxb, (Wed Apr  3 07:03:23 EDT 2019)

  #marked_fname="marked.png"
  extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # I think it's THIS line changing the aspect ratio. TODO TODO understand what this line does.    (ie. play with it in the REPL)
  orig_edges=extent.get_points()
  tighter=deepcopy(orig_edges)
  # these crop amts seem p generalized
  tighter[0,0]+=0.03000 # x left    TODO: generalize based on y_len and x_len.
  tighter[1,0]-=0.0200  # x right
  tighter[0,1]+=0.03000 # y bot
  tighter[1,1]-=0.0227  # y top       A little tooo little off each end.  (Tue Apr  2 13:51:11 EDT 2019)
  print("left        :",tighter[0,0])   # left
  print("right       :",tighter[1,0])   # right
  print("bot         :",tighter[0,1])   # bot
  print("top         :",tighter[1,1])   # top
  orig_x_len=orig_edges[1,0] - orig_edges[0,0]
  orig_y_len=orig_edges[1,1] - orig_edges[0,1]
  print("orig_x_len:",orig_x_len)
  print("orig_y_len:",orig_y_len)
  extent.set_points(tighter)

  print('type(extent):',type(extent))   #type(extent): <class 'matplotlib.transforms.Bbox'>
  print('extent:',extent)               #extent: Bbox(x0=0.765, y0=0.493, x1=5.795, y1=4.2589999999999995)
  """
  #fig.axes.get_xaxis().set_visible(False)
  #fig.axes.get_yaxis().set_visible(False)
  #plt.imsave(marked_fname, ) # old answer, but worth a shot
  #plt.savefig(marked_fname,bbox_inches=extent)#,bbox_inches=extent) #pad_inches=0,bbox_inches=extent)    # Note: plt.savefig() doesn't pltshow()
  '''
  plt.clf() # or cla() or plt.close()
  # TODO: plt.imsave() but do it properly.
  '''
  """
  print("leaving",funcname)
  plt.savefig(marked_fname,bbox_inches=extent)#,bbox_inches=extent) #pad_inches=0,bbox_inches=extent)    # Note: plt.savefig() doesn't pltshow()
  """
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
  MuVS_OP_kps_fname     = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006____VHMR_keypoints.json'  # from MuVS
  n8_OP_kps_fname       = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE_____keypoints.json'# NOTE: goal img
  MuVS_img_fname        = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/MuVS_HEVA_frame0006_ORIG____VHMR.png'
  n8_img_fname          = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg'
  # '/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg'
  n8_back_OP_kps_fname  = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_back___jesus_pose___legs_closed___nude___grassy_background_Newark_DE_____keypoints.json'
  n8_back_img_fname     = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_back___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg'
  img                   = openpose_2_deepercut(n8_OP_kps_fname, n8_img_fname) # TODO: test on multiple images; try to get n8 working on your own video.
  #img                  = openpose_2_deepercut(MuVS_OP_kps_fname, MuVS_img_fname) # TODO: test on multiple images; try to get MuVS working on your own video.
  marked_fname          = "marked.png"
  ii.imwrite(marked_fname, img)
  pltshow(img)
#===================================================================================================================================





































































