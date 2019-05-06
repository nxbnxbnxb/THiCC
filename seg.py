from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for records of code that USED to be in seg.py, see "old_seg.py" at:  /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/old/old_seg.py
#   https://www.diffchecker.com/image-diff

import matplotlib.pyplot as plt
import numpy as np
import imageio as ii

from copy import deepcopy
import sys

from utils import pltshow

#====================================================================
def img_diff_show(imgs):
  '''
    See "/home/n/Documents/code/old/hmr/demo.py" for the full version of "pltshow(many_imgs)"
    This is the general idea, but it might be buggy.
  '''
  funcname=  sys._getframe().f_code.co_name
  print("entering function "+funcname)
  plt.figure(1)
  plt.clf()

  img1=imgs[0]
  plt.subplot(231)
  plt.imshow(img1)
  plt.title('img1')
  plt.axis('off')

  # 2nd img
  img2=imgs[1]
  plt.subplot(232)
  plt.imshow(img2)
  plt.title('img2')

  plt.draw()
  plt.show()
  print("leaving function "+funcname)
  plt.close()
#====================================================================

#====================================================================
def img_diff(img1, img2):
  '''
    Adaptive to images
    As of Sun May  5 19:18:40 EDT 2019, bugs with np.greater(   np.linalg.norm(  img1-img2)):
      It breaks:

      1.  If the person's SHADOW is visible in the "T-pose picture" and the floor/ground color in the "background" photo is light 
        a.  (if the floor is black or dark, we don't have problems because we don't recognize the shadow as part of the silhouette)
      2.  If there are REFLECTIONS in the "T-pose picture"
        a.  (ie. shiny floors, mirrors, etc.)
        b.  img_diff() will pick these up as part of the silhouette, when actually they are artifacts of the environment rather than of the person
      3.  If skin/clothing color is too similar to the color of the background,
      4.
      5.
      6.
      7.

    If time complexity is too slow here,   gotta speed it up somehow...
    This assumes we actually have users, lol
  '''
  # Convert to 'int64':
  #   This lets the diff, "np.linalg.norm", and "np.greater()" properly  cut the person out of the picture:
  if img1.dtype=='uint8':
    img1=deepcopy(img1).astype('int64')
  if img2.dtype=='uint8':
    img2=deepcopy(img2).astype('int64')

  RGB=2
  RGB_THRESH=123#99
  simple_diff=img1-img2
  big_enough_diff=np.greater(
    np.linalg.norm(simple_diff, axis=RGB),
    RGB_THRESH)
  return big_enough_diff
#====================================================================



#========================================================================
if __name__=='__main__':
  nxb_img1=np.asarray(ii.imread(
    '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/n8_T_pose___reflective_floor_ISE_UDel_____2019_05_05____18:29_PM.jpg'))
  nxb_img2_background=np.asarray(ii.imread(
    '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/n8_T_pose___reflective_floor_ISE_UDel___just_background_____2019_05_05____18:29_PM.jpg'))
  diff=img_diff(nxb_img1,nxb_img2_background)
  pltshow(diff)
  #diff=diff.reshape(*diff.shape,1)
  #diff=np.concatenate((diff,diff,diff),axis=2)
  #print("diff.dtype = ",diff.dtype)
  #print("nxb_img1.dtype = ",nxb_img1.dtype)
  pltshow(diff)
  #np.logical_and(nxb_img1,diff))     # old/old_seg.py's "def segment_black_background(local_fname):"
#========================== end __main__ =====================================



















































# Glossary:         glossary:
'''
  Function definitions (function headers)

  As of Sun May  5 19:46:27 EDT 2019, headers are:
    17:def img_diff_show(imgs):
    45:def img_diff(img1, img2):


  tags glossary gloss defs funcs britney bitch
'''


