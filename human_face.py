import imageio as ii
import numpy as np
import scipy.misc as scpm
from copy import deepcopy

import viz

def paste_face___onto_mesh(facemask, color_img):
  pass # TODO:  this here is the whole chalupa

def face(facemask,color_img):
  '''
  return color_img and\
    np.dstack((facemask,facemask,facemask))
  '''
  scpm.imshow(color_img)
  face_img=deepcopy(color_img)
  facemask_w_RGB=np.dstack((facemask,facemask,facemask))
  print("facemask_with_RGB.shape:\n",facemask_w_RGB.shape)
  for i in range(face_img.shape[0]):
    for j in range(face_img.shape[1]):
      for k in range(face_img.shape[2]):
        if not facemask_w_RGB[i,j,k]:
          face_img[i,j,k]=0
  return face_img
  #on_locs=np.nonzero(facemask_with_RGB) # this one DOESN'T work
  #return color_img == np.dstack((facemask,facemask,facemask))
  #return np.logical_and(color_img,
    #np.dstack((facemask,facemask,facemask))).astype('float64')

def fill_in_below(mask, longersshape):
  '''
    img util
  '''
  return np.concatenate(
    (mask, 
      np.zeros((longersshape[0]-mask.shape[0], mask.shape[1])).astype('float64')),
    axis=0)


def facemask_from_mask(mask):
  '''
    precondition: input parameter mask is in Jesus pose (arms in a "T" pose)
  '''
  # TODO: fix human body segmentation s.t. Jesus-pose-segmentation works properly; as is, the arms become very shriveled and almost vanish.
 
  arm_length  = np.max(np.count_nonzero(mask,axis=1))
  y_arms      = np.argmax(np.count_nonzero(mask,axis=1))
  for y in range(y_arms,0,-1):
    if np.count_nonzero(mask[y]) < arm_length/3.:
      return fill_in_below(mask[:y],mask.shape) # face
    # TODO: vectorize this for-loop
  raise RuntimeError("please check the segmentation of the customer in \"Jesus pose\"")

def main1():
  color_img=np.asarray(ii.imread("/home/n/Pictures/smpl_result___male___0000000000.png")).astype('float64')
  mask=np.asarray(ii.imread("SMPL_blender_male_0000000000____mask_.jpg"))  # correct this first
  facemask=facemask_from_mask(mask)
  facemask=scpm.imresize(facemask, color_img.shape[:2], interp='bilinear')
  f=face(facemask,color_img)
  #print("f.shape:\n", f.shape)
  print("f.dtype:\n", f.dtype)
  print("f.shape:\n", f.shape)
  scpm.imshow(face(facemask,color_img))



if __name__=="__main__":
  main1()
