import imageio as ii
import numpy as np
import scipy.misc as scpm
from copy import deepcopy

import sys


def paste_face___onto_mesh(facemask, color_img):
  pass # TODO:  this here is the whole chalupa

def face(facemask,color_img):
  '''
    Cuts out the face from the image
    Relies too much on out-of-the-box precision of the segmentation algorithm
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
  # TODO: optimize (vectorize) the above for loop; I tried quite a few ways and somehow masking-in-3-D-with-floating-points-values-instead-of-just-booleans was harder than anticipated.  Props if you can figure out which out-of-the-box numpy function perfectly does this.
  return face_img

def fill_in_below(mask, longersshape):
  '''
    Img util to help mask
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
  # NOTE:  I think this might be a more general flaw with Google's deeplab_model.tar.gz?  I'm not 100% sure whether Matterport's Mask-R-CNN would be any better, though.  We can always try; I doubt we're going to get a panacea solution
 
  arm_length  = np.max(np.count_nonzero(mask,axis=1))
  y_arms      = np.argmax(np.count_nonzero(mask,axis=1))
  # TODO: vectorize this for-loop (into numpy).  Maybe count_nonzero outside the for_loop?  If I remember correctly, np.count_nonzero(...,axis=0) was vectorized recently
  for y in range(y_arms,0,-1):
    if np.count_nonzero(mask[y]) < arm_length/3.:
      return fill_in_below(mask[:y],mask.shape) # face and garbage below it
  raise RuntimeError("please check the segmentation of the customer in \"Jesus pose\"")

def main1():
  if len(sys.argv) > 1:
    RGB_img_fname=sys.argv[1]
    binary_mask_fname=sys.argv[2]
  else:
    RGB_img_fname     = '/home/ubuntu/Pictures/jesus_pose____surfer.jpg'#'/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_29____09:19_AM__/000000000.jpg'
    binary_mask_fname = '/home/ubuntu/Pictures/jesus_pose____surfer.jpg____binary_mask_.jpg'#'/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_29____09:19_AM__/000000000.jpg'
  print("cutting out face from {0} \n  using mask from {1}".format(RGB_img_fname, binary_mask_fname))
  print("cutting out face from {0} \n  using mask from {1}".format(RGB_img_fname, binary_mask_fname))
  color_img=np.asarray(ii.imread(RGB_img_fname)).astype('float64')
  mask=np.asarray(ii.imread(binary_mask_fname)).astype('bool')  # correct this first
  facemask=facemask_from_mask(mask)
  facemask=scpm.imresize(facemask, color_img.shape[:2], interp='bilinear')
  f=face(facemask,color_img)
  scpm.imshow(f)



if __name__=="__main__":
  main1()
