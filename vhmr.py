# vhmr, or "Video Human Mesh Recovery"
import skimage.io as skio
import numpy as np
import sys
import datetime
import os

from demo import betas
import demo as hmr  # LOL.  Easier than changing the module name, though.

#==============================================================
def prepend_0s(int_str, num_digits=9):
  '''
    Params:
    -------
    int_str is a string

  '''
  #NOTE: I know there's a less "hack-y" way of putting zeros in front of a number.  But I don't wanna look it up when I can just rewrite the code myself.
  return '0'*(num_digits-len(int_str))+int_str
#=======================================================
def make_mesh(vid_fname, secs_between_frames, root_img_dir, should_put_timestamps=True, output_img_filetype='jpg'):
  '''
  SMPL mesh (search pdf on Google)

  -------
  Params:
    vid_fname is the fullpath, not a localpath

  '''

  import cv2 # TODO: resolve this cv2 import such that either base or conda or a virtualenv has access to every damn python module I need
  delay = secs_between_frames
  vidcap = cv2.VideoCapture(vid_fname)
  success = True # THOUGHT: there's gotta be a smarter way to do this without the special case at the beginning.
  BIG= 10000000 # TODO: put in the calculate-how-long-the-vid-is  code.  NOTE: 10,000,000 doesn't overflow my T420 Thinkpad Laptop's memory.  But 100,000,000 does.
  all_betas=np.zeros((BIG,10)).astype("float64")  # NOTE: numpy arr? dict?   Can calculate precisely how long we need this to be based on "delay" and the length of the vid

  # NOTE: I save the intermediate files partially just to let hmr demo.py run its course as it came, fresh out of the github repo.  I realize this might be problematic for numerous reasons later on, but for debugging purposes initially, it will be good to be able to look at each individual image that I "cut out" from the video file and 
  img_write_dir=root_img_dir
  if not root_img_dir.endswith('/'):
    img_write_dir+='/'
  if should_put_timestamps:
    timestamp=datetime.datetime.now().strftime('%Y_%m_%d____%H:%M_%p__') # p is for P.M. vs. A.M.
    img_write_dir+=timestamp+'/'
  os.system('mkdir '+img_write_dir) # note: if directory is already there, execution will continue

  i = 0
  while success:
    success, img = vidcap.read()
    if cv2.waitKey(delay) &  0xFF == ord('q'): # note: double-check exactly what this "ord('q')" means.
      break
    # note:  "I put the imwrite() AFTER the break so if the image fails to read, we don't save an empty .jpg file."   - Nathan (Mon Jan 14 13:33:26 EST 2019)
    img_fname='img_write_dir+"{0}.{1}'.format(prepend_0s(str(i)),output_img_filetype)
    cv2.imwrite(img_fname, img)
    sys.argv=['demo.py', '--img_path', 'hmr_input_img.png'] # FIXME:   this line (sys.argv=['...','...','...'] is very fragile and nonrobust
    all_betas[i]=hmr.betas(img, json_path=None)
    if i < 5:
      print("all_betas[i] is {0}".format(all_betas[i]))
    i += 1
  return SMPL_approx(all_betas) # success
#===== end func def of  make_mesh(**lotsa_params): =====
def SMPL_approx(all_betas):
  # TODO: try mean, median, various other beta-combinations
  return all_betas
#==============================================================

if __name__=="__main__":
  vid_fname='unclothed_outside_delaware_____uniform_background_with_wobbling.mp4'
  secs=1
  img_dir='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018'#'./imgs/'
  
  smpl=make_mesh(vid_fname, secs, img_dir)













