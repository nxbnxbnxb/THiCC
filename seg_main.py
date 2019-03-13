import glob
import os
import sys
from seg import segment_local as seg_local
from utils import pn
import numpy as np
np.seterr(all='raise')

def segs(img_dir):
  # TODO: sort properly according to .... time?  You'll figure it out later.
  all_imgs=img_dir
  if all_imgs[-1]=='/':
    all_imgs+='*'
  else:
    all_imgs+='/*'
  print('all_imgs:\n',all_imgs)
  img_paths=glob.glob(all_imgs)
  print("img_paths:\n",img_paths); pn(3)
  #img_paths=sorted(img_paths)    # nOTE: this line works if img_filenames are 0000000.png, 0000001.png,0000002.png,0000003.png,... etc.
  # sort earliest to beginning of list
  img_paths=sorted(img_paths, key=os.path.getmtime)
  print("img_paths after sorting:\n",img_paths)
  '''
  for img_path in img_paths:
    mask=seg_local(img_path)
  '''

if __name__=="__main__":
  img_dir=sys.argv[1]
  segs(img_dir)

'''

  This plan WILL change, but at the same time it's important to organize ideas.

  Basically, the idea is to:
  1.  Force data acquisition in Jesus pose (or really any controlled pose)
  2.  frames  = cut(vid)
  3.  mask    = seg(frame)
  4.  angle   = detect(mask, vid) 
    #Autodetect angle of frame
  5.  model   = upd8(model, mask, angle)
    #Use that angle and the image (and resultant segmentation mask) at that angle to mask the voxels
  6.  smpl    = fit(model)
    #Based on some loss function, fit SMPL model to data from customer video.
    #Potential metrics:
    a.  L2 distance between shell and SMPL mesh pts
    b.  Perimeter of a convex hull surrounding the points at that height
    c.  Area of slice @ that height vs area of a slice of the SMPL model @ that height



'''


