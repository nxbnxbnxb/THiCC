import numpy as np
import math

from on_locs import *




angle=179.0 # degrees
angle=math.radians(angle)
model=np.zeros((9,9,9)).astype('bool')
model[1,1,1]=True
ons=on_locs_nonzero(model).T
print "initially, voxels are \n{0}".format(model)
print ons.shape
print ons

def shift(locs, delta):
  return locs+delta

ons=shift(ons, -model.shape[0]/2.)
print ons; print '\n'*3
z_rot = np.array([[ cos(angle), -sin(angle),      0     ],
                  [ sin(angle),  cos(angle),      0     ],
                  [     0     ,      0     ,      1     ]]).astype('float64') # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
print 'rotation matrix:'; print z_rot; print '\n'*2
ons=z_rot.dot(ons)
print 'rotated, on locs are \n{0} \n\n'.format(ons)
ons=shift(ons,  model.shape[0]/2.)
print 'shifted back, on locs are \n{0} \n\n'.format(ons)

def round____all_8_adj_voxels(locs, max_idx_val):
  # TODO:  double check all the type-casting.  When we rotate the locations, we need them to allow for floating point decimals, but whenever we use the indices as indices they need to be positive integers
  # wow, that was some long documentation.  Here's the function header again:  
  '''
def round____all_8_adj_voxels(locs, max_idx_val):
  '''
  full_locs=np.zeros((locs.shape[0]*  8 ,     locs.shape[1])).astype('uint64') # NOTE:   is uint64 too memory intensive?
  idx=0
  for loc in locs:
    x_low=floor(loc[0]);x_hi=ceil(loc[0]);y_low=floor(loc[1]);y_hi=ceil(loc[1]);z_low=floor(loc[2]);z_hi=ceil(loc[2])
    # error-checking  (make sure we don't index outside the bounds of the model).  duplicates are taken care of by np.unique() at the end
    if x_low < 0:
      x_low=   0
    if y_low < 0:
      y_low=   0
    if z_low < 0:
      z_low=   0
    if x_hi > max_idx_val:
      x_hi  = max_idx_val
    if y_hi > max_idx_val:
      y_hi  = max_idx_val
    if z_hi > max_idx_val:
      z_hi  = max_idx_val
    full_locs[idx  ]= x_low, y_low, z_low
    full_locs[idx+1]= x_low, y_low, z_hi
    full_locs[idx+2]= x_low, y_hi , z_low
    full_locs[idx+3]= x_low, y_hi , z_hi
    full_locs[idx+4]= x_hi , y_low, z_low
    full_locs[idx+5]= x_hi , y_low, z_hi
    full_locs[idx+6]= x_hi , y_hi , z_low
    full_locs[idx+7]= x_hi , y_hi , z_hi
    # you can think of this as the equivalent of enumerating all 8 of the "3 digit" binary numbers: 000, 001, 010, 011, 100, 101, 110, 111
    # TODO: check for out of bounds (ceil() > shape[0] or floor() < 0)
    idx+=8
  return np.unique(full_locs, axis=0).astype('uint64') # shape == (large_num, 3)
#========= end func def of round____all_8_adj_voxels(locs): ==========
# returning to function rot8():
#'''
ons   = round____all_8_adj_voxels(ons.T,model.shape[0]-1)  # ons.shape should be (large_num, 3) after this line.   (tall, not long)
#'''
print "unrounded, ons =\n{0}  \n\n".format(ons)
print np.round(ons)

rot8d = np.zeros(model.shape).astype('bool')
rot8d[ons[:,0], ons[:,1], ons[:,2]] = True  # the voxels with these x,y,z values
print "finally, voxels are \n{0}".format(rot8d)









































































