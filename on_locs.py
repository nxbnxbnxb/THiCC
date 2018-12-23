# imports
import numpy as np
import math
from math import cos, sin, ceil, floor


#================================================================
def on_locs_nonzero(a):
  num_dims=3
  on_voxels=np.nonzero(a) # as of Dec. 21, 2018 with the following versions, dtype is default 'int64'
  #  numpy info from conda:
  #python      2.7.15  h33da82c_4    conda-forge
  #numpy       1.15.4  py27_blas_openblashb06ca3d_0  [blas_openblas]  conda-forge
  #numpy-base  1.15.4  py27h2f8d375_0  
  n_pts=on_voxels[0].shape[0]
  coords=()
  for i in range(num_dims):
    coords+=(on_voxels[i].reshape(n_pts,1).astype('int64'),)
  return np.concatenate(coords,axis=1).astype('int64') # locations have to be integer (indices)
  # TODO: Test!     shape=(3817528,3)
#============  end func def of  on_locs_nonzero(a):  ============
def rot8(model,angle,axis='z'):
  '''
    rot8s the model (counterclockwise from the top?  TODO: double-check this) angle, by default around the z axis

    Parameters
    ----------
    model: a 3-d numpy array that is a cube (model.shape[0]==model.shape[1]==model.shape[2])
    angle: in degrees, a float
  '''
  assert model.shape[0] == model.shape[1] == model.shape[2] # model is cube-shaped
  print "angle (in degrees) is {0}".format(angle)
  angle = math.radians(angle)       # for use in cos(angle), sin(angle)
  ons   = on_locs_nonzero(model).T  # Transposed so we can rotate the coords with R_z*coords
  print "ons.shape is {0}".format(str(ons.shape))  # oughta have shape (big_num, 3)
  #=====================================================
  def shift(locs, delta):
    '''
      Shifts a np array +delta, +delta
                or      +delta, +delta, +delta for 3D np arrays (our use case)
      Locs are a np.array of shape (num_pts, 3), for example: a specification of a cube near the origin ranging from (0, 0, 0) to (1, 1, 1)
        >>> locs
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  1.],
               [ 0.,  1.,  0.],
               [ 0.,  1.,  1.],
               [ 1.,  0.,  0.],
               [ 1.,  0.,  1.],
               [ 1.,  1.,  0.],
               [ 1.,  1.,  1.]])
    '''
    return locs+delta
  #======== end func def of shift(locs, delta): =========
  ons   = shift(ons,-((model.shape[0]/2.0)-0.5)) # shift the array s.t. center of 3d array is at (0, 0, 0)    # TODO: rewrite as "center_at____0_0_0()"
  # NOTE: we have a "-0.5" in here for the following reason.  Consider the simple 3x3x3 array in numpy.  It is centered around the value at indices (1,1,1).  This is (3.0/2.0)-0.5.  Consider the 4x4x4 array.  It is centered between indices 1 and 2, at (1.5,1.5,1.5).  Both of these would be centered if they were shifted -((model.shape[0]/2.0)-0.5).  If you are not satisfied with 2 examples, please realize this is the algorithm because it would be shape[0]/2.0 if the indexing went from 0 to shape[0].  ie. if shape[0] were 4 and therefore the indices went from 0 to 4, 2 WOULD be the central index.  The -0.5 is only present because shape[0]==4 implies indices from 0 to 3 instead; the "subtracted -1 from the length" turns into "subtraced -0.5 from the middle"
  np.save("ons____before_rotating.npy", ons)
  z_rot = np.array([[ cos(angle), -sin(angle),      0     ],
                    [ sin(angle),  cos(angle),      0     ],
                    [     0     ,      0     ,      1     ]]).astype('float64') # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
  ons   = z_rot.dot(ons)
  ons   = shift(ons,  (model.shape[0]/2.0)-0.5) # shift the array s.t. center of 3d array is back at (max/2, max/2, max/2.)  # TODO: rewrite as "recenter()"
  np.save("ons____before_rounding.npy", ons)
  print "="*99
  print "saved on_locs"
  print "="*99
  #====================================================================
  def round____all_8_adj_voxels(locs, max_idx_val):
    '''
      Returns "rounded" version of locs, but with lots of "filling in the dots."
        For instance,
          Consider a loc (1.1,1.1,1.1).  We want to return locs containing (1,1,1),(1,1,2),(1,2,1),(1,2,2),    (2,1,1),(2,1,2),(2,2,1), and (2,2,2),  rather than containing 1 or some of those new locations
      Assumes the 3-D numpy array we're working with is a cube!

      Function first written
      ----------------------
      Dec. 21, 2018

      Last edit date
      --------------
      Dec. 21, 2018

      Parameters
      ----------
        locs: a numpy array with shape (large_num, 3), where the dtype of each element is probably 'float64' (any decimal representation will do)

      Notes
      -----
      I wrote this code because originally I was using scipy.ndimage.rotate().  But that left little "holes" and other weird behavior I couldn't explain.  Imagine if you rotated a numpy array that looked like the following:
                                                     00000
                                                     01110
                                                     01110
                                                     01110
                                                     00000

      by 45 degrees and the result looked like:
                                                    0000000
                                                    0001000
                                                    0011100
                                                    0110110
                                                    0011100
                                                    0001000
                                                    0000000
      It's true, there's weirdness b/c the new Cartesian grid doesn't neatly contain all the voxels that were defined by the first Cartesian grid, but this is inevitable as long as our skin data is discrete voxels rather than continuous surfaces (an example of a continuous surface would be the paraboloid defined by z = x^2 + y^2; no matter how high-definition the 3-D voxel grid is, the continuous surface will always be more precise).  The problem is that '0' (non-human) in the middle.  I haven't tested it fully yet (Dec. 21, 2018), but I wrote the code this way to avoid those holes (non-human 0 voxel) as much as possible.  NOTE: scipy.ndimage.rotate() is ESPECIALLY unacceptable when the input is 3-D voxels.  The "holes" illustrated above don't appear when you use rotate() on a 2-D image (ie. https://stackoverflow.com/questions/46657423/rotated-image-coordinates-after-scipy-ndimage-interpolation-rotate)
      Assumes the 3-D numpy array we're working with is a cube!
    '''
    # TODO:  double check all the type-casting.  When we rotate the locations, we need them to allow for floating point decimals, but whenever we use the indices as indices they need to be positive integers
    # wow, that was some long documentation.  Here's the function header again:  
    '''
  def round____all_8_adj_voxels(locs, max_idx_val):
    '''
    print "max_idx_val is {0} \n".format(max_idx_val)
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
        if np.random.random() < 0.05:
          print "x_hi greater.  max_idx_val is {0}".format(max_idx_val)
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
  ons   = round____all_8_adj_voxels(ons.T,model.shape[0]-1)  # ons.shape should be (large_num, 3) after this line.   (tall, not long)
  print "at line 114 of on_locs.rot8(), ons.shape      is {0}".format(str(ons.shape))
  print "at line 115 of on_locs.rot8(), ons[:,0].shape is {0}".format(str(ons[:,0].shape))
  print "at line 116 of on_locs.rot8(), ons[:,1].shape is {0}".format(str(ons[:,1].shape))
  print "at line 117 of on_locs.rot8(), ons[:,2].shape is {0}".format(str(ons[:,2].shape))
  print "at line 118 of on_locs.rot8(), max(ons[:,0]) is {0}   \n    and min(ons[:,0]) is {1}".format(str(np.amax(ons[:,0])), str(np.amin(ons[:,0])))
  print "at line 119 of on_locs.rot8(), max(ons[:,1]) is {0}   \n    and min(ons[:,1]) is {1}".format(str(np.amax(ons[:,1])), str(np.amin(ons[:,1])))
  print "at line 120 of on_locs.rot8(), max(ons[:,2]) is {0}   \n    and min(ons[:,2]) is {1}".format(str(np.amax(ons[:,2])), str(np.amin(ons[:,2])))
  print '\n'*2
  rot8d = np.zeros(model.shape).astype('bool')
  rot8d[ons[:,0], ons[:,1], ons[:,2]] = True  # the voxels with these x,y,z values
  return rot8d
#========= end func def of rot8(model,angle,axis='z'): =========

'''
>>> y = np.arange(35).reshape(5,7)
>>> y
array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34]])
>>> y[np.array([0,2,4]), np.array([0,1,2])]
===========================================================

y = np.arange(35).reshape(5,7)
print y
y[np.array([0,2,4]).reshape((1,3)), np.array([0,1,2]).reshape((1,3))]

'''































if __name__=='__main__':
  #    array alloc only:
  # voxels=np.round(np.random.random(( 363,363,363))).astype('bool')
  #        1.060s
  ons

  ons = on_locs_random_timing_test(363,363,363)
  #   our basic voxel 3d-np-arrays are 513x513x513
  '''
    For 400,400,400, the results were:
      574.208s
    and the process took up 12+ Gigabytes of RAM (slowed down and almost killed my dinky lil laptop)
    O(6.4*10^7)
    For 200x200x200,
       10.845s
      and
      1 GB
    For 252x252x252,  (   O(twice as much as 200x200x200)     )
       19.101s
      and
      2 GB
    For 317x317x317   (   2   times as much as 252x252x252    )
       38.077s
      and
      4 GB
    For 363x363x363   (   1.5 times as much as 252x252x252    )
       57.557 - 58.247s  (range)
      and
      6.2 GB
    For 400x400x400
      519.2 seconds
      9.8 GB
      This added complexity (ESPECIALLY the time complexity is 100% due to memory-swap interchange, as it's called on my system monitor)





  timing ON_LOCS for 513x513x513:
    29.2014601231 seconds
  on the other hand,
  np.ARGSORT() takes:
    3.28244996071 seconds

  '''






































































#================================================================
def on_locs_random_timing_test(a,b=1,c=1):
  '''
    finds where is "on" in the rand array  (mainly for speed tests to see whether the naive for loop is viable)
  '''
  voxels=np.round(np.random.random((   a,  b,  c))).astype('bool') # prob a faster way to do this but who cares
    # I think the above line was really slow and took up a lotta space b/c it had to first create the giant random array and THEN converted it all into "rounded booleans" BEFORE it could free up the original randoms array
  #                                 (400,400,400)
  return on_locs(voxels)
#================================================================
def on_locs(nparr):
  '''
    finds where is "on" in the array
  '''
  # NOTE:  weird!  this was fast for when I put it in model.py.  Dec. 20, 2018
  on_locs=[]
  for i in range(nparr.shape[0]): 
    for j in range(nparr.shape[1]): 
      for k in range(nparr.shape[2]): 
        if nparr[i,j,k]:
          on_locs.append(np.array([i,j,k]).reshape(3,1))
          # I can already see ways to optimize this method, at least row-wise, by returning a whole nparray   instead of doing it this way
          #   We could make on-locs a nparray and "cut off" the zeros at the end
  return on_locs
#================================================================
def on_locs_argsort(a):
  flat=a.reshape(np.prod(a.shape,1,1))
  on_locs=np.argsort(flat)
  on_locs=on_locs.reshape(a.shape)
  return on_locs
#================================================================
