
import numpy as np
np.seterr(all='raise')
from copy import deepcopy
from utils import pif

#=========================================================================
def hollow_3d_good(model):
  '''
      Takes a 'filled in' 3d np array (a volume) and makes it a shell (just the voxels on the surface)
      This func is SLOW.
    Tests:
      Dec. 25, 2018.  For np arr of shape (513, 513, 513), timing run gives results:
        real  0m28.008s
        user  0m27.792s
        sys   0m 0.484s
      Is this for real???  Hard to be sure
    
    Notes:
    ------
      I considered the cases where the voxels are on one of the edges of the np array i==0,j==0,k==0,  or i==shape[0]-1, j==shape[1]-1, or k==shape[2]-1.  But the best thing imho is to assume the human body is extending off into infinity in these cases.  Consider it for a second, and I hope you'll agree.  If model[0,j,k] is "on," probably model[-1,j,k] would also be "on."   (Dec. 25, 2018.  Yes, I'm a heathen for hating Christmas.  No, I won't join your Satanic Metal band.)
      Admittedly, it's also much simpler to write and understand code that doesn't do special things to account for this undefined edge-behavior.  "plus_i = i==model.shape[0]-1 or model[i+1,j,k]" may disguise my intent a bit, but it's fine so long as you understand why I'm doing it.  -NBendich, Dec. 25, 2018
  
  
  '''
  # NOTE:  this method makes a copy and uses the ORIGINAL as reference for the copy (so we don't have to worry about destroying our data while we attempt to properly process it)
  # TODO:   dramatically speed up this function 'hollow_3d(m)'
  #       perhaps there's a built-in np or    plt func
  hollowed = deepcopy(model)
  pif('model.shape: {0}'.format(model.shape))
  for i in range(model.shape[0]):
    for j in range(model.shape[1]):
      for k in range(model.shape[2]):
        if model[i,j,k]:
# doesn't work for edges!  hopefully we never encounter a model literally on the edge of its space
          plus_i  = i==model.shape[0]-1 or model[i+1,j,k]
          minus_i = i==0                or model[i-1,j,k]
          plus_j  = j==model.shape[1]-1 or model[i,j+1,k]
          minus_j = j==0                or model[i,j-1,k]
          plus_k  = k==model.shape[2]-1 or model[i,j,k+1]
          minus_k = k==0                or model[i,j,k-1]
          hollowed[i,j,k] = not(
                                    plus_i
                                and minus_i
                                and plus_j
                                and minus_j
                                and plus_k
                                and minus_k)
          # the voxel will probably end up still filled in, unless all adjacent voxels are 'on' too
          #  should leave a shell
          # I considered the cases where the voxels are on one of the edges of the np array i==0,j==0,k==0,  or i==shape[0]-1, j==shape[1]-1, or k==shape[2]-1.  But the best thing imho is to assume the human body is extending off into infinity in these cases.  Consider it for a second, and I hope you'll agree.  If model[0,j,k] is "on," probably model[-1,j,k] would also be "on."   (Dec. 25, 2018.  Yes, I'm a heathen for hating Christmas.  No, I won't join your Satanic Metal band.)
          # Admittedly, it's also much simpler to write and understand code that doesn't do special things to account for this undefined edge-behavior.  "plus_i = i==model.shape[0]-1 or model[i+1,j,k]" may disguise my intent a bit, but it's fine so long as you understand why I'm doing it.  -NBendich, Dec. 25, 2018
  return hollowed
#=========================================================================
def hollow_3d_fast(model):
  '''
    Uses approximation to cut time required to skin by a factor of ~6.  Still a bit buggy tho, prob best to use the "good" version of this function which doesn't leave these artifacts
    Tests:
      Dec. 25, 2018.  For np arr of shape (513, 513, 513), timing run gives results:
        real  1m3.249s
        user  1m3.154s
        sys 0m0.372s
  '''
  # NOTE:  perhaps we encounter the top or bottom of the human body (head or feet)
    #  is it worth throwing in more complex code to handle these situations?
  # NOTE 2:  1-somes or 2-somes of consecutive "on" same-k lines  (ie.  010 or 0110).  We shouldn't kill any 1s in this case.  Perhaps this is automatically handled by numpy indexing though!
  skin=deepcopy(model); OFF=0
  def empty(lis):
    return len(lis)==0
  for i in range(model.shape[0]):
    prev_start=[]     # TODO:    replace with box (only needs to store max 1 element)
    for j in range(model.shape[1]):
      for k in range(model.shape[2]):
        if empty(prev_start) and model[i,j,k]:
          prev_start.append(k)
        if not empty(prev_start) and not model[i,j,k]:
          if k > prev_start[0]+2:
            skin[i,j,prev_start[0]+1:k-1]=OFF # TODO: double check that this doesn't "turn off" all starts or all ends
            prev_start.pop()
      if not empty(prev_start): # reached end of this i,j sequence and was still "on" til the "wall"
        skin[i,j,prev_start[0]+1:model.shape[2]]=OFF
        prev_start.pop()
  return skin
#=========================================================================

# TODO: rename to "skin()" ??????  But it's a less general name.  Dec. 25, 2018
def hollow(model):
  '''
      Takes a 'filled in' 3d np array (a volume) and makes it a shell (just the voxels on the surface)
  '''
  # NOTE: IDEA:  use np.nonzero() and manipulate/analyze the locs instead of the actual model.  Perhaps the result will be faster or more space-effective
  in_rush=False
  if in_rush:
    return hollow_3d_fast(model)
  else:
    return hollow_3d_good(model)
# =========== end func def of hollow(model): =============

def save_hollowed(orig_npy_fname, dest_fname):
  m=np.load(orig_npy_fname)
  np.save(dest_fname, hollow(m)); return
# end func def of save_hollowed(orig_npy_fname, dest_fname):

if __name__=="__main__":
  save_hollowed('body_nathan_.npy', 'skin_nathan_.npy')

