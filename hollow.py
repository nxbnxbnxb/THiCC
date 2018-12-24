
import numpy as np
from copy import deepcopy

def save_hollowed(orig_npy_fname, dest_fname):
  def hollow(model):
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
  # =========== end func def of hollow(model): =============
 # back to func save_hollowed():
  m=np.load(orig_npy_fname)
  np.save(dest_fname, hollow(m)); return
# end func def of save_hollowed(orig_npy_fname, dest_fname):

if __name__=="__main__":
  save_hollowed('body_nathan_.npy', 'skin_nathan_.npy')

