

import numpy as np
skin_nathan=np.load("skin_nathan_.npy")
locs=np.nonzero(skin_nathan)

print("A local skin region.  Will it work with Voro poles?")
for i in range(9):
  print((locs[0][i],locs[1][i],locs[2][i]))

