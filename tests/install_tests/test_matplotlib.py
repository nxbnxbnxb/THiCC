import matplotlib.pyplot as plt
import numpy as np

def pltshow(im):
  plt.imshow(im); plt.show(); plt.close()

sqr=np.ones((3,3))
sqr[1,1]=0
pltshow(sqr)

import os
print('  completed '+os.path.basename(__file__)+'\n'*1)
