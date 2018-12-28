
from model  import *
from viz    import *
from utils  import *
from d      import *
import seg

#===================================================================================================================================
def test_rot8_4_noise(angle):
  # mask 1
  mask_2d = seg.main('http://columbia.edu/~nxb2101/180.0.png'); max_dim   = max(mask_2d.shape); shape=(max_dim, max_dim, max_dim); shape_2d=(max_dim, max_dim); mask_2d   = pad_all(mask_2d, shape_2d); model     = np.ones(shape).astype('bool'); print "mask: "; pltshow(mask_2d)
  model     = mask(model, mask_2d)
  print "before rotate:"
  show_all_cross_sections(model, freq=50) # NOTE: good.  it worked this time
  model     = rot8(model, angle)
  print "after  rotate:"
  show_all_cross_sections(model, freq=20); return model
#===================================================================================================================================
if __name__=='__main__':
  test_rot8_4_noise(45.9)

