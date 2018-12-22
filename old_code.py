
#===================================================================================================================================
def binarize(mask_3_colors):
  RED=0; CERTAIN=256.0; probable = np.array(int(CERTAIN / 2)-1) # default color is magenta, but the red part shows 
  mask_binary = deepcopy(mask_3_colors[:,:,RED])
  return np.greater(mask_binary, probable).astype('bool')
#===================================================================================================================================
def test_mask_func(mask_fname):
  mask_2d   = np.asarray(ii.imread(mask_fname)).astype('bool')
  #pltshow(mask_2d)     # this mask was fine
  depth     = mask_2d.shape[0]
  model     = np.ones((mask_2d.shape[0], depth, mask_2d.shape[1])).astype('bool')
  if debug:
    cross_sections_biggest(model)

  model     = mask(model, mask_2d)
  if debug:
    cross_sections_biggest(model)
    show_cross_sections(model, axis='y')

  return model
#====================================   end func def test_mask_func(mask_fname):   =======================================================

