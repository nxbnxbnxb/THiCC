import numpy as np
import skimage
from viz import pltshow
import seg

#===================================================================================================================================
def crop_person(img, mask):
  '''
    ------
    Params:
    ------
    mask and img both numpy arrays representing images  (see chest_circum() in measure.py)

    Get mask by calling seg.seg_local(fname).   (in seg.py)
  '''
  # reuse of mask from earlier call to seg(imgfname)

  # Make mask and img the same size
  if img.shape != mask.shape:
    mask=skimage.transform.resize(mask.astype('float64'),
      img.shape[:2], anti_aliasing=True)
  mask=np.round(mask).astype('int64')

  # Adaptive PAD (width of img)
  PAD=int(round(img.shape[1]*0.10)) 
  ons   = np.nonzero(mask)

  # Here I use max(), min() b/c adding PADding might land bounds outside of img
  top   = max(  np.min(ons[0])-PAD,     0)
  bot   = min(  np.max(ons[0])+PAD,     mask.shape[0])
  # left and right are from our (an onlooker's) perspective, 
  #   not the perspective of a person within the picture's view
  left  = max(  np.min(ons[1])-PAD,     0)
  right = min(  np.max(ons[1])+PAD,     mask.shape[1])

  # Crop.
  RGB=3
  if len(img.shape) == RGB:
    cropped=img[top:bot, left:right,:]
  else:
    cropped=img[top:bot, left:right]
  crop_amts={
    # TODO: there might be an extra +-1 offset here depending on how we use dist_bot, etc. later
    'crop_amt_bot':img.shape[0]-bot,
    'crop_amt_top':top,
    'crop_amt_left':img.shape[1]-left,
    'crop_amt_right':right,
  }
  #pltshow(cropped) # for debugging
  return cropped, crop_amts  # returning mask becuase it has been resized
#===================================================================================================================================
def crop_mask(mask):
  '''
    ------
    Params:
    ------
    mask and img both numpy arrays representing images  (see chest_circum() in measure.py)

    Get mask by calling seg.seg_local(fname).   (in seg.py)
  '''
  # reuse of mask from earlier call to seg(imgfname)
  # Make mask and img the same size
  mask=np.round(mask).astype('int64')

  # Adaptive PAD (width of img)
  PAD=int(round(img.shape[1]*0.10)) 
  ons   = np.nonzero(mask)

  # Here I use max(), min() b/c adding PADding might land bounds outside of img
  top   = max(  np.min(ons[0])-PAD,     0)
  bot   = min(  np.max(ons[0])+PAD,     mask.shape[0])
  # left and right are from our (an onlooker's) perspective, 
  #   not the perspective of a person within the picture's view
  left  = max(  np.min(ons[1])-PAD,     0)
  right = min(  np.max(ons[1])+PAD,     mask.shape[1])

  # Crop.
  RGB=3
  if len(img.shape) == RGB:
    cropped=img[top:bot, left:right,:]
  else:
    cropped=img[top:bot, left:right]
  crop_amts={
    # TODO: there might be an extra +-1 offset here depending on how we use dist_bot, etc. later
    'crop_amt_bot':img.shape[0]-bot,
    'crop_amt_top':top,
    'crop_amt_left':img.shape[1]-left,
    'crop_amt_right':right,
  }
  #pltshow(cropped) # for debugging
  return cropped, crop_amts

#===================================================================================================================================
def center_vert(cropped1, cropped2):
  '''
    Overview function
  '''
  mask1=seg(img1)
  mask2=seg(img2)
  mid1=((max(mask1)-min(mask1))/2) + min(mask1)
  mid2=((max(mask2)-min(mask2))/2) + min(mask2)
  crop()
#===================================================================================================================================



if __name__=="__main__":
  crop_person()
