import numpy as np
np.seterr(all='raise')


# seg.py manually cropping "Jesus pose"

#================================================================
def seg_local_man_crop(local_filename):
  # man_crop means "manual crop"
  # tried, didn't work, at least on the shitty photo I took in the library.  (n_jesus_pose___library_0___plt_nums.png in my Google Drive)

  #img=scipy.ndimage.io.imread(local_filename)
  img=np.asarray(ii.imread(local_filename)).astype('float64') # TODO: delete this commented-out line
  img=img[200:1000] # crop for 
  #================================================================
  #================================================================
  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])
  #================================================================
  #================================================================
  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
  MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'
  model_dir = './'
  download_path = os.path.join(model_dir, _TARBALL_NAME)
  # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
  MODEL = DeepLabModel(download_path) # segment_local()
  FAIRLY_CERTAIN=127
  return seg_map(img, MODEL)
#=====  end func def of   segment_local(local_filename) =====
#=======================================================================================================================================
#=======================================================================================================================================
#=======================================================================================================================================


#measure.py calculation of circumference of ellipse
  #  My current understanding is that perfect calculation for the ellipse circumference is impossible.  So trying to brute-force by sympy.integrate() just won't work; I need an acceptable approximation.
#=======================================================================================================================================
def ellipse_circum_approx(a, b, precision=9):
  # TODO: Document this function
  '''
    precision
    https://www.mathsisfun.com/geometry/ellipse-perimeter.html:
    See: "Infinite Series 2"
 
    Picked this one because I can get it arbitrarily accurate.  Also, calculation is quick.
  '''
  #=======================================================================================================================================
  def fac(n):
    # rewrote this b.c. I didn't find a library that could do half-factorials.
    # n is any mixed number with "1/2" as the fraction at the end. n must also be greater than or equal to -0.5. (ie. -0.5, 0.5, 1.5, 2.5, ... etc.)
    #assert n-0.5 == int(n-0.5) and n >= -0.5   # asserts sometimes **** up in python just b/c... idek why.
    # TODO: double-check that n-0.5==int(n-0.5) works in all cases.
    if n ==-0.5:
      # base case for recursion
      return math.sqrt(math.pi)
    else:
      return n*fac(n-1)
  #=======================================================================================================================================
  def half_c_n(n):
    #choose_n
    #(0.5)
    #( n )
    #assert int(n) == n # integer   # asserts sometimes **** up in python just b/c... idek why.
    from math import factorial as f
    return fac(0.5)/\
      (f(n) * fac(n-0.5))
  #=======================================================================================================================================
  def series(n,h):
    # https://www.mathsisfun.com/numbers/factorial.html [and more www.mathsisfun.com links]

    # TODO: clean up all the "n"s and "i"s floating around and confusing things
    return np.sum([half_c_n(i)**2*(h**i) for i in range(n)])
  if b > a:
    tmp=a; a=b; b=tmp # swap such that a is always the semi-major axis (bigger than b)
  # TODO: double-check this is right
  h   = ((a-b)**2) / ((a+b)**2)
  return 4*a*math.pi*(a+b)*series(precision,h)
#==================================== ellipse_circum_approx() ==========================================================================





#from smpl/smpl_webuser_hello_world/hello_smpl.py
    '''
    if isinstance(flabbiness, Number):
      m.betas[2]          = flabbiness
    else:
      couch_potato        = m.betas[2]  # I honestly don't know how we can measure this variable.  Prob by asking the user.  But will they tell the truth?  Will they know?    (negative values for this mean a well-muscled guy)
    '''

#seg.py's  overlay_imgs()   (oughta be called  "superimpose_imgs()")
    #ii.imwrite(tmpfname,cutout2) # write specialized funcs that convert to/from PIL/np.
    #cutout2=np_img(tmpfname)

    #ii.imwrite(tmpfname,cutout1)
    #cutout1=np_img(tmpfname)
    #sp.call(['rm',tmpfname])
    #ii.imwrite(tmpfname,cutout2)
    #cutout2=np_img(tmpfname)
    #sp.call(['rm',tmpfname])
    #pltshow(cutout2)

    #sp.call(['rm',tmpfname])
    # TODO: finish!
    pass
    '''
    # `pltshow(cutout1[x_min_1:x_max_1,y_min_1:y_max_1])` is my version of "crop()" function for numpy representations of images
    # this ugly assignment is much shorter than the alternative (all fits on one line)
    edge_x_1, edge_y_1  = min(locs1[0]), min(locs1[1])  
    edge_x_2, edge_y_2  = min(locs2[0]), min(locs2[1])
    print("edge_x_1: \n",edge_x_1)
    print("edge_x_2: \n",edge_x_2)
    print("edge_y_1: \n",edge_y_1)
    print("edge_y_2: \n",edge_y_2)
    mask1_shape = (max(locs1[0])-min(locs1[0]),  max(locs1[1])-min(locs1[0]))
    mask2_shape = (max(locs2[0])-min(locs2[0]),  max(locs2[1])-min(locs2[0]))
    # "0" b/c no color shift
    pltshow(shift(cutout1, (-edge_x_1, -edge_y_1, 0)))
    pltshow(shift(cutout2, (-edge_x_2, -edge_y_2, 0)))
    # I'm doing this imwrite() b/c I don't know how to resize an image in np.ndarray();  I only know how to with PIL.Image.resize((x,y))
    tmpfname='tmp.png'
    ii.imwrite(tmpfname,cutout1)
    cutout1=Image.open(tmpfname)
    cutout1=cutout1.resize((mask2_shape), Image.ANTIALIAS)
    pltshow(cutout1)
    # TODO: scale both images s.t. they overlay
    # TODO: crop imgs b4 overlaying
    # TODO: finish this overlay_imgs() function!

    cutout2=np.array(cutout2)
    pltshow(cutout2+cutout1)  # looks funky.
    '''
#====================================================================
def overlay_imgs(img_fname_1, img_fname_2):
    '''
      works somewhat kinda (basically not at all)
    '''
    # nOTE:   the reason I did all this was I was trying to be too precise about the notion of "fit" to adjust SMPL directly to each image.  
    cutout1,  mask1 = segment_black_background(img_fname_1)
    cutout2,  mask2 = segment_black_background(img_fname_2)
    # centers of mass; we want ints so shift(img) is easy
    CoM1=np.round(np.array(
      CoM(mask1)))
    CoM2=np.round(np.array(
      CoM(mask2)))

    if debug:
      print("img_fname_1:\n",img_fname_1)
      print("img_fname_2:\n",img_fname_2)
      pltshow(cutout1)
      pltshow(cutout2)
    locs1=np.nonzero(mask1)
    locs2=np.nonzero(mask2)
    # TODO: double-check whether x and y here are ACTUALLY x and y
    x_min_1 = min(locs1[0])
    x_min_1, y_min_1, x_min_2, y_min_2, x_max_1, y_max_1, x_max_2, y_max_2 =\
       min(locs1[0]), min(locs1[1]),\
       min(locs2[0]), min(locs2[1]),\
       max(locs1[0]), max(locs1[1]),\
       max(locs2[0]), max(locs2[1])
    pltshow(cutout1[x_min_1:x_max_1,y_min_1:y_max_1])  # this version of crop() works!  TODO: use an official python `crop()` function
    pltshow(cutout2[x_min_2:x_max_2,y_min_2:y_max_2])  # this version of crop() works!
    cutout1=cutout1[x_min_1:x_max_1,y_min_1:y_max_1]
    cutout2=cutout2[x_min_2:x_max_2,y_min_2:y_max_2]

    mask1_shape = (y_max_1-y_min_1, x_max_1-x_min_1)
    mask2_shape = (y_max_2-y_min_2, x_max_2-x_min_2)
    print('\n'*2)
    print(mask1_shape) #(152, 336)
    print(mask2_shape) #(105, 260)
    print('\n'*2)

    tmpfname='tmp.png'
    ii.imwrite(tmpfname,cutout1)
    # I'm doing these `imwrite()`s b/c I don't know how to resize an image in np.ndarray();  I only know how to with PIL.Image.resize((x,y))
    cutout1=Image.open(tmpfname)
    sp.call(['rm',tmpfname])
    ii.imwrite(tmpfname,cutout2)
    # I'm doing these `imwrite()`s b/c I don't know how to resize an image in np.ndarray();  I only know how to with PIL.Image.resize((x,y))
    cutout2=Image.open(tmpfname)
    sp.call(['rm',tmpfname])
    print('hit1\n\n')
    pltshow(cutout1.resize((mask2_shape), Image.ANTIALIAS))
    #cutout1=cutout1.resize((mask2_shape), Image.ANTIALIAS)
    print('hit2\n\n')
    print(mask1_shape)
    print(mask2_shape)
    pltshow(cutout2.resize((mask1_shape), Image.ANTIALIAS))
    cutout2=cutout2.resize((mask1_shape), Image.ANTIALIAS)
    cutout1 = cutout1.convert("RGBA")
    cutout2 = cutout2.convert("RGBA")
    overlaid=Image.blend(cutout1, cutout2, 0.5)
    pltshow(overlaid)
    return overlaid
#===== end func def of  overlay_imgs(img_fname_1, img_fname_2): =====







# NOTE: don't call vis_segmentation() if matplotlib.pyplot crashes conda!
#================================================================
def vis_segmentation(image, seg_map):
  # NOTE:  currently not working.  To debug, please look backwards at prev version in git
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #ii.imwrite("_segmented____binary_.jpg", binarize(seg_image))
  #  NOTE:  saving is happening outside this method; we should have no side effects besides "show_img()" in this func
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.show()
# end func def of   vis_segmentation(image, seg_map):

# Doesn't really work
#====================================================================
def segment_transparent_background(local_fname):
  # TODO: clear out messy comments, old print statements, etc.
  # TODO:  put some of this code in a separate function
  # TODO: cleanup like "segment_black_background(local_fname); all NOTES at the top, etc.
  segmap= segment_local(local_fname)
  img   = Image.open(local_fname)
  # TODO: I really OUGHT to scale the mask to fit the dimensions of the image (we'd have better resolution this way)
  segmap= segmap.reshape(segmap.shape[0],segmap.shape[1],1)
  segmap=np.concatenate((segmap,segmap,segmap),axis=2)
  segmap=np.rot90(segmap)

  # NOTE:  PIL.resize(shape)'s shape has the width and height reversed from numpy's
  img=img.resize((segmap.shape[1],segmap.shape[0]), Image.ANTIALIAS)

  # weird stuff happened when I tried to convert to 'float64' in this `np.array(Image.open(fname))` line.
  img = np.array(img) 

  # as of (Wed Feb 20 17:49:37 EST 2019), segmap is 0 for background, 15 for human ( before astype('bool'))
  segmap=segmap.astype('bool')
  segmap=np.logical_not(segmap)
  img[segmap]=WHITE
  # cut out the human from the img
  if debug:
    pltshow(img)
  fname='person_cutout.png'
  ii.imwrite(fname,img)
  cutout=Image.open(fname)
  os.system('rm '+fname) # cleaning up the intermediate step
  cutout=cutout.convert('RGBA')
  datas =cutout.getdata()
  newData=[]
  for item in datas:
      if item[0] == WHITE and item[1] == WHITE and item[2] == WHITE:
          newData.append((WHITE, WHITE, WHITE, TRANSPARENT))
      else:
          newData.append(item)
  cutout.putdata(newData)
  cutout.save("person_cutout_transparent_background.png", "PNG")
  return img, np.logical_not(segmap) # NOTE: returns white background b/c can't return transparent in numpy arrs
#======== end func def of  segment_transparent_background(params): ========

# part of model.py's test_human():
"""
on_locs=np.nonzero(model)
model_2=deepcopy(model)
model_2[on_locs[0],on_locs[1],on_locs[2]]=MID

print "on_locs[0].shape is {0}".format(on_locs[0].shape)
  # on_locs[0].shape is (3817528,)

on_locs=np.nonzero(model)
model_2=deepcopy(model)
model_2[on_locs[0],on_locs[1],on_locs[2]]=MID
assert np.array_equal(model, model_2)
print "assertion passed!"
  value was True.  !!!
"""

"""
  desired final result
ons = on_locs_nonzero(model)
print ons.shape
model_2=np.zeros(model.shape).astype('bool'); model_2[ons]=1
"""
def find_crotch(mask):
  '''
    As of Thu Jan 17 09:15:12 EST 2019,
      This func assumes 0 noise in the mask, which is an unrealistic assumption
  '''
  locs=np.nonzero(mask)
  toe_height=np.amax(locs[0])
  both_feet_height=float('inf')
  #both_feet_height=toe_height+30 # TODO: fiddle with this until it's right
  crotch_x_loc=float('inf')
  for height in range(toe_height,0,-1):
    if both_feet_height==float('inf'): # both_feet_height will change values once we find the pt where we see both legs separately
      in_1st=False
      looking_for_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st and mask[height,x]:
          in_1st=True
        elif in_1st and not mask[height,x]:
          looking_for_2nd=True
        elif looking_for_2nd and mask[height,x]:
          both_feet_height=height
          break # out of for loop
    else: # found height where both legs are present in one row
      in_1st=False
      looking_for_2nd=False
      found_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st        and     mask[height,x]:
          in_1st=True
        elif in_1st          and not mask[height,x]:
          looking_for_2nd=True
        elif looking_for_2nd and     mask[height,x]:
          crotch_x_loc=x-2
          found_2nd=True
      if looking_for_2nd and not found_2nd:
        crotch_height=height
        return (crotch_x_loc, crotch_height)
  raise CrotchNotFoundException
  return

#===================================================================================================================================
def rot8(model, angle):
  '''
    rotates counterclockwise (by default around the positive z-axis; if you were looking down at the model from the head, it would rotate counterclockwise)
      TODO:  make sure the direction is right
      TODO:  extend to rotations around other axes

    output: 
      rotated 3-D numpy array
    parameters:
      model is a 3-D numpy array
      angle in degrees (float: floating point)
    Notes:
      mode : not 100% sure what this parameter does, but here are:
        Notes from scipy.ndimage.rotate():
          The given matrix and offset are used to find for each point in the
          output the corresponding coordinates in the input by an affine
          transformation. The value of the input at those coordinates is
          determined by spline interpolation of the requested order. Points
          outside the boundaries of the input are filled according to the given mode.

          According to these notes, possible values of mode are:
            'constant'
            'nearest'
            'reflect'
            'wrap'

      reshape=False will cut off the corners when we rotate.  So we NEED to make sure the human body is centered within the numpy array before proceeding.  I used this so the dimensions don't get fucked up later
  '''
  xy=(1,0); return uint_mids(
                              scipy.ndimage.rotate(model, angle, axes=xy, reshape=False, mode='constant')
                              )
#=============================================  end func def of rot8(model, angle):  ===============================================
def uint_mids(arr):
  # NOTE:  do we want this to run on floats?  TODO:   try multiple ways (first on uint8, then float, etc.)
  UINT8_MAX=np.iinfo('uint8').max; MID=int(round(UINT8_MAX/2.))
  arr[np.greater(arr, 0)]=MID; return arr
#===================================================================================================================================
def binarize(mask_3_colors):
  RED=0; CERTAIN=256.0; probable = np.array(int(CERTAIN / 2)-1) # default color is magenta, but the red part shows 
  mask_binary = deepcopy(mask_3_colors[:,:,RED])
  return np.greater(mask_binary, probable).astype('bool')
#===========================================  end func def of binarize(mask_3_colors):  ============================================
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

#===================================================================================================================================
# This code is a half-formed thought.  Probably will never use it again.
def find_nip(mask):
  find_nip_by_tracing()
def find_nip_by_tracing():
  pass
def armpit(mask):
  '''
    Finds the customer's armpit given a segmented-out mask of their body
    Customer is in "Jesus pose" as of Mon Feb 25 13:34:31 EST 2019
  '''
  return armpit_by_tracing(mask)
def armpit_by_tracing(mask):
  # Finds the armpit by locating the extended arms of the customer and then going "down" the mask the proper corresponding amount.
  # "Jesus Pose" required
  # TODO: update the comment above this line after I've actually written the function.
  counts=np.count_nonzero(mask, axis=1)
  fingertips_y_idx=np.argmax(counts)
  CONST=10
  pltshow(mask)
  pltshow(mask[fingertips_y_idx-CONST:fingertips_y_idx+CONST])
  pltshow(mask)
  # TODO: paste this code into the function "chest_circum(json_fname, front_fname, side_fname)"
#===================================================================================================================================


  As of Sat Mar  2 07:39:17 EST 2019,

    47:  def __init__(self, tarball_path):
    63:  def run(self, image):
    75:  def segment_nparr(self, img):
    98:def create_pascal_label_colormap():
    109:def label_to_color_image(label):
    119:def binarize(mask_3_colors):
    125:def run_visualization(url, model):
    144:def seg_map(img, model):
    157:def segment_local(local_filename):
    188:def segment_URL(IMG_URL):
    224:def segment_black_background(local_fname):

