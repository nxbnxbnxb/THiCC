WHITE=255
BLACK=0
TRANSPARENT=0

import os, tarfile, tempfile
import numpy as np
np.seterr(all='raise')
import tensorflow as tf
import scipy
from scipy.ndimage.measurements import center_of_mass as CoM
from io import BytesIO
import imageio as ii
import skimage
from PIL import Image

import sys

from d import debug
from save import save
from viz import pltshow
from utils import np_img

if debug:
  from matplotlib import pyplot as plt  # NOTE:  problem on Nathan's machine (Dec. 21, 2018) is JUST with pyplot.  None of the rest of matplotlib is a problem AT ALL.
from matplotlib import gridspec
from six.moves import urllib

from copy import deepcopy

'''
import matplotlib
matplotlib.use('Agg')      
'''
# The above lines didn't end up being necessary.    But it's a good FYI so future programmers can solve matplotlib display problems, though

class DeepLabModel(object):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.graph = tf.Graph()
    graph_def = None
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.Session(graph=self.graph)

#================================================================
  def run(self, image):
    if debug:
      print("image.size:")
      print(image.size);print('\n'*3)
      print("type(image)")
      print(type(image));print('\n'*3) # image is a    <class 'PIL.PngImagePlugin.PngImageFile'>
    width, height = image.size
    resize_ratio = float(self.INPUT_SIZE / max(width, height))
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    if debug:
      pltshow(np.asarray(resized_image))
      print("np.asarray(resized_image).shape:")
      print(np.asarray(resized_image).shape);print('\n'*3)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map
#================================================================
  def segment_nparr(self, img):
    '''
      README
    '''
    # TODO: fill out docstring above once this is finalized
    #  NOTE NOTE NOTE:   the below "NOTE"s are no longer relevant; I realized why Vishal included them; it was to resize the image properly.  Nonetheless, there is some useful info contained within those 2 comments.
    # NOTE:  this really oughta be just a one-liner; probably no need for this function to have a name
    # style  NOTE:   I don't really like encapsulation; it can make it harder to debug shit.  A programmer on this project ought to be smart enough that they can handle a few levels of nesting with a comment to explain shit?  Open to hearing other opinions, but that's mine. ----- NXB (Nathan Xin-Yu Bendich): Mon Jan 14 19:57:16 EST 2019
    width, height, RGB = img.shape
    resize_ratio = float(self.INPUT_SIZE / max(width, height))
    target_size = (int(resize_ratio * width), int(resize_ratio * height),3)
    resized_image = skimage.transform.resize(img,target_size, anti_aliasing=True)
    return (self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [resized_image]})[0], 
      resized_image)
  #===== end func def of    segment_nparr(self, img): =====
  #================================================================
# end class definition DeepLabModel()
#================================================================


#================================================================
def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  colormap = create_pascal_label_colormap()
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')
  return colormap[label]

# NOTE: don't call this function if matplotlib.pyplot crashes conda!
def vis_segmentation(image, seg_map):
  '''
    CURRENTLY NOT WORKING
  '''
  # NOTE:  CURRENTLY NOT WORKING.  To debug, please look backwards at prev version in git
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

#================================================================
def binarize(mask_3_colors):
  RED=0; CERTAIN=256.0; probable = np.array(int(CERTAIN / 2)-1) # default color is magenta, but the red part shows 
  mask_binary = deepcopy(mask_3_colors[:,:,RED])
  return mask_binary.astype('bool')
# end def binarize(mask_3_colors):
#================================================================

def run_visualization(url, model):
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return
  if debug:
    print('running deeplab on image %s...' % url)
  resized_im, seg_map = model.run(original_im)
  if save:
    ii.imwrite("_segmented____binary_mask_.jpg", seg_map)  # Dec. 14, 2018:  I think something in the saving process ***ks up the mask with noise
  #PROBABLE = 127  # NOTE:  experimental from 1 data point;  PLEASE DOUBLE CHECK if u get a noisy segmentation
  #ii.imwrite("_segmented____binary_mask_.jpg", np.greater(seg_map, PROBABLE).astype('bool'))
  return np.rot90(seg_map,k=3) # k=3 b/c it gives us the result we want   (I tested it experimentally.  Dec. 26, 2018)   # this is true for the URL version, not the other
#===== end func def of  run_visualization(url): =====
#================================================================
def seg_map(img, model):
  print('running deeplab on image')
  seg_map, resized_im = model.segment_nparr(img)
  if debug:
    pltshow(seg_map)
  if save:
    fname = "_segmented____binary_mask_.jpg"
    print('saving segmentation map in ', fname)
    ii.imwrite(fname, seg_map)  # Dec. 14, 2018:  I think something in the saving process ***ks up the mask with noise
  return np.rot90(seg_map,k=3) # k=3 b/c it gives us the result we want   (I tested it experimentally.  Dec. 26, 2018)
#=========== end func def of  seg_map(img, model): ==============

#================================================================
def segment_local(local_filename):
  #img=scipy.ndimage.io.imread(local_filename)
  img=np.asarray(ii.imread(local_filename)).astype('float64') # TODO: delete this commented-out line
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
  MODEL = DeepLabModel(download_path)
  FAIRLY_CERTAIN=127
  return seg_map(img, MODEL)
#=====  end func def of   segment_local(local_filename) =====

#==================================================
def segment_URL(IMG_URL):
  '''
    NOTE: segmentation requires internet connection
  '''
  # TODO:   allow us to set the URL from parameter
  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])

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

  MODEL = DeepLabModel(download_path)

  FAIRLY_CERTAIN=127
  return np.greater(run_visualization(IMG_URL, MODEL), FAIRLY_CERTAIN)
#===== end func def of  segment_URL(IMG_URL): =====

#====================================================================
def overlay_imgs(img_fname_1, img_fname_2):
    cutout1,mask1=segment_black_background(img_fname_1)
    cutout2,mask2=segment_black_background(img_fname_2)
    CoM1=CoM(mask1)
    CoM2=CoM(mask2)

    print("img_fname_1:\n",img_fname_1)
    pltshow(cutout1)
    print("img_fname_2:\n",img_fname_2)
    pltshow(cutout2)
    print("CoM1:\n",CoM1)
    print("CoM2:\n",CoM2)
    # TODO: finish this overlay_imgs() function!
    pass 























#===== end func def of  overlay_imgs(img_fname_1, img_fname_2): =====




















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

  pltshow(img)
  pltshow(segmap)
  # as of (Wed Feb 20 17:49:37 EST 2019), segmap is 0 for background, 15 for human ( before astype('bool'))
  segmap=segmap.astype('bool')
  segmap=np.logical_not(segmap)
  img[segmap]=WHITE
  # cut out the human from the img
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

#====================================================================
def segment_black_background(local_fname):
  '''
    BLACK is 0; so this function can be used to "add" two images together to superimpose them.
  '''
  # NOTE:  PIL.resize(shape)'s shape has the width and height in the opposite order from numpy's height and width
  # NOTE: weird sh!t happened when I tried to convert to 'float64' in the `np.array(Image.open(fname))` line.
  segmap  = segment_local(local_fname)
  segmap  = segmap.reshape(segmap.shape[0],segmap.shape[1],1)
  segmap  = np.rot90(       np.concatenate((segmap,segmap,segmap),axis=2)        )
  # I really OUGHT to scale the mask to fit the dimensions of the image (we'd have better resolution this way)
  img     = Image.open(local_fname)
  img     = img.resize((segmap.shape[1],segmap.shape[0]), Image.ANTIALIAS)
  img     = np.array(img) 
  if debug:
    pltshow(img); pltshow(segmap)
  # as of (Wed Feb 20 17:49:37 EST 2019), segmap is 0 for background, 15 for human ( before astype('bool'))
  segmap=np.logical_not(segmap.astype('bool'))

  # cut out the human from the img
  img[segmap]=BLACK
  pltshow(img)
  fname='person_cutout__black_background.png'
  ii.imwrite(fname,img)
  return img, np.logical_not(segmap) # mask with 1s where there's a person, not where there's background
#========== end func def of  segment_black_background(params): ==========

if __name__=='__main__':
  if len(sys.argv) == 1:
    IMG_URL = 'http://columbia.edu/~nxb2101/180.0.png'
    #'http://vishalanand.net/green.jpg'
    print ("\nusage: python2 seg.py [url_of_img_containing_human(s)] \n  example: python2 seg.py http://vishalanand.net/green.jpg   \n\n")
    print ("currently segmenting image found at url: \n  "+IMG_URL)
  else:
    img_path=sys.argv[1]
    #IMG_URL = sys.argv[1]# TODO: uncomment to segment images on the internet.
  print('\n'*2)

  fnames=sys.argv[1:] # TODO: change the analogous code in render_smpl.py
  # TODO: if we extend seg.py (processing more cmd line args and more configuration parameters) the line `fnames=[sys.argv[1:]]` will break.
  #   A better way to do it is with `import absl; absl.configs`
  #     (see akanazawa's HMR for example of these configs.)

  # TODO: error checking:
  #   if len(sys.argv) < 3:   # should it be less than 3?  more?
  overlay_imgs(fnames[0], fnames[1])  # tODO: hardcode the fnames?
























# end if __name__=='__main__':







































  '''
  #segmap = segment_URL(IMG_URL) # TODO: uncomment to segment images on the internet.
  segment_transparent_background(img_path)
  print('now doing black')
  segment_black_background(img_path)
  # TODO: more thorough tests.  For now, I'm going to move on b/c I wanna sleep and there are more important things.  But if you think of any bugs, please fix ASAP rather than 3 months from now.
  '''






















