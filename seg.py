import os, tarfile, tempfile
import numpy as np
np.seterr(all='raise')
import tensorflow as tf
import scipy
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


if __name__=='__main__':
  # TODO:  put this code in a separate function
  if len(sys.argv) == 1:
    IMG_URL = 'http://columbia.edu/~nxb2101/180.0.png'
    #'http://vishalanand.net/green.jpg'
    print ("\nusage: python2 seg.py [url_of_img_containing_human(s)] \n  example: python2 seg.py http://vishalanand.net/green.jpg   \n\n")
    print ("currently segmenting image found at url: \n  "+IMG_URL)
  else:
    img_path=sys.argv[1]
    #IMG_URL = sys.argv[1]# TODO: uncomment to segment images on the internet.
  #segmap = segment_URL(IMG_URL) # TODO: uncomment to segment images on the internet.
  segmap= segment_local(img_path)
  print('segmap.shape: {0}'.format(segmap.shape))
  print('\n'*2)
  img=Image.open(img_path)
  #img=np_img(img_path) # I really OUGHT to scale the mask to fit the dimensions of the image (we'd have better resolution this way)
  #print("img.shape : \n",img.shape)
  segmap= segmap.reshape(segmap.shape[0],segmap.shape[1],1)
  segmap=np.concatenate((segmap,segmap,segmap),axis=2)
  segmap=np.rot90(segmap)
  # 1920, 1080
  #  513,  288
  # NOTE: for NOW, I've hardcoded this;     TODO: change as soon as we're looking at a diff img.
  #targ_shape=(640,360,3)
  #pltshow(img.resize((segmap.shape[0],segmap.shape[1]), Image.ANTIALIAS))
  img=img.resize((segmap.shape[1],segmap.shape[0]), Image.ANTIALIAS) #targ_shape, Image.ANTIALIAS)
  print("type(img):\n",type(img))
  # NOTE:  PIL.resize(shape)'s shape has the width and height reversed from numpy's
  #segmap = utils.pad_color(segmap, targ_shape)
  #img = np.asarray(img).astype('float64') np.asarray doesn't work on  type <class 'PIL.Image.Image'>
  img = np.array(img)
  pltshow(img)
  print("img.dtype:\n",img.dtype)
  print("   img.shape:\n", img.shape)
  print("segmap.shape:\n", segmap.shape)
  # as of (Wed Feb 20 17:49:37 EST 2019), segmap is 0 for background, 15 for human ( before astype('bool'))
  pltshow(segmap)
  segmap=segmap.astype('bool')
  segmap=np.logical_not(segmap)
  img[segmap]=0
  # cut out the human from the img
  pltshow(img)
  ii.imwrite('nathan_cutout.png',img)
  # how 2 make background transparent?
 
# end if __name__=='__main__':


