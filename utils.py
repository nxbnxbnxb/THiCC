
# imports useful for access in python shell rather than for use in utils.py
#import imageio as ii
#import pandas as pd
#from   mpl_toolkits.mplot3d import Axes3D   # import no longer used (Dec. 16, 2018).  plots VERY BASIC 3d shapes
import numpy as np
np.seterr(all='raise')

import pickle as pkl

import datetime
import sys
import os
import math
from   math import sin, cos, tan, pi, radians, degrees, floor, ceil, sqrt
from   collections import OrderedDict
from   pprint import pprint as p

from   os import listdir as ls
from   time import sleep
import subprocess as sp
import random
from   copy import deepcopy

from   d import debug

# TODO:   vim "repeat any"
# TODO:   vim command "undo any"
# TODO:   auto-updating ls, l, pwd, etc. commands (variable gets updated every 10 seconds within python shell and automatically pretty-prints)

# TODO:   refactor: move all pltshow(), show_3d(), etc. type functions into utils.py or something like utils.py and rename it
# TODO:   def pif(txt): if global_debug_boolean: print(txt)

#######################################################################################################
#################################    utility functions    #############################################
#######################################################################################################


#=========================================================================
def pif(s=''):
    if debug: print (s)
#=========================================================================
def sq(x):
    return x*x
#=========================================================================
def blenderify(nparr):
    '''
        A more descriptive name would be "def make_list_of_tuples_from_nparr(arr):"
        3-tuples contain (x, y, z) locations of the nonzero elements in arr
        precondition: len(arr.shape)==3
    '''
    li=[]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if(arr[i][j][k]):
                    li.append((i,j,k))
    return li
#=========================================================================
def approx_eq(x,y):
    '''
        approx_eq() is only for np arrays
        curr just 2-d arrs, but should extend to 3d later
      already exists [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html)
    '''
    if not x.shape == y.shape:
      print ("the shapes are not the same:\n first_param.shape is {0} \n second_param.shape is {1} \n".format(x,y))
      return False
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        e=x[i,j]; tolerance=abs(e)/20
        if abs(y[i,j]-x[i,j]) > tolerance:
          return False
    return True
#=========================================================================
def eq(x,y):
    '''
        eq() is only for np arrays
    '''
    return np.array_equal(x,y)
#=========================================================================
def sort_fnames_list(fnames_list):
    '''
        assumes filenames fnames are all precisely of format:
          0.0.png
          45.0.png
          90.0.png
          ...
          315.0.png
        Terminating decimals are okay, but repeating (ie. 3.33333333) are not supported
    '''
    trimmed = []
    for fname in fnames_list:
        trimmed.append(float(fname.replace('.png', '')))  # TODO: extend to image filetypes other than png.  Best way TODO this is by reverse-searching for the last decimal pt and trimming that way instead
    _sorted = sorted(trimmed)
    final = []
    for num in _sorted:
        final.append(str(num)+'.png')
    return final
#=========================================================================


#==============================================================
def prepend_0s(int_str, num_digits=9):
  '''
    Params:
    -------
    int_str is a string

  '''
  #NOTE: I know there's a less "hack-y" way of putting zeros in front of a number.  But I don't wanna look it up when I can just rewrite the code myself.
  return '0'*(num_digits-len(int_str))+int_str
#==============================================================

#================================================================
def save_mp4_as_imgs(mp4_local_path, root_img_dir, freq=1/4., should_put_timestamps=True, output_img_filetype='jpg'):
  # TODO: finish this function, test the 1st part of it
  # TODO: generalize this to multiple vid filetypes, not just mp4
  '''
    Mutates the local file directory with new image files

    Sources:
      https://stackoverflow.com/questions/25182278/opencv-python-video-playback-how-to-set-the-right-delay-for-cv2-waitkey
      https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
  '''
  import cv2
  delay=int(round(1/freq))  # why the **** does "delay" have to be an int?
  img_write_dir=root_img_dir
  if not root_img_dir.endswith('/'):
    img_write_dir+='/'
  if should_put_timestamps:
    timestamp=datetime.datetime.now().strftime('%Y_%m_%d____%H:%M_%p__') # p is for P.M. vs. A.M.
    img_write_dir+=timestamp+'/'
  os.system('mkdir '+img_write_dir) # NOTE: if directory is already there, execution will continue

  print(mp4_local_path)
  vidcap = cv2.VideoCapture(mp4_local_path)
  success = True
  count = 0
  while success:
    success, img = vidcap.read()
    if count < 1:
      if debug:
        print('We just read a new frame, true or false? ', success)
    if cv2.waitKey(delay) &  0xFF == ord('q'):
      break
    # NOTE:  "I put the imwrite() AFTER the break so if the image fails to read, we don't save an empty .jpg file."   - Nathan (Mon Jan 14 13:33:26 EST 2019)
    cv2.imwrite(img_write_dir+"{0}.{1}".format(prepend_0s(str(count)),output_img_filetype), img) # TODO: there's probably a better way to do the prepend_0s() function
    count += 1
  return 0 # success
# NOTE: in order to get the masks at the ideal angles of the body's rotation, we gotta come up with some smart way of calculating the angles.  Maybe counting the total number of img files between 0 and 360 degrees and just dividing?  It'll probably do for now, but unfortunately stepping in a circle is not like a smooth lazy-susan
#===== end func def of  save_mp4_as_imgs(**lotsa_params): =====



















 






































def newline(f):
    f.write('\n')


def hist():
    '''
        print python history
        TODO:  hg() == UNIX hg
    '''
    import readline
    print ('\n'*2)
    for i in range(readline.get_current_history_length()):
        print (readline.get_history_item(i + 1))
    print ('\n'*2)
h=hist

def print_dict(d):
    print_dict_recurs(d, 0)

def print_dict_recurs(d, indent_lvl):
    for k,v in d.items():
        print (('  ')*indent_lvl+'within key '+str(k)+': ')
        if type(v)==type({}) or type(v)==type(OrderedDict()):
            print_dict_recurs(v, indent_lvl+1)
        elif type(v)==type([]):
            print_list_recurs(v, indent_lvl+1)
        else:
            print (('  ')*indent_lvl+'  value in dict: '+str(v))

def print_list(l):
    print_list_recurs(l, 0)
def print_list_recurs(l, indent_lvl):
    print (('  ')*indent_lvl+'printing list')
    for e in l:
        if type(e)==type({}) or type(e)==type(OrderedDict()):
            print_dict_recurs(e, indent_lvl+1)
        elif type(e)==type([]):
            print_list_recurs(e, indent_lvl+1)
        else:
            print (('  ')*indent_lvl+'  element in list: '+str(e))
 
def print_visible(s):
    '''
            Prints like the following:
    >>> print_visible(s)




        ("pad" # of newlines)


===============================================
            inputted string s here
===============================================


        ("pad" # of newlines)

    '''
    s = str(s)
    pad = 21
    num_eq = 3*len(s)
    print(pad*"\n"           +\
            (num_eq*"="+"\n")+\
            len(s)*" "+s+"\n"+\
            (num_eq*"=")     +\
            (pad*"\n"))

def count(arr):
  counts={}
  if len(arr.shape)==3:
    for subarr_1 in arr:
      for subarr_2 in subarr_1:
        for val in subarr_2:
          if val in counts:
            counts[val]+=1
          else:
            counts[val]=1
  if len(arr.shape)==2:
    for subarr_1 in arr:
      for val in subarr_1:
        if val in counts:
          counts[val]+=1
        else:
          counts[val]=1
  return counts


# NOTE:   the most basic example of exception inheritance
class MeanHasNoPtsException(RuntimeError):
    pass
#=========================================================================
def pad_all(mask, biggers_shape):
    '''
        pads 2d mask with zeros on the end til mask.shape==biggers_shape
    '''
    # TODO:  debug dis bish.  we get weird cutoffs in the middle of shit
  #=========================================================================
    def pad_top_bot(mask, biggers_shape):
        '''
            puts zeros on top and bottom of mask until mask.shape[1] == biggers_shape[0]
        '''
        padded         = np.copy(mask)

        big_h          = biggers_shape[0]
        mask_h         = mask.shape[0]
     
        top_h          = (big_h - mask_h) / 2.0
        if top_h.is_integer():
            bot_h      = int(top_h)
        else:
            bot_h      = int(floor(top_h + 1))
        top_h          = int(floor(top_h))
        # convert to int if not already 
        pad_w          = padded.shape[1]
        top            = np.zeros((top_h, pad_w), dtype=int)
        bottom         = np.zeros((bot_h, pad_w), dtype=int)
        padded         = np.concatenate((top, padded, bottom), axis = 0)

        return padded
    # end func def pad_top_bot(mask, biggers_shape):
  #=========================================================================
    def pad_sides(mask, biggers_shape):
        '''
            puts zeros on sides of mask until mask.shape[1] == biggers_shape[0]
        '''
        padded         = np.copy(mask)

        big_w          = biggers_shape[1]
        mask_w         = mask.shape[1]
        left_w         = (big_w - mask_w) / 2.0 
        # NOTE: single slash is integer division in python2 unless dividing by 2.0
        #       This is NOT true in python3, which distinguishes '/' from '//'
        if left_w.is_integer():
            right_w    = int(left_w)
        else:
            right_w    = int(floor(left_w + 1))
        left_w         = int(floor(left_w))
        pad_h          = padded.shape[0]
        left           = np.zeros((pad_h, left_w), dtype=int)
        right          = np.zeros((pad_h, right_w), dtype=int)
        padded         = np.concatenate((left, padded, right), axis = 1)

        return padded
    # end func def pad_sides(mask, biggers_shape):
  #=========================================================================
    return pad_top_bot(
                pad_sides(
                    mask,
                    biggers_shape),
                biggers_shape)
  # end func def pad_sides(mask, biggers_shape):
#=========================================================================
                


if __name__=='__main__':
    pif(('*'*99)+'\n debug is on \n'+('*'*99))
    raise MeanHasNoPtsException(" hi i'm an exception msg")








































































