
# imports useful for access in python shell rather than for use in utils.py
import matplotlib.pyplot as plt
import imageio as ii
from   mpl_toolkits.mplot3d import Axes3D
from   pylab import savefig
import numpy as np
import pandas as pd

import pickle as pkl

import sys
import os
import math
from   math import sin, cos, tan, pi, radians, degrees, floor, ceil, sqrt
import cv2
from   collections import OrderedDict
from   pprint import pprint as p

from   os import listdir as ls
from   time import sleep
import subprocess as sp
import random
from   copy import deepcopy

import scipy.misc
import scipy.ndimage
from   scipy.ndimage import rotate

from   d import debug
from   view_3d_model import show_cross_sections

# TODO:   vim "repeat any"
# TODO:   vim command "undo any"
# TODO:   auto-updating ls, l, pwd, etc. commands (variable gets updated every 10 seconds within python shell and automatically pretty-prints)

# TODO:   refactor: move all pltshow(), show_3d(), etc. type functions into utils.py or something like utils.py and rename it
# TODO:   def pif(txt): if global_debug_boolean: print(txt)

#######################################################################################################
#################################    utility functions    #############################################
#######################################################################################################


#=========================================================================
def pif(s):
    if debug: print s
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
def eq(x,y):
    '''
        eq() is only for np arrays
    '''
    return np.all(np.equal(x,y))
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



























def pltshow(im):
    plt.imshow(im); plt.show(); plt.close()

def newline(f):
    f.write('\n')

def h():
    hist()

def hist():
    '''
        print python history
        TODO:  hg() == UNIX hg
    '''
    import readline
    for i in range(readline.get_current_history_length()):
        print (readline.get_history_item(i + 1))

def print_dict(d):
    print_dict_recurs(d, 0)

def print_dict_recurs(d, indent_lvl):
    for k,v in d.items():
        print ('  ')*indent_lvl+'within key '+str(k)+': '
        if type(v)==type({}) or type(v)==type(OrderedDict()):
            print_dict_recurs(v, indent_lvl+1)
        elif type(v)==type([]):
            print_list_recurs(v, indent_lvl+1)
        else:
            print ('  ')*indent_lvl+'  value in dict: '+str(v)

def print_list(l):
    print_list_recurs(l, 0)
def print_list_recurs(l, indent_lvl):
    print ('  ')*indent_lvl+'printing list'
    for e in l:
        if type(e)==type({}) or type(e)==type(OrderedDict()):
            print_dict_recurs(e, indent_lvl+1)
        elif type(e)==type([]):
            print_list_recurs(e, indent_lvl+1)
        else:
            print ('  ')*indent_lvl+'  element in list: '+str(e)
 
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



# NOTE:   the most basic example of exception inheritance
class MeanHasNoPtsException(RuntimeError):
    pass


if __name__=='__main__':
    pif(('*'*99)+'\n debug is on \n'+('*'*99))
    raise MeanHasNoPtsException(" hi i'm an exception msg")



