from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#TODO? rename to mzr?
import json
import numpy as np
np.seterr(all='raise')
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import ConvexHull
import math
from math import pi
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import imageio as ii
import os
import sys
#import sympy as s

import viz
from viz import pltshow
from seg import segment_local as seg_local, segment_black_background as seg_black_back
from utils import pn, crop_person, get_mask_y_shift, np_img, pif, pe
from d import debug
from pprint import pprint as p
from copy import deepcopy

# TODO: make the earlier JSON-generation via openpose automated end-to-end like this.  Everything must be MODULAR, though




# Measurement data:
'''
  435.225,676.318,  RShoulder
  650.009,660.497,  LShoulder
  498.099,1011.46,  RHip        
  623.893,1001,     LHip        
  "__main__" spit out THIS answer:
    chest_area: -29648.446561999983

  the code given by @Walt W from [this](https://stackoverflow.com/questions/1329546/whats-a-good-algorithm-for-calculating-the-area-of-a-quadrilateral) link is here, and gives this answer:
  area = 0.5 * abs( x0 * y1 - x1 * y0 + x1 * y2 - x2 * y1 + 
                    x2 * y3 - x3 * y2 + x3 * y0 - x0 * y3 )
    chest_area:
      15080.677047
  But the real answer really oughta be something like 60,000; maybe somewhere around 45,000 in my estimation.  It's most definitely above 30,000.  30,000 would be the area for the rectangle ((500,1000),(500,700),
                                                         (600,1000),(600,700)).
  February 1, 2019:
    I got 57768.96495200002

  [This](https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon) answer for chest area was:
    15080.677046999946

'''

































































pr=print  
# NOTE: in my infinite quest for conciseness, I may have turned some other pprint() function into ppr(), which will break the code

#===================================================================================================================================
def load_json(json_fname):
  with open(json_fname) as json_data:
    data = json.load(json_data)
    return data
#===================================================================================================================================
def measure(json_fname):
  return parse_ppl_measures(load_json(json_fname))
#===================================================================================================================================
def parse_ppl_measures(json_dict):
  '''
    NOTE: Extensible to whatever measurements we want later
  '''
  # reference: /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/IMPORTANT_REF/openpose__json_order___format.txt

  # for sample file, see /home/ubuntu/Documents/code/openpose/output/front__nude__grassy_background_keypoints.json
  measures                  = json_dict[u'people'][0][u'pose_keypoints_2d']
  measures_dict             = {}
  measures_dict["Nose"]     = {'x':measures[0*3],   'y':measures[0*3+1],       'c':measures[0*3+2]}
  measures_dict["Neck"]     = {'x':measures[1*3],   'y':measures[1*3+1],       'c':measures[1*3+2]}
  measures_dict["RShoulder"]= {'x':measures[2*3],   'y':measures[2*3+1],       'c':measures[2*3+2]}
  measures_dict["RElbow"]   = {'x':measures[3*3],   'y':measures[3*3+1],       'c':measures[3*3+2]}
  measures_dict["RWrist"]   = {'x':measures[4*3],   'y':measures[4*3+1],       'c':measures[4*3+2]}
  measures_dict["LShoulder"]= {'x':measures[5*3],   'y':measures[5*3+1],       'c':measures[5*3+2]}
  measures_dict["LElbow"]   = {'x':measures[6*3],   'y':measures[6*3+1],       'c':measures[6*3+2]}
  measures_dict["LWrist"]   = {'x':measures[7*3],   'y':measures[7*3+1],       'c':measures[7*3+2]}
  measures_dict["MidHip"]   = {'x':measures[8*3],   'y':measures[8*3+1],       'c':measures[8*3+2]}
  measures_dict["RHip"]     = {'x':measures[9*3],   'y':measures[9*3+1],       'c':measures[9*3+2]} 
  measures_dict["RKnee"]    = {'x':measures[10*3],  'y':measures[10*3+1],      'c':measures[10*3+2]}
  measures_dict["RAnkle"]   = {'x':measures[11*3],  'y':measures[11*3+1],      'c':measures[11*3+2]}
  measures_dict["LHip"]     = {'x':measures[12*3],  'y':measures[12*3+1],      'c':measures[12*3+2]}
  measures_dict["LKnee"]    = {'x':measures[13*3],  'y':measures[13*3+1],      'c':measures[13*3+2]}
  measures_dict["LAnkle"]   = {'x':measures[14*3],  'y':measures[14*3+1],      'c':measures[14*3+2]}
  measures_dict["REye"]     = {'x':measures[15*3],  'y':measures[15*3+1],      'c':measures[15*3+2]}
  measures_dict["LEye"]     = {'x':measures[16*3],  'y':measures[16*3+1],      'c':measures[16*3+2]}
  measures_dict["REar"]     = {'x':measures[17*3],  'y':measures[17*3+1],      'c':measures[17*3+2]}
  measures_dict["LEar"]     = {'x':measures[18*3],  'y':measures[18*3+1],      'c':measures[18*3+2]}
  measures_dict["LBigToe"]  = {'x':measures[19*3],  'y':measures[19*3+1],      'c':measures[19*3+2]}
  measures_dict["LSmallToe"]= {'x':measures[20*3],  'y':measures[20*3+1],      'c':measures[20*3+2]}
  measures_dict["LHeel"]    = {'x':measures[21*3],  'y':measures[21*3+1],      'c':measures[21*3+2]}
  measures_dict["RBigToe"]  = {'x':measures[22*3],  'y':measures[22*3+1],      'c':measures[22*3+2]}
  measures_dict["RSmallToe"]= {'x':measures[23*3],  'y':measures[23*3+1],      'c':measures[23*3+2]}
  measures_dict["RHeel"]    = {'x':measures[24*3],  'y':measures[24*3+1],      'c':measures[24*3+2]}
  return measures_dict # NOTE: you gotta use the version of openpose I ran if you just return pose_keypoints_2d.
#===================================================================================================================================
def segments(polygon):
  '''
    Just a subroutine of area_polygon().  For some reason I was having trouble nesting segments()'s function definition within def area_polygon(polygon):
  '''
  return zip(polygon, polygon[1:] + [polygon[0]])
#===================================================================================================================================
def area_polygon(polygon):
	'''
    Calculates the polygon's area.  Also works for concave polygons

    Parameters:
    ----------
    Given a polygon represented as a list of (x,y) vertex coordinates, implicitly wrapping around from the last vertex to the first




    ----------
 
  More documentation from the StackOverflow question [here](https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon):
		Here is the standard method, AFAIK. Basically sum the cross products around each vertex. Much simpler than triangulation.

Python code, given a polygon represented as a list of (x,y) vertex coordinates, implicitly wrapping around from the last vertex to the first
    David Lehavi comments: It is worth mentioning why this algorithm works: It is an application of Green's theorem for the functions −y and x; exactly in the way a planimeter works. More specifically:

    Formula above =
    integral_over_perimeter(-y dx + x dy) =
    integral_over_area((-(-dy)/dy+dx/dx) dy dx) =
    2 Area



    -----------
    Green's theorem if from "Calculus IV," as it's called at Columbia University.  (Calc IV means integrals generalized to higher dimensions)
    -----------
	'''
	return 0.5 * abs(sum(x0*y1 - x1*y0
											 for ((x0, y0), (x1, y1)) in segments(polygon))) # cross product
"""
#===================================================================================================================================
def estim8_w8(json_fname, front_fname, side_fname, height):
  # NOTE: probably they can supply us height and weight in the beginning.
  '''
    json from openpose
    mask from seg.py
  '''
  # NOTE: running openpose is ALWAYS dirt slow.  If you want a bunch of openpose jsons, make a shell script and just run it in the background somehow.
  # BIG picture function; we should build from ground-up too.  To estimate weight, we prob need:
    #  height
    #  rough volume of mask
  measurements = get_keypoints(json_fname)



  Height
  Weight
  Chest
  Waist
  Hips
  Inseam
  Exercise
"""
 


#===================================================================================================================================
def measure_body_viz(json_fname, front_fname, side_fname, cust_height):
# used to be called "chest_waist_hips_circum(json_fname, front_fname, side_fname, cust_height)"
  '''
    Circumference (perimeter) of body at the nipple level
    This method of finding the chest circumference assumes the customer's arms are perpendicular to the person's height (I call this "jesus pose,").  They don't NEED to be in this pose, but they definitely cannot be relaxed at the person's sides or "below" parallel to the ground.

    -------
    Params:
    -------
    json_fname:  keypoints from openpose
    front_fname: person in color from the front (looking them in the face)
    side_fname:  person in color from the side (I think this is called "profile view")

    ------
    Notes:
    ------
    This method of finding the chest circumference assumes the customer's arms are perpendicular to the person's height (I call this "jesus pose,").  They don't NEED to be in this pose, but they definitely cannot be relaxed at the person's sides or "below" parallel to the ground.
    Hardcoded 1./3. won't work.  It's better to trace the edge of the mask with a method like seen in old_measure.py's "find_toes(), find_crotch(), etc."
    Better still would be to train a separate neural network to identify nipples, but this would take too much time for the MVP/prototype.

    -------------
    Improvements:
    -------------
    Everyone's torso is a tad different.  This hard-coded ratio that finds the nipples won't always work.
  '''

  # TODO:
  '''
    0.  Clean up this function ("chest_waist_hips_circum()")
    2.  Find armpits like I wrote the function find_crotch() to do in "old_measure.py"
      a. how?
    3.  Standardize the rotation pose for all get_measurements() code. (ie. all Jesus pose)
    5.  Test this function
    6.  F
    7.
    8.
  '''
  # ideas:  How to find armpit
  '''
    1.  Maximize dist from bottom left corner
      a.
    2.  Maximize dist from bottom right corner (for other armpit)
      a.
    3.  1 and 2 while also
      a.  Trace the arm until you hit the body
      b.  Trace the leg until you hit the arm
      c.
      d.
    4.
  '''
  # We're within func chest_waist_hips_circum()
  measurements  = measure(json_fname)

  # Because of how openpose sets up 'y', Hip['y'] is larger than LShoulder['y'], 
    # even though the hip is below the shoulder in real life.
  shoulder_h    = measurements['Neck']['y']
  hip_h         = measurements['MidHip']['y']
  torso_len     = hip_h - shoulder_h
  # nipple height is where we measure the chest, according to www.bodyvisualizer.com ("chest")

  NIP_IN_TORSO  = 0.31460 #0.31460 is approximately 28/89, derived empirically from ONE image of myself 
  BELLY_IN_TORSO= 0.8175225846406439
  # empirically derived, albeit from an off-orthogonal photo of myself (Shaina was holding the camera at like a 85 degree angle)

  # previous values: 1./3., 2./5.
  nip_h         = shoulder_h + (torso_len*NIP_IN_TORSO) 
  belly_button_h= shoulder_h + (torso_len*BELLY_IN_TORSO) # belbut
  # I'm going to assume everyone wears pants way up near the belly button.  Then they won't need to wear a belt.
  # It's probably easiest if we standardize this such that everyone wears their pants at the same height.
  orig_imgs_nip_h=int(round(nip_h))
  front_mask    = np.rot90(seg_local(front_fname)) # shape == (513, 288)
  side_mask     = np.rot90(seg_local( side_fname)) # shape == (513, 288)
  pn(3) # b/c seg_local prints some shit to stdout
  pix_height    = pix_h(front_mask)
  #origs_height  = orig_pix_h(front_fname) # not necessary
  orig_h        = np.asarray(ii.imread(front_fname)).shape[0]
  masks_h       = front_mask.shape[0]
  if debug:
    pr("masks_h: ",masks_h)
    pr("orig_h:  ",orig_h)

  # Change heights (ie. nip_h, hip_h) to be within the mask's "height scale" :
  # mask_scaling is here because deeplab spits out segmaps of shape (513,288)
  mask_scaling  = float(masks_h)/float(orig_h)
  nip_h        *= mask_scaling
  nip_h         = int(round(nip_h))
  belly_button_h*= mask_scaling # TODO: separate scale variable for this (  "float(masks_h)/float(orig_h)" )
  belly_button_h = int(round(belly_button_h))
  hip_h        *= mask_scaling
  hip_h         = int(round(hip_h))
  if debug:
    pr("belly_button_h is \n",belly_button_h)
    pr("nip_h is \n",nip_h)
    pr("hip_h is \n",hip_h)

  # Data:
  #   For these particular images, the side view is 7 units shifted "up" from the   front view
  #   NOTE: we ought to identify a rotation point where from the side view the arms are directly at the customer's sides.  We also need to tell the customer exactly how to put their arms to enable easy measurement (ideally straight out; no angles)

  # NOTE:  picture/video should be taken  such that no part of the customers' arms are at the same height ("y value") as the customer's nipples.  "Jesus pose" or "Schwarzenegger pose"
  chest_w = np.count_nonzero(front_mask[nip_h])
  waist_w = np.count_nonzero(front_mask[belly_button_h])
  hip_w   = np.count_nonzero(front_mask[hip_h])

  if debug:
    pltshow(front_mask)
    front_mask[nip_h-1:nip_h+1]=0
    front_mask[belly_button_h-1:belly_button_h+1]=0
    front_mask[hip_h-1:hip_h+1]=0
    pltshow(front_mask)
    pr(front_mask.shape)
    pr(front_mask[np.nonzero(front_mask)]) # 15s
    pr(front_mask.dtype) # int64
    pr(side_mask.shape)
    pr(side_mask.dtype)  # int64
    pltshow(side_mask)
  # People shift up-down when rotating themselves for the camera.   
  # We have to identify the heights of body parts in both views so we can estimate the waist circumference, hip circumference, etc.
  y_shift=get_mask_y_shift(front_mask, side_mask)
  nip_h           +=  y_shift
  belly_button_h  +=  y_shift
  hip_h           +=  y_shift
  if debug: pr("after adjustment, \n  nip_h is \n    ",nip_h)
  chest_l=np.count_nonzero( side_mask[nip_h])           # "length," but what this really means is distance from back to nipple.
  waist_l=np.count_nonzero( side_mask[belly_button_h])  # "length," but what this really means is distance from back to nipple.
  hip_l=np.count_nonzero( side_mask[hip_h])  # "length," but what this really means is distance from back to nipple.
  real_h_scale=cust_height/pix_height
  if debug:
    side_mask[nip_h-1         :nip_h+1          ] = 0
    side_mask[belly_button_h-1:belly_button_h+1 ] = 0
    side_mask[hip_h-1         :hip_h+1          ] = 0
    pltshow(side_mask)
    # tODO: rename chest_w and chest_l more descriptively?
    pr("chest_w: {0}".format(chest_w)); pr("chest_l: {0}".format(chest_l))
    pr("waist_w: {0}".format(waist_w)); pr("waist_l: {0}".format(waist_l))
    pr("hip_w: {0}".format(hip_w)); pr("hip_l: {0}".format(hip_l))
    pr("chest_circ: {0}".format(ellipse_circum(chest_w/2., chest_l/2.)*real_h_scale))
    pr("waist_circ: {0}".format(ellipse_circum(waist_w/2., waist_l/2.)*real_h_scale))
    pr("hip_circ: {0}".format(ellipse_circum(hip_w/2., hip_l/2.)*real_h_scale))
    # ellipse circumference is approximately chest circumference (it MAY overestimate a teensy bit.  TODO: double-check whether ellipse circ overestim8s or underestim8s)
  measurements['chest_circum_inches'] = ellipse_circum(chest_w/2., chest_l/2.)  * real_h_scale
  measurements['waist_circum_inches'] = ellipse_circum(waist_w/2., waist_l/2.)  * real_h_scale
  measurements['hip_circum_inches']   = ellipse_circum(hip_w/2.  , hip_l/2.)    * real_h_scale

  # Find actual heights of various body parts:
  #   In numpy, 0,0 is at the top left.  so we have to switch indexes to more intuitive human "height"
  side_locs=np.nonzero(side_mask)
  mask_pix_foot_loc=side_mask.shape[0]- np.max(side_locs[0])
  chest_h         = side_mask.shape[0]- nip_h           - mask_pix_foot_loc
  hip_h           = side_mask.shape[0]- hip_h           - mask_pix_foot_loc
  waist_h         = side_mask.shape[0]- belly_button_h  - mask_pix_foot_loc

  if debug: 
    pr("side_locs is \n", side_locs)
    pr('foot_loc_np: \n',foot_loc_np)
    pr("mask_pix_foot_loc:\n",mask_pix_foot_loc)
    pltshow(side_mask)
  # Scale from pixels to inches:
  measurements['chest_height_inches'] = chest_h * real_h_scale
  measurements['waist_height_inches'] = waist_h * real_h_scale
  measurements['hip_height_inches']   = hip_h   * real_h_scale
  return measurements
  '''
  This algorithm:

  Nathan:
   'chest_circum_inches': 34.575946247110544,
   'hip_circum_inches':   36.59951865808404,
   'waist_circum_inches': 33.58803601511559,
   'chest_height_inches': 55.90405904059041,    # I got 56 inches.
   'hip_height_inches':   41.78966789667897,    # I measured 44 inches.  Also, what openpose calls the "hip" is lower than the hipbone.
   'waist_height_inches': 45.664206642066425}


    Waist is subjective; I think I measured it from where my pantsline is in the picture /home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____rendered.png

    Waist  was      a failure.  I bet it's a segmentation issue, not a circumference-finding issue.
    Hip    was also a failure.  Requires highly-precise segmentation

  Actual empirical measurements (circumferences):
    # I measured myself.  Customers won't want to do this; it took awhile, and I had to find a measuring tape and a mirror
    height      =  75.
    weight      = 157.4
    chest       =  34.
    waist       =  30.
    hips        =  32.
    inseam      =  35.

  '''
  # tODO: try to catch all bugs before they get too serious
#================================================= chest_waist_hips_circum() ======================================================================

#======================================= all for ellipse circum calculation.  Doesn't work yet.  def ellipse_circum_approx(a,b, precision=2): =======================================
#=======================================================================================================================================
def perim_e(a, b, precision=6):
  """Get perimeter (circumference) of ellipse

  Parameters
  ----------
  %(input)s
  a : scalar
    Length of one of the SEMImajor axes (like radius, not diameter)
  b : scalar
    Length of the other semimajor axis
  precision : int
    How close to the actual circumference we want to get.  Note: this function will always underestimate the ellipse's circumference.

  Returns
  -------
  float
    Perimeter
  gaussian_filter : ndarray
      Returned array of same shape as `input`.

  Notes
  -----
  As far as I can tell from ~5 hours working on this, the precise circumference of an ellipse is an open problem in mathematics.  If you see the integral on Wikipedia and have some idea how to solve it, please let the mathematics community (and me: [nathanbendich@gmail.com]) know!
  The real ellipse's circumference will always be more than what this infinite series predicts because the infinite series is a sum of a positive sequence.

  According to https://www.mathsisfun.com/geometry/ellipse-perimeter.html, Ramanujan's approximation that perimeter ~= pi*(a+b)*(1+(3h/(10+sqrt(4-3*h)))) seems to be higher than this series approximation.
  I think Sympy uses Approx. 3 from that ellipse page (https://www.mathsisfun.com/geometry/ellipse-perimeter.html).  I only tested for a = 10 and values of b on the page https://www.mathsisfun.com/geometry/ellipse-perimeter.html (I really hope this link doesn't end up breaking)
    But sympy is slowwww.  Maybe for refactoring, use Ramanujan's approx 3 but turn it into pure python or numpy or something fast?  Or take the mean of this series and Ramanujan's approx. 3?  There's no analytical reason for that; I'm just basing it off the fact that this series undershoots and Ramanujan's approximation overshoots.
  Another potential improvement is to actually take the time to understand the "Binomial Coefficient with half-integer factorials" mentioned on https://www.mathsisfun.com/geometry/ellipse-perimeter.html.  I attempted this for awhile, but got caught on some bug I didn't understand, opting for hard-coding 6 "levels of precision" instead.  So long as there are no overflows, extending this to arbitrary precision should be doable; it just takes a little more mathematical rigor and care than I'm giving right now.  Anyway, I've spent too long writing this documentation, but it was very personally satisfying to take the time to do this right.  I really need to be practical about churning out code faster though, haha.  While just prototyping, this level of detail might not be practical.

  Examples
  --------
  """
  # This function uses approximations mentioned in these sources:
  #   1.(https://www.mathsisfun.com/geometry/ellipse-perimeter.html) and 
  #   2. https://en.wikipedia.org/wiki/Ellipse
  #   3. https://stackoverflow.com/questions/42310956/how-to-calculate-perimeter-of-ellipse
  #   4. 
  # For more reading while refactoring, please consult the wikipedia page, https://math.stackexchange.com/, or wherever else.
  # func perim_e():
  funcname=  sys._getframe().f_code.co_name
  if b > a:
    tmp=a; a=b; b=tmp # swap such that a is always the semi-major axis (bigger than b)
  if precision <= 0:
    pn(2); pr("In function ",funcname); pr("WARNING:  precision cannot be that low"); pn(2)
  if precision >  6:
    # precision higher than 6 not supported as of Tue Feb 26 12:06:35 EST 2019
    pn(2); pr("In function ",funcname); pr("WARNING:  precision that high is not yet supported"); pn(2)
  # To understand what each symbol (h, seq, a, b) means, please see our sponsor at 
  #   https://www.mathsisfun.com/geometry/ellipse-perimeter.html.
  h=((a-b)**2)/((a+b)**2)
  seq=[  1/          h**0,
         1/        4*h**1,
         1/       64*h**2,
         1/      256*h**3,
        25/    16384*h**4, 
        49/    65536*h**5, 
       441/  1048576*h**6]  # only up to 7 terms
  perim=pi*(a+b)*sum(seq[:precision])
  return perim
#=====================================================    perim_e()   ==================================================================

#=======================================================================================================================================
def ellipse_circum_approx(a, b, precision=6):
  return perim_e(a, b, precision=6)
#=============================================    ellipse_circum_approx()   ============================================================
 


#=======================================================================================================================================
def ellipse_circum(a, b):
  '''
  '''
  # semi-major and semi-minor axes
  # https://en.wikipedia.org/wiki/Ellipse
  # https://stackoverflow.com/questions/22560342/calculate-an-integral-in-python
  # TODO: generalize this s.t.  the bigger of a and b gets assigned as a and  the smaller as b
  # TODO: finish this function and integrate it into chest_circum().
  # TODO: conda install sympy into env "cat"
  # TODO: finish this function and integrate it into chest_circum().
  # sympy is TOO SLOW.
  if b > a:
    tmp=a; a=b; b=tmp # swap such that a is always the semi-major axis (bigger than b)
  return ellipse_circum_approx(a, b)
#================================================= ellipse_cirum() =====================================================================



































 
#===================================================================================================================================
def measure_chest(json_fname):
  '''
    Goal: we can automatically sell you the best-fitting shirt from a single picture of you in form-fitting clothing
    This should return the shirt size.
  '''
  # NOTE: get the order of the points right to properly calculate the 
  json_fname='/home/ubuntu/Documents/code/openpose/output/front__nude__grassy_background_keypoints.json'
  measurements_json_dict=parse_ppl_measures(load_json(json_fname))
  x_LShoulder = measurements_json_dict['LShoulder']['x'];y_LShoulder=measurements_json_dict['LShoulder']['y']
  x_RShoulder = measurements_json_dict['RShoulder']['x'];y_RShoulder=measurements_json_dict['RShoulder']['y']
  x_LHip      = measurements_json_dict['LHip']['x']     ;y_LHip     =measurements_json_dict['LHip']['y']     
  x_RHip      = measurements_json_dict['RHip']['x']     ;y_RHip     =measurements_json_dict['RHip']['y']     
  quadrilateral =[(x_LShoulder,y_LShoulder),
                  (x_RShoulder,y_RShoulder),
                  (x_RHip     ,y_RHip     ),
                  (x_LHip     ,y_LHip     )]
  # chest_area_front just means "chest area as measured from the front"
  chest_area_front=area_polygon(quadrilateral)
  # TODO: correlate this with the shirt sizing.  Also, earlier in this process, we have to account for pixel-reality differences in the original images taken; pixel height needs to scale with height of the person.  Height can be "gotten" from iPhone ARKit or "measure" app
  return chest_area_front, measurements_json_dict
#===================================================================================================================================
def show_overlaid_polygon_measures(pic_filename___with_openpose_keypoints_, openpose_keypts_dict, N=4):
  '''
    Parameters
    ----------
    N is the number of sides in the polygon; default 4 means (as of Fri Feb  1 12:02:06 EST 2019) that we take the 2 shoulders and the 2 hips to get the shirt (chest) measurement
      NOTE!  This may (probably WILL) change as problem re-formulation happens
  '''
  img_w_keypts = np.asarray(ii.imread(pic_filename___with_openpose_keypoints_))
  measurements=openpose_keypts_dict # TODO: rename in the function header too?
  pr("img_w_keypts.shape: \n",img_w_keypts.shape)
  plt.imshow(img_w_keypts)
  if N==4:
    # shirt-sizing with 4 torso-points (as you can probably see below, Right and Left Shoulders, and   Right and Left Hips.
    plt.plot( [measurements['LShoulder']['x'] , measurements['RShoulder']['x']],
              [measurements['LShoulder']['y'] , measurements['RShoulder']['y']],  'k-', lw=2) # across clavicle
    plt.plot( [measurements['RHip']['x']      , measurements['RShoulder']['x']],
              [measurements['RHip']['y']      , measurements['RShoulder']['y']],  'k-', lw=2) # down right side
    plt.plot( [measurements['RHip']['x']      , measurements['LHip']['x']],
              [measurements['RHip']['y']      , measurements['LHip']['y']],  'k-', lw=2)
    plt.plot( [measurements['LShoulder']['x'] , measurements['LHip']['x']],
              [measurements['LShoulder']['y'] , measurements['LHip']['y']],  'k-', lw=2)
  plt.show()
  plt.close()
  return

#===================================================================================================================================
def vert_info(vs):
  # TODO: sprinkle this magic sauce vert_info(vs) EVERYWHERE.
  '''
    vert_info stands for "Print Vertices Info."

    -------
    Params:
    -------
      vs are the vertices.  format: a numpy array of shape (n,3).  (in other words, vs=np.nonzero(cartesian))
  '''
  assert vs.shape[1]==3
  x_max = np.max(vs[:,0]); x_min = np.min(vs[:,0]); y_max = np.max(vs[:,1]); y_min = np.min(vs[:,1]); z_max = np.max(vs[:,2]); z_min = np.min(vs[:,2])
  x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
  if debug:
    pe(69); pr("Entering function ",sys._getframe().f_code.co_name); pe(69)
    pr("x_max:",x_max); pr("x_min:",x_min); pr("y_max:",y_max);pr("y_min:",y_min); pr("z_max:",z_max); pr("z_min:",z_min);pn()
    pr("x_len:",x_len); pr("y_len:",y_len); pr("z_len:",z_len);pn()
    pe(69); pr("Leaving function ",sys._getframe().f_code.co_name); pe(69);pn()
  data=(x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len)
  return data
#===================================================================================================================================
def cross_sec(verts, midpt_h, window=0.652, which_ax="z", faces=None):
  '''
    Takes a cross section of an .obj mesh
      As of Sat Mar  9 18:50:20 EST 2019, this method doesn't work.  Or at least, I don't know how to understand/debug it.  The "window" is particularly a problem; it seems to need to adapt to whether we're doing the hip, waist, or chest.

    -------
    Params:
    -------
    verts is the .obj mesh
    midpt_h is the height where we center our "cross section" 

    -------
     TODO:
    -------
      Make adaptive to body mesh resolution.
      Test function "triang_area_R3()."  I at least tested it on a few cases.

    -------
    Notes:
    -------
    "faces" was put in here in case we want to automatically scale the window size to "resolution(SMPL)", where resolution(SMPL) is the average area of a triangle in the SMPL .obj mesh.
    verts is a numpy array of shape (n,3).  (in other words, vs=np.nonzero(cartesian))
  '''
  NUM_DIMS=1; XYZ=3
  assert verts.shape[NUM_DIMS]==XYZ
  X=0; Y=1; Z=2
  if    which_ax.lower()=='x': ax=X
  elif  which_ax.lower()=='y': ax=Y
  elif  which_ax.lower()=='z': ax=Z

  # Sort.  Makes the operation faster
  sorted_verts=verts[np.argsort(verts[:,ax])]

  # Find bounds 
  top = midpt_h + ( window/2.); bot = midpt_h - ( window/2.)
  found_bot=False; found_top=False
  for i,loc in enumerate(sorted_verts[:,ax]):
    if (not found_bot) and (loc > bot):
      found_bot=True; bot_idx=i
    elif (not found_top) and (loc > top):
      found_top=True; top_idx=i
  targ_verts=sorted_verts[bot_idx:top_idx]

  # Only take xy values
  cross_sec=targ_verts[:,:Z]
  if debug:
    plt.title("Vertices in SMPL mesh at loc {0} along the \n{1} axis\nwith {2} points".format(midpt_h, which_ax, len(targ_verts)))
    plt.scatter(cross_sec[:,0],cross_sec[:,1]); plt.show()
  return cross_sec
#=============================================== end cross_sec(params) =============================================================
#===================================================================================================================================
def dist(pt1, pt2, norm='L2'):
  # TODO: see scipy.spatial.distance.euclidean
  # This function "dist()" should be easily vectorizable.

  # points as np arrays
  return math.sqrt(np.sum(np.square(pt1-pt2)))
#=============================================== end dist(params) ==================================================================
def pixel_height(mask):
  locs=np.nonzero(mask)
  return np.amax(locs[0])-np.amin(locs[0])
#===================================================================================================================================
def pix_h(mask):
  return pixel_height(mask)
#===================================================================================================================================















#===================================================================================================================================
def orig_pix_h(img_fname):
  '''
    Pixel height is different in the original vs. after deeplab is done with it
    The result spit out after deeplab segmentation is resized
    See: pix_h(img_fname)
  '''
  # h = height
  orig_img=np.asarray(ii.imread(img_fname))
  orig_h=orig_img.shape[0]
  mask=np.rot90(seg_local(img_fname))
  h_after_seg=pix_h(mask)
  h_as_frac_of_frame=h_after_seg / mask.shape[0] # python3 so no need for (+ 0.0)
  return h_as_frac_of_frame * orig_h
#===================================================================================================================================
def other_toe(verts, toe_1, crotch):
  '''
    Given crotch, find the 1st pt in the mesh on the other side of crotch_x from toe_1, (where crotch_x=crotch[0])
  '''
  funcname=  sys._getframe().f_code.co_name # other_toe(verts, toe_1, crotch):
  X=0; crotch_x=crotch[X]

  # bottom to top:  toes to head
  verts_z_sorted_indices=np.argsort(verts[:,2])
  # greedy search for toe
  for i in verts_z_sorted_indices:
    if (verts[i][X] < crotch_x and toe_1[X] > crotch_x) or\
       (verts[i][X] > crotch_x and toe_1[X] < crotch_x):
      return verts[i]
  raise Exception("Something weird happened in function {0}.\n  Other toe wasn't found".format(funcname)) 
#==================================== end other_toe(params) ========================================================================
#===================================================================================================================================
def normalize_mesh(vs, mode='HMR'): 
  '''
    Standardizes mesh position and orientation

    -------
    Params:
    -------
      vs means vertices of a mesh, 

    return value "vs" ought to have the mesh's head "point" in the +z direction, every mesh vertex should have (+x,+y,+z) values
  '''
  # Note:   refactor.  used to call it standardize_mesh_position_and_orientation(), but was too long prob not descriptive enough?

  # NOTES:  I think currently y is  "height," x is "width," and z is "depth"
  #               but we want z     "height," x    "width," and y is "depth"     (helpfully, this is ALSO how blender does it)
  #           This yz_swap solution below: (Wed Mar  6 13:49:35 EST 2019) is specifically tailored to:
  #             obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_mesh.obj'
  pr("This mesh was generated via ",mode)

  #==============================================================================
  #                         Geometric transformations:
  #==============================================================================
  Z=2
  if mode == 'HMR':
    yz_swap=np.array([[   1,   0,   0],
                      [   0,   0,   1],
                      [   0,  -1,   0]]).astype('float64')
    # TODO: somehow ensure this transformation doesn't turn our mesh "upside down."  Maybe use pltshow() combined with the cKDTree.  
    #   Funny, for the "approx mask" operation we'd really like to have that KDTree() "all-neighbors queries functionality".  https://stackoverflow.com/questions/6931209/difference-between-scipy-spatial-kdtree-and-scipy-spatial-ckdtree

    # Rotate
    vs=vs.dot(yz_swap)

    # Shift
    vs=to_1st_octant(vs)

    # Flip  (this particular mesh was "feet up")
    x_max = np.max(vs[:,0]); x_min = np.min(vs[:,0]); y_max = np.max(vs[:,1]); y_min = np.min(vs[:,1]); z_max = np.max(vs[:,2]); z_min = np.min(vs[:,2])
    x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
    flipped=deepcopy(vs) # could avoid the deepcopy step
    for i,row in enumerate(flipped): # TODO TODO: vectorize; actually affects runtime
      flipped[i,Z]=-flipped[i,Z]
    flipped=shift_verts(flipped,0,0,z_len)
    vs=flipped
    extrema=(x_min,x_max,y_min,y_max,z_min,z_max)
  elif mode == 'NNN':
    # TODO: ensure mesh is head-z-up.
    x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(vs)
    yz_swap=np.array([[   1,   0,   0],
                      [   0,   0,   1],
                      [   0,  -1,   0]]).astype('float64')

    # Rotate
    vs=vs.dot(yz_swap)
    x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(vs)

    # Shift to all positive (+x,+y,+z)
    vs=to_1st_octant(vs)

    x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(vs)
    # TODO: figure out w.t.f. is going on in cross_sec(which_ax='y')
    '''
    for h in np.linspace(z_max-0.1, z_min+0.1, 21):
      cross_sec(vs, h, window=0.05, which_ax='z') # maybe chest is ~1.2 1.3 1.4?
    '''

    '''
      I think we don't need to flip.
    # Flip  (this particular mesh was "feet up")
    x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(vs)
    # TODO TODO TODO TODO TODO TODO TODO TODO shift the NNN .obj mesh appropriately TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
    flipped=deepcopy(vs) # could avoid the deepcopy step
    for i,row in enumerate(flipped): # TODO TODO: vectorize; actually affects runtime
      flipped[i,Z]=-flipped[i,Z]
    flipped=shift_verts(flipped,0,0,z_len)
    vs=flipped
    '''
    extrema=(x_min,x_max,y_min,y_max,z_min,z_max)
  return vs, extrema # TODO TODO TODO TODO TODO:  finish for NNN (SMPL-betas-manually-tuned)      Where do we scale up the mesh??
#============  end normalize_mesh(params): ==============




#=====================================================================
def triang_walk(verts, faces, start_face, height, adjacents, bots_idx, tops_idx, which_ax='z', ax=2):
  '''
    Imagine your waist is at height 50 inches.  Now imagine the vertices (points) of a mesh representing your body.  There will be a set of points right about 50 inches, and a set of points right below 50 inches.  Now imagine the lines connecting each of those points above and below like a zigzagging snake until the snake bites it's own tail.  We're looking for the zigzagging snake right now.  That's what this function does
    pt_A \/\/\/\/\/\/\/\/\/\/\ pt_A


     If we're looking down through your head:

       A    B  

        *  *        
     *        *     
    *          *    
    *          *    
     *        *     
        *  *        

    If we're looking from the side through your belly button:

      point A       tops_idx                                                                     point A
           \/\/\/\/\/\/\/\/\/\/\   ...    (wraps around the waist, back to the belly button) ...  /\
        point B      bots_idx                                                                      point B
  '''
    # Todo: walk_recurs()
  #=====================================================================================
  def walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, top_or_bot='top', which_ax='z', ax=2):
    pn(2); pe(69); pr("Entering function ",sys._getframe().f_code.co_name); pe(69)
    pe(99);print('vert_idx_list is: \n',vert_idx_list);pn();pe(99)
    if vert_idx_list[0]==bots_idx: # TODO: check recursion's base case
      vert_idx_list.append(tops_idx)
      return vert_idx_list
    print("passed base case")
    print("bots_idx:",bots_idx)
    print("tops_idx:",tops_idx)

    pe(69); print("top_or_bot==",top_or_bot); pe(69)
    # Looking for top:
    if top_or_bot=='top':
      faces_top=adjacents[tops_idx]

      # find start_face:
      for face in faces_top:
        if bots_idx in face and tops_idx in face:
          print("face: ",face)
          for v_idx in face:
            if v_idx != bots_idx and v_idx != tops_idx and v_idx not in vert_idx_list:
              next_top_idx=v_idx
              if verts[next_top_idx][ax] < height:
                raise Exception("God dammit we can't solve this problem with recursion.")
              print("next_top_idx:",next_top_idx)
              if next_top_idx not in vert_idx_list:
                vert_idx_list.append(tops_idx) # append PREV
                return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, 'bot', which_ax='z', ax=2)

    # Looking for bot:   NOTE: if there's a bug in one, there's a bug in the other
    elif top_or_bot=='bot':
      faces_bot=adjacents[bots_idx]

      # find start_face:
      for face in faces_bot:
        print("face: ",face)
        print("vert_idx_list:\n",vert_idx_list);pe(69);pn(2)
        if bots_idx in face and tops_idx in face:
          for v_idx in face:
            if v_idx != bots_idx and v_idx != tops_idx and v_idx not in vert_idx_list:
              next_bot_idx=v_idx
              print("height:",height)
              for v_i in vert_idx_list:
                print("verts[{0}] : {1}".format(v_i,verts[v_i]))
              print(  "verts[{0}] : {1}".format(next_bot_idx,verts[next_bot_idx]))
              print(  "verts[{0}] : {1}".format(tops_idx,verts[tops_idx]))
              print(  "verts[{0}] : {1}".format(bots_idx,verts[bots_idx]))
              if verts[next_bot_idx][ax] > height:
                raise Exception("God dammit we can't solve this problem with recursion.")
              print("next_bot_idx:",next_bot_idx)
              if next_bot_idx not in vert_idx_list:
                vert_idx_list.append(bots_idx) # append PREV
                return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, 'top', which_ax='z', ax=2)




















    # Looking for bot:  TODO:
    '''
    if top_or_bot=='bot':
      for face in faces_top:
        print("face:\n",face)
        if bots_idx in face:
          for v_idx in face:
            if v_idx != bots_idx and v_idx != tops_idx:
              next_top_idx=v_idx
              print("next_top_idx:",next_top_idx)
              if next_top_idx not in vert_idx_list:
                vert_idx_list.append(tops_idx) # append PREV
                
                return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, which_ax='z', ax=2)
          vert_idx_list
          loc=np.where(face,bots_idx)
          print("loc:",loc)
    '''

    # prev's code TODO check whether we followed these steps in our recurs() algorithm
    '''
    if next_vert[ax] > height:
      for face in adjacents[next_vert_idx]:
        print("face:", face)
        print("bots_idx       : ",bots_idx)
        print("next_vert_idx  : ",next_vert_idx);pn()
        if bots_idx in face and tops_idx not in face:
          for next_vert_idx_2 in face:
            if next_vert_idx_2 != bots_idx and next_vert_idx_2 != next_vert_idx:
              print("tops_idx       : " , tops_idx)
              print("bots_idx       : " , bots_idx)
              print("next_vert_idx  : " , next_vert_idx)
              print("next_vert_idx_2: " , next_vert_idx_2)
              bots_idx=next_vert_idx_2
              tops_idx=next_vert_idx
              return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, which_ax='z', ax=2)
              # I'd really want a GOTO here...
    '''
    # Todo
  #========================== end walk_recurs(params) ==================================




































































  #========================= begin triang_walk(params) =================================
  pn(2); pe(69); pr("Entering function ", sys._getframe().f_code.co_name); pe(69)
  print("start_face : ",start_face)
  print("bots_idx   : ",bots_idx)
  print("tops_idx   : ",tops_idx)
  #print("height     : ",height);pn()
  vert_idx_list=[bots_idx, tops_idx]

  # Look for 3rd and 4th verts_indices so the recursive base case doesn't immediately kill the recursion
  for vert_idx in start_face:
    if vert_idx != bots_idx and vert_idx != tops_idx:
      next_vert_idx=vert_idx
      next_vert=verts[next_vert_idx]
      print("tops_idx:",tops_idx)
      print("bots_idx:",bots_idx)
      print("next_vert_idx:",next_vert_idx)
      print("verts[tops_idx]      : ",verts[tops_idx])
      print("verts[bots_idx]      : ",verts[bots_idx])
      print("verts[next_vert_idx] : ",verts[next_vert_idx])
      pn()
      print("adjs[tops_idx]:        (tops_idx={0})".format(tops_idx))
      p(adjacents[tops_idx])
      print("adjs[bots_idx]:        (bots_idx={0})".format(bots_idx))
      p(adjacents[bots_idx])
      print("adjs[next_vert_idx]:   (next_vert_idx={0})".format(next_vert_idx))
      p(adjacents[next_vert_idx])
      pn()

      # 3rd pt is above height and 4th pt below
      if next_vert[ax] > height:
        pn(2); pe(69); pr("GOING INTO RETURN "); pe(69)
        for face in adjacents[next_vert_idx]:
          print("face:", face)
          print("bots_idx       : ",bots_idx)
          print("next_vert_idx  : ",next_vert_idx);pn(5)
          if bots_idx in face and tops_idx not in face:
            for next_vert_idx_2 in face:
              if next_vert_idx_2 != bots_idx and next_vert_idx_2 != next_vert_idx:
                print("height     : ",height);pn()
                print("tops_idx       : " , tops_idx)
                print("bots_idx       : " , bots_idx)
                print("next_vert_idx  : " , next_vert_idx)
                print("next_vert_idx_2: " , next_vert_idx_2)
                print("verts[tops_idx]        : " , verts[tops_idx])
                print("verts[bots_idx]        : " , verts[bots_idx])
                print("verts[next_vert_idx]   : " , verts[next_vert_idx])
                print("verts[next_vert_idx_2] : " , verts[next_vert_idx_2])
                bots_idx=next_vert_idx_2
                tops_idx=next_vert_idx
                # TODO: ensure we're starting with top and moving to bot next.
                print("vert_idx_list:\n",vert_idx_list)
                print("right before entering walk_recurs(), face is {0} ".format(face))
                #return
                return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, top_or_bot='top', which_ax='z', ax=2)
                # I'd really want a GOTO here...







































      # TODO See your notebook!
      '''
      make sure to set up the recursion right in the first place (idx0, idx1, idx2, )
        See your notebook!

      vert_idx_list=[bots_idx,tops_idx]
      elif next_vert[ax] <= height: # I can't anticipate a time when they'd be equal
        print("in README README README README README   elif next_vert[ax] <= height")
        pass # TODO: copy from "if next_vert[ax] > height:" block once everything is debugged
      '''



      #NOTE NOTE NOTE NOTE NOTE:  this ((concave quadrilateral)) had better NEVER happen in SMPL.  If it does, our recursion in walk_recurs() won't work.
      #NOTE NOTE NOTE NOTE NOTE:  this ((concave quadrilateral)) had better NEVER happen in SMPL.  If it does, our recursion in walk_recurs() won't work.
      #NOTE NOTE NOTE NOTE NOTE:  this ((concave quadrilateral)) had better NEVER happen in SMPL.  If it does, our recursion in walk_recurs() won't work.
      '''
   \     /
    \   /
     \ /
      V
      # We will iff both of the other adj points are below.   So include it.
      # 3rd pt is below height and 4th pt above
      elif next_vert[ax] <= height: # I can't anticipate a time when they'd be equal
        print("in README README README README README   elif next_vert[ax] <= height")
        pass # TODO: copy from "if next_vert[ax] > height:" block once everything is debugged
      '''


















  # TODO: throw exception
  return walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, which_ax='z', ax=2)
  #=========================== end triang_walk(params) =================================
#=====================================================================
def mesh_cross_sec(verts, faces, height, which_ax="z"):
  '''
    Takes a cross section of an .obj mesh
    The plane that creates the cross section is at "height" along "which_ax"

    verts are the pts of the mesh
    faces are the triangles
  '''
  pn(2); pe(69); pr("Entering function ",sys._getframe().f_code.co_name); pe(69)
  pr("height:",height)
  NUM_DIMS=1; XYZ=3
  assert verts.shape[NUM_DIMS]==XYZ
  X=0; Y=1; Z=2
  if    which_ax.lower()=='x': ax=X
  elif  which_ax.lower()=='y': ax=Y
  elif  which_ax.lower()=='z': ax=Z

  adjs=adjacents(verts, faces)

  # Sort low to high (small to big).  Makes the operation faster
  sort_indices  = np.argsort(verts[:,ax])
  sorted_verts  = verts[sort_indices]
  #np.greater(sorted_verts[:,ax],height) # NOTE: there must be a way to do this directly without iterating cumbersome-ly

  # Triangle walk straddling the height
  for i,loc in enumerate(sorted_verts[:,ax]):
    #print("loc:",loc) # GOOD   # NOTE: in func "mesh_cross_sec"
    if loc > height: # the 1st one above "height" because it's harder to find the 1st one directly below height
      tops_idx          = sort_indices[i]
      these_faces       = adjs[tops_idx]
      for face in these_faces:
        for vert_idx in face:
          if verts[vert_idx][ax] < height: # find an adj "below the belt" :    nxb Sun Mar 10 09:07:58 EDT 2019
            bots_idx=vert_idx
            meshs_belt=triang_walk(verts, faces, face, height, adjs, bots_idx, tops_idx, which_ax=which_ax, ax=ax)

            # Once we have 2 verts, we should be able to track the height until we get back to the original 2.  Then the cross section is the intersection of the plane at "height" with the triangles from the smpl mesh.
      break
    # [2,2,2], [1,1,1], [0,0,0],
    # 6888: [ array([6887, 6886, 6888]),
    #         array([6888, 6886, 6889]),
    #         array([5238, 6887, 6888]),
    #         array([6889, 3953, 6888]),
    #         array([5238, 6888, 5239]),
    #         array([5239, 6888, 3953])],

  # Find bounds
  top = midpt_h + ( window/2.); bot = midpt_h - ( window/2.)
  found_bot=False; found_top=False
  for i,loc in enumerate(sorted_verts[:,ax]):
    if (not found_bot) and (loc > bot):   # should happen first
      found_bot=True; bot_idx=i
    elif (not found_top) and (loc > top): # before this guy
      found_top=True; top_idx=i
  targ_verts=sorted_verts[bot_idx:top_idx]

  # Only take xy values
  cross_sec=targ_verts[:,:Z]
  if debug:
    plt.title("Vertices in SMPL mesh at loc {0} along the \n{1} axis\nwith {2} points".format(midpt_h, which_ax, len(targ_verts)))
    plt.scatter(cross_sec[:,0],cross_sec[:,1]); plt.show()
  return cross_sec
#====================== mesh_cross_sec(params) =======================
#=====================================================================
def adjacents(verts, faces):
  # .obj mesh
  adjs={i:[] for i in range(len(verts))}
  for face_idx,face in enumerate(faces): # 1,  v1 v2 v3
    v0=face[0]; v1=face[1]; v2=face[2]
    adjs[v0].append(face)
    adjs[v1].append(face)
    adjs[v2].append(face)
  return adjs
#=====================================================================



#=====================================================================
def lines_intersection_w_plane(vert_0, vert_1, height, which_ax='z'):
  '''
    The line in "line's intersection" we're referring to is    the line segment going through vert_0 and vert_1.
    The plane referred to in "w_plane" is the plane "which_ax==height"

    Notes:
      See algebra from https://math.stackexchange.com/questions/404440/what-is-the-equation-for-a-3d-line
  '''
  x0, y0, z0= vert_0
  a , b , c = vert_0- vert_1
  x = y = z = height
  # Note: there's probably a way to generalize this function better so we only have to edit 1 line of code instead of 3
  if    which_ax=='x':
    y = (b*(x-x0)/a)  +y0
    z = (c*(x-x0)/a)  +z0
  elif  which_ax=='y':
    x = (a*(y-y0)/b)  +x0
    z = (c*(y-y0)/b)  +z0
  elif  which_ax=='z':
    x = (a*(z-z0)/c)  +x0
    y = (b*(z-z0)/c)  +y0
  intersect_pt=x,y,z
  return intersect_pt
#============ end lines_intersection_w_plane(params) =================
#=====================================================================
def mesh_perim_at_height(verts, faces, height, window=19.952, which_ax='z', ax=2, plot=False):
  if debug:
    funcname=sys._getframe().f_code.co_name
    pe(69);pr("Entering function ",funcname);pe(69);pn()
  # verts.shape     : (6890,3)

  # TODO: iterate through the faces once, wherever one edge has 2 pts: 1 above height and the other below, calculate the intersection point and toss the intersection points into ConvHull_perim().
  if    which_ax=='x': ax=0
  elif  which_ax=='y': ax=1
  elif  which_ax=='z': ax=2
  else: raise Exception("when calling the function '{0}',  please only supply the 'which_ax' parameter as which_ax='x', which_ax='y', or which_ax='z', ".format(sys._getframe().f_code.co_name))
  adjs=adjacents(verts, faces)
  perim_pts=[]
  #=====================================================================
  def line_seg_crosses_height(pt1,pt2,h,ax=2):  #ax=2 means that axis='z'):
    return (pt1[ax]>h and pt2[ax]<h) or (pt1[ax]<h and pt2[ax]>h) # default is z
  #=====================================================================
  for face in faces:
    vert_0=verts[face[0]]; vert_1=verts[face[1]]; vert_2=verts[face[2]]
    if line_seg_crosses_height(vert_0,vert_1,height,ax):
      perim_pts.append(lines_intersection_w_plane(vert_0,vert_1,height,which_ax))
    if line_seg_crosses_height(vert_0,vert_2,height,ax):
      perim_pts.append(lines_intersection_w_plane(vert_0,vert_2,height,which_ax))
    if line_seg_crosses_height(vert_1,vert_2,height,ax):
      perim_pts.append(lines_intersection_w_plane(vert_1,vert_2,height,which_ax))
  perim_pts=np.array(perim_pts)
  if debug or plot:
    title="Vertices in SMPL mesh at loc {0} along the \n{1} axis\nwith {2} points".format(height, which_ax, len(perim_pts))
    viz.plt_plot_2d(perim_pts, title,ax=ax)
  if    which_ax=='x':
    perim=conv_hulls_perim(perim_pts[:,1:])
  elif  which_ax=='y':
    x=perim_pts[:,0].reshape((perim_pts[:,0].shape[0],1))
    z=perim_pts[:,2].reshape((perim_pts[:,2].shape[0],1))
    xz=np.concatenate((x,z),axis=1)
    perim=conv_hulls_perim(xz)
  elif  which_ax=='z':
    perim=conv_hulls_perim(perim_pts[:,:2])
  #pr("perim_pts.shape:",perim_pts.shape)  #(600,3)

  bots_idx=np.argmin(perim_pts[:,2])
  crotch=perim_pts[bots_idx]
  if debug:
    pe(69);pr("Leaving function ",funcname);pe(69);pn()
  return perim, crotch # I know this is sketch AF, but easier to return the crotch as an afterthought than to not find the crotch at all.
#=============== end mesh_perim_at_height(params) ===============






#================================================================
def parse_obj_file(obj_fname):
  verts=[]
  faces=[]
  with open(obj_fname, 'r') as f:
    for line in f.readlines():
      if line.startswith('v'):
        verts.append(line)
      elif line.startswith('f'):
        faces.append(line)
        # Note: faces start indexing at 1
  verts_arr=np.zeros((len(verts), 3)).astype("float64") # 'vs' means verts
  fs=np.zeros((len(faces), 3)).astype("int64"  ) # 'fs' means faces
  X,Y,Z=1,2,3
  for idx,vert in enumerate(verts):
    vert=vert.split(' ')
    verts_arr[idx]=vert[X],vert[Y],vert[Z]
    #case={}.get(x,DEFAULT_RET_VAL)
    #generating_method='HMR' or 'NNN' or 'VHMR' or ...
  #pr("measures:"); #p(measures) #print("chest_h:", chest_h) # measured, 55.9 inches is ~correct

  # read in faces
  for idx,f in enumerate(faces):
    f=f.split(' ')
    fs[idx]=np.array([int(f[1]),int(f[2]),int(f[3])]).astype("int64")-1
    # see .obj file for why these conventions are like they are.  each face is described as a sequences of 3 vertices:  ie. "f 1 99 4"
    # in .obj files, face indexing starts at 1  (face 1 might be 1 2 3 rather than 0 1 2, referring to vertices 1, 2, and 3)
  faces=fs
  verts=verts_arr
  return verts, faces
#=============== end parse_obj_file(params) =============== 


#====================================================================================
def mesh_err(obj_fname, json_fname, front_fname, side_fname, cust_height):
  '''
    Program's curr state: (Mar 11, 2019):
    1. Find height from openpose keypoints json (debug)
      a.  This part (openpose) is WAY TOO SLOW.  Maybe with AWS' higher RAM/GPU we will be able to get into a workflow.
    2. Use height to approximately find the points in SMPL that should correspond to the chest circumference
    3. Take the perimeter of the ConvexHull() of that polygon (the approx cross-section of SMPL)

    ------
    Params
    ------
    obj_fname:  the .obj mesh filename (in .obj format) to read faces and vertices from
    json_fname:  openpose keypoints json
    front_fname:  picture of the customer taken from the front
    side_fname:   Picture of the customer from the side (ie. prisoners' mugshot 2 (https://proxy.duckduckgo.com/iu/?u=http%3A%2F%2Fcdn-s3.thewrap.com%2Fimages%2F2014%2F01%2Fjustin-bieber-side-mugshot.jpg&f=1))
    cust_height: customer's height in INCHES

  '''
  # func name mesh_err()
  #obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_mesh.obj'

  # Load .obj file
  verts, faces=parse_obj_file(obj_fname)

  # geometric transformations to put the mesh in the "upright" position (ie. +z is where the head is, arms stretch out in +/-x direction, chest-to-back is +/-y direction.)
  if 'HMR' in obj_fname.upper():
    # do shifts like done below
    mode='HMR'
    verts, extrema=normalize_mesh(verts, mode)
  elif 'NNN' in obj_fname.upper():
    mode='NNN'
    verts, extrema=normalize_mesh(verts, mode)
  print("mesh was generated by ",mode)
  x_min, x_max, y_min, y_max, z_min, z_max = extrema

  # Scale to real-life-sizes (cust_height in inches):
  verts    = verts * cust_height / z_max
  x_min, x_max, y_min, y_max, z_min, z_max, x_len, y_len, z_len= vert_info(verts)

  # Use openpose keypoints json to get measurements
  measures= measure_body_viz(json_fname, front_fname, side_fname, cust_height)
  chest_h = measures['chest_height_inches']  # Nathan's real chest_h is 57 inches
  hip_h   = measures['hip_height_inches']    # Nathan's real hip_h   is    inches
  waist_h = measures['waist_height_inches']  # Nathan's real waist_h is    inches

  print("chest_h: ",chest_h )
  print("hip_h  : ",hip_h   )
  print("waist_h: ",waist_h )
  x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(verts)
  print("x_min: {0}\nx_max: {1}\ny_min: {2}\ny_max: {3}\nz_min: {4}\nz_max: {5}\nx_len: {6}\ny_len: {7}\nz_len: {8}\n".format(x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len))

  # TODO: clean up the searching-for-x-and-y-slices code below:
  # Code for scanning for crotch:
  PAD=3.1
  RESOLUTION=41

  '''
  for z in np.linspace(z_max- PAD, z_min+ PAD,  RESOLUTION):
    z_slice             = mesh_perim_at_height( verts, faces, z, which_ax='z',plot=True)
  '''
  # -x is the raised hand (approx. x==0)

  # for loops do crotch-length-finding:
  '''
  for x in np.linspace(x_max- PAD, x_min+ PAD,  RESOLUTION):
    x_slice             = mesh_perim_at_height( verts, faces, x, which_ax='x',ax=0, plot=True)
  for x in np.linspace(x_max- PAD, x_min+ PAD,  RESOLUTION):
    x_slice             = mesh_perim_at_height( verts, faces, x, which_ax='x')
  for y in np.linspace(y_max- PAD, y_min+ PAD,  RESOLUTION):
    y_slice             = mesh_perim_at_height( verts, faces, y, which_ax='y')
  '''

  '''
  x_mid=(x_max+x_min)/2.
  y_mid=(y_max+y_min)/2.
  x_slice             = mesh_perim_at_height( verts, faces, x_mid-10, which_ax='x',ax=0)
  y_slice             = mesh_perim_at_height( verts, faces, y_mid-10, which_ax='y',ax=1)
  for z in np.linspace(z_max- 0.1, z_min+0.1, 11):
    z_slice             = mesh_perim_at_height( verts, faces, z, which_ax='z',ax=2)
  '''

  calced_chest_circum , _ = mesh_perim_at_height(verts, faces, chest_h, which_ax='z')
  calced_hip_circum   , _ = mesh_perim_at_height(verts, faces, hip_h  , which_ax='z')
  calced_waist_circum , _ = mesh_perim_at_height(verts, faces, waist_h, which_ax='z')

  #crotch ratio: {'height': 255/433 down from the top, 'x_loc': 120/221 from the left to the right}
  CROTCH_LR_RATIO=120/211.
  crotch_depth=int(round(x_len*(CROTCH_LR_RATIO)))
  min_height=np.inf
  real_crotch=None
  real_crotch_depth=crotch_depth
  start = crotch_depth-3
  end   = crotch_depth-1
  print("start = {0}     and end = {1}".format(start, end))
 
  # TODO NOTE This perfect-crotch-search takes too long.  We have to adapt it somehow to find a saddle point in z.   Or to only find the highest min point instead of calculating the whole ConvexHull and perimeter every time.  But before we make it fast, we prob have to check that we actually WANT the crotch so precisely.
  # Note: refactor into separate get_crotch()
  for d in np.linspace(start, end, 99):
    calced_crotch_2_head_circum, crotch = mesh_perim_at_height(verts, faces, d, which_ax='x')
    if calced_crotch_2_head_circum < min_height:
      min_height=calced_crotch_2_head_circum
      real_crotch_depth=d
      real_crotch=crotch
  print("min_height:",min_height)
  print("real_crotch_depth:",real_crotch_depth)
  print("real_crotch:",real_crotch)# NOTE: real_crotch: [27.65771812 12.4297897  31.31119602] 
  pn(9);pe();pe();pe();pe();pr(" "*24+"about to calculate crotch");pe();pe();pe();pe();pn(9)
  calced_crotch_2_head_circum, _ = mesh_perim_at_height(verts, faces, crotch_depth, which_ax='x')
  pr("calced_crotch_2_head_circum:", calced_crotch_2_head_circum) # real is ~    calcul8d is ~102.02471693093469 inches
  calced_crotch_2_head_circum, _  = mesh_perim_at_height(verts, faces, real_crotch_depth, which_ax='x', plot=True)

  # NOTE: real_crotch: [27.65770833 12.42979636 31.31119489],    NOT  [27.65771812 12.4297897  31.31119602]  
  pn();pe();pe();pe();pe();pr(" "*24+"about to calculate crotch");pe();pe();pe();pe();pn()

  # toe calculation:   properly gets inseam length (inches)
  bots_idx=np.argmin(verts[:,2])
  toe1=verts[bots_idx] # left toe as of Mon Mar 18 13:42:42 EDT 2019
  # NOTE: not ACTUALly "toes," but definitely parts of each leg.
  '''
    toe1: [24.92924817 10.61380322  0.        ]
    toe2: [34.64956495 10.4945032   0.50675734]
  '''
  toe1_to_head_perim, _ = mesh_perim_at_height(verts, faces, toe1[0], which_ax='x', plot=True)
  toe2=other_toe(verts, toe1, real_crotch) # right toe as of Mon Mar 18 13:42:42 EDT 2019
  toe2_to_head_perim, _ = mesh_perim_at_height(verts, faces, toe2[0], which_ax='x', plot=True)
  # ToDO: use real_crotch_depth to calculate inseam.  Is this good enough?  (Better/ worse than finding the toes & calcul8ing the inseam by dist_btwn(crotch, toe)
  # TOdO: generalize the crotch-finding calculation, hook up the openpose shit (hip, chest, waist, etc.  heights)     end-to-end




  #  toe: [24.92924817 10.61380322  0.        ]

  #x_min: 0.0
  #x_max: 52.79738902171417
  #y_min: 0.0
  #y_max: 26.23053931888615
  #z_min: 0.0
  #z_max: 75.0
  #x_len: 52.79738902171417
  #y_len: 26.23053931888615
  #z_len: 75.0


  # 
  pr("calced_crotch_2_head_circum:",calced_crotch_2_head_circum) # real is ~    calcul8d is ~102.02471693093469 inches.    Lower is 101.51852706256683
  pr("calced_chest_circum:  ", calced_chest_circum)   # real is ~34 inches (ConvexHull)
  pr("calced_hip_circum  :  ", calced_hip_circum)     # real is ~32 inches (ConvexHull)
  pr("calced_waist_circum:  ", calced_waist_circum)   # real is ~30 inches (ConvexHull)

  # HMR:                                                  #   when chest_len=29:
  # PRECISE answers (using ConvexHull of the actual plane-intersection-with-.obj-file-mesh):
  #calced_chest_circum:   41.37312593391364
  #calced_hip_circum  :   35.30089642536351
  #calced_waist_circum:   34.32977560896902

  # Todo: make "err" calculation more comprehensive
  real_chest  = measures['chest_circum_inches']
  real_hip    = measures['hip_circum_inches'  ]
  real_waist  = measures['waist_circum_inches']
  err_percents=np.array([(calced_chest_circum-real_chest)  / real_chest,
                         (calced_hip_circum  -real_hip)    / real_hip,
                         (calced_waist_circum-real_waist)  / real_waist])

  #   From "measures":
  # 'chest_circum_inches': 34.575946247110544,
  # 'chest_height_inches': 55.90405904059041,
  # 'hip_circum_inches'  : 36.59951865808404,
  # 'hip_height_inches'  : 41.78966789667897,
  # 'waist_circum_inches': 33.58803601511559,
  # 'waist_height_inches': 45.664206642066425}

  #   But where did these measurements ACTUALLY come from?  The masks?  I think so (as of Mon Mar 11 15:19:31 EDT 2019)

  pr("err_percents:\n",err_percents)
  avg_err_percent=100.0*np.mean(err_percents) # this "avg_err_percent" is really not the best measure of error.
  #IMHO, we want to see the WORST the method does; not the avg
 
  #err_percent = 100.0 * (calced_chest_circum-real_chest) / real_chest
  #err_percent = 100.0 * (calced_chest_circum-real_chest) / real_chest
  err_percent = avg_err_percent
  pe(69); pr("Leaving function ",sys._getframe().f_code.co_name); pe(69)
  pn(29)
  # TODO: MSE, not MAE (punish big deviations more)
  return err_percent, measures, verts
#========================== end mesh_err() ===========================




#===================================================================================================================================
def conv_hulls_perim(xy_pts):
  '''
    Takes the perimeter of the convex hull of a set of points in the xy plane
 
    xy_pts is np array of shape (n,2)
  '''
  XY=2
  assert xy_pts.shape[1]==2
  hull=ConvexHull(xy_pts)
  vertices = hull.vertices.tolist() + [hull.vertices[0]] # Todo: shorten
  perim_edgepts=xy_pts[vertices]
  X=0; Y=1
  if debug:
    plt.title(" cross section's ConvexHull:    \n with {0} points".format(perim_edgepts.shape[0]))
    plt.scatter(perim_edgepts[:,X],perim_edgepts[:,Y]); plt.show()
  perim     = np.sum([euclidean(x, y) for x, y in zip(perim_edgepts, perim_edgepts[1:])])
  return perim
#================================================== end conv_hulls_perim(params) ===================================================














#===================================================================================================================================
def perim_poly(verts):
  '''
    verts have to be in order
  '''
  pn(3);pr("within ",sys._getframe().f_code.co_name)
  prev=verts[-1]
  p=0
  for v in verts:
    print(v)
    p+=dist(prev,v)
    prev=v
  return p
#===================================================================================================================================
def scale(v, s):
  '''
    v vertices
    s scale
  '''
  return v * s
#===================================================================================================================================
def rot8_obj(v, rotation):
  '''
    v = vertices
    rotation = 3x3 np.ndarray
  '''
  return v.dot(rotation)
#===================================================================================================================================
def shift_verts(v, del_x, del_y, del_z):
  '''
    v = vertices
  '''
  shifted=v+\
    np.concatenate((
      np.full((v.shape[0], 1),del_x), # v.shape[0] is num_verts
      np.full((v.shape[0], 1),del_y),
      np.full((v.shape[0], 1),del_z)),axis=1)
  return shifted
#===================================================================================================================================
def to_1st_octant(v):
  '''
    v = vertices
  '''
  funcname=  sys._getframe().f_code.co_name
  if debug:
    pe(69);pr("In function ",funcname);pe(69);pn(2)
  return shift_verts(v, -np.min(v[:,0]), -np.min(v[:,1]), -np.min(v[:,2]))
#===================================================================================================================================
def tri_area(tri_3x3):
  '''
    Calculates the area of a triangle with (x,y,z) for each point.
    This area calculation comes from half the cross product: 

      |AB X AC|       |AB||AC||sin(theta)|
      _________   =   ____________________
                                                 
          2                    2

    Where theta is the angle between vectors AB and AC.



    -------
    Params:
    -------
    tri_3x3 is a 3x3 numpy array:
      [xA, yA, zA]
      [xB, yB, zB]
      [xC, yC, zC]

    ------
    Notes:
    ------
    https://math.stackexchange.com/questions/128991/how-to-calculate-area-of-3d-triangle

    ------
    Tests:
    ------
    triang=np.array([ [ 0   , 0   , 0   ],
                      [ 0   , 0   , 1/9 ],
                      [ 0   , 9   , 0   ]]).astype("float64")
    print(triang)
    print(triang_area_R3(triang))
  '''
  A=tri_3x3[0]; B=tri_3x3[1]; C=tri_3x3[2]
  # consider if A is at the origin; the vector from (0,0,0) to B is just (x_B, y_B, z_B).
  AB=B-A; AC=C-A
  x_AB,y_AB,z_AB = AB
  x_AC,y_AC,z_AC = AC
  term_1=(y_AB*z_AC - z_AB*y_AC)**2
  term_2=(z_AB*x_AC - x_AB*z_AC)**2
  term_3=(x_AB*y_AC - y_AB*x_AC)**2
  return 1/2 * math.sqrt(term_1+term_2+term_3)
#===================================================================================================================================












#===================================================================================================================================
def test_measure():
  # as of Feb 28, 2019,  timing is:  
  #   real     0m 48.063s
  #   user     0m 18.796s
  #   sys      0m  0.747s
  # in other words, much much much better than openpose.

  # NATHAN
  NATHAN_H    = 75 # inches
  json_fname  = '/home/n/Dropbox/vr_mall_backup/json_imgs_openpose/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____keypoints.json'
  #'/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose__legs_closed___nude__grassy_background_keypoints.json'
  #'/home/n/Dropbox/vr_mall_backup/IMPORTANT/front__nude__grassy_background_keypoints.json'
  #'/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'

  # orig front and side imgs
  front_fname = '/home/n/Dropbox/vr_mall_backup/imgs/n8_front___jesus_legs_closed/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  # '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/front__nude__grassy_background.jpg'
  side_fname  = '/home/n/Dropbox/vr_mall_backup/imgs/n8_side___jesus_pose_legs_closed/n8_side___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  # '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/unclothed_outside_delaware_____uniform_background_with_wobbling/000000143.jpg'

  measures=measure_body_viz(json_fname,front_fname,side_fname, NATHAN_H)
  pn(3)
  pr("chest_circ: ",measures['chest_circum_inches'])
  return measures['chest_circum_inches']
#================================================= end test_measure() ==============================================================














#===================================================================================================================================
if __name__=="__main__":
  NATHANS_HEIGHT_INCHES=75
  json_fname  = '/home/n/Dropbox/vr_mall_backup/json_imgs_openpose/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____keypoints.json'
  side_fname  = '/home/n/Dropbox/vr_mall_backup/imgs/n8_side___jesus_pose_legs_closed/n8_side___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  front_fname = '/home/n/Dropbox/vr_mall_backup/imgs/n8_front___jesus_legs_closed/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  HMR_HEMAN_obj_fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/n8_mesh__HMR_gener8d__heman_pose__legs_spread.obj'
  HMR_obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nxb_HMR_gener8d_mesh___gamma_the_magnet_warrior___pose.obj'
  # HMR calced_chest_circum: 41.31167530329451.  Error percent is 19.480968092801966% (overshooting my real chest circumference)
  NNN_obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nxb_manually_tuned_(NNN)___smpl_mesh____4th_iteration__02-2.250000000.obj' # NNN stands for "Nathan the Neural Net"
  # NNN calced_chest_circum: 38.31877278356377.  Error percent is 10.824943183633078% (overshooting my real chest circumference)
  obj_fname=HMR_HEMAN_obj_fname#HMR_obj_fname # NOTE: This line is where I change which .obj file we read in.
  err, measures, vs=mesh_err(obj_fname, json_fname, front_fname, side_fname,NATHANS_HEIGHT_INCHES)
  print("error percentage was {0} percent".format(abs(err)))

  #print("measures:",measures)
  #print("verts:",vs)
  #pr(err, measures, vs) #if __name__=="__main__":

  #test_measure()
























  '''
  # all code here assumes the person we're dealing with is Nathan.  
  #   ie. Nathan is Male, height=tall, width=small, shoulders-hip ratio is reasonable

  NATHANS_HEIGHT = 75 # note: inches.  Perhaps we ought to make it cm instead?
  customer_height = NATHANS_HEIGHT # TODO:  instead of blanket setting customer_height = NATHANS_HEIGHT, we should set customer_height = iPhone.ARKit.measure(height) 
  #this should be made adaptive to customer's height later

  orig_img=np.asarray(ii.imread('/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/front__nude__grassy_background.jpg'))
  orig_imgs_height_in_pixels=orig_img.shape[0]
  # orig_img and orig_imgs_height_in_pixels are necessary because the original json measurements were fetched from an image with these (orig_img and origs_height_in_pixels) dimensions, whereas the deeplab segmentation mask has different pixel dimensions

  mask=np.asarray(ii.imread('/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_29____09:20_AM___/000000220.jpg')).astype('float64')
  mask=np.asarray(ii.imread('/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_29____09:20_AM___/000000220.jpg')).astype('float64')
  pr("mask.shape:\n",mask.shape);pr('\n'*1)
  height_in_pixels=pixel_height(mask)
  nathans_gender='male'
  gender=nathans_gender

  json_fname='/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'
  chest_area, other_body_measurements=measure_chest(json_fname) # TODO: change the measurements to shoulders,
  pr("chest_area:",chest_area)
  LShoulder = np.array([ other_body_measurements['LShoulder']['x'],  other_body_measurements['LShoulder']['y']]).astype('float64')
  RShoulder = np.array([ other_body_measurements['RShoulder']['x'],  other_body_measurements['RShoulder']['y']]).astype('float64')
  LHip      = np.array([ other_body_measurements['LHip']['x']     ,  other_body_measurements['LHip']['y']]     ).astype('float64')
  RHip      = np.array([ other_body_measurements['RHip']['x']     ,  other_body_measurements['RHip']['y']]     ).astype('float64')
  pr("LShoulder,RShoulder,LHip,RHip:",LShoulder,RShoulder,LHip,RHip)
  if gender.lower()=='female':
    s2h_ratio_const   = 1/3.
    # TODO:  find "zero point" for shoulders to hips for a female.
  else:
    s2h_ratio_const   = 2/5.  # TODO: empirically figure out what this s2h_ratio_const ought to be
    zero_beta__shoulder_2_hips=3.481943933038265
    # I've empirically derived that Nathan's shoulders_hips_diff___inches in that one picture (/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/front__nude__grassy_background.jpg) is 3.48.   This is for the variable assignment "shoulders_hips_diff___inches= (dist(LShoulder,RShoulder) - dist(LHip,RHip)) / orig_imgs_height_in_pixels * customer_height"

  # TODO: fuck with s2h_ratio_const until it's right
  shoulders_hips_diff___inches= (dist(LShoulder,RShoulder) - dist(LHip,RHip)) / orig_imgs_height_in_pixels * customer_height
  # NOTE: we have to make sure the image is well-cropped (consistently cropped) s.t. the customer height is a consistent fraction of the total image height.  TODO TODO!
  pr("shoulders_hips_diff___inches:\n",shoulders_hips_diff___inches)
  beta_shoulders_hips = (shoulders_hips_diff___inches - zero_beta__shoulder_2_hips) * s2h_ratio_const 
  pr("beta_shoulders_hips:    {0}".format(beta_shoulders_hips))
  with open('6th_beta.txt', 'w+') as fp: # overwrites *.txt      # 6th beta, 5 is the array idx
    fp.write(str(beta_shoulders_hips)+'\n')

  with open('/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world/gender.py', 'w+') as fp:
    fp.write(str('gender=\'')+gender+'\'')

  """
  # draw relevant polygon on top of image
  openpose_fname='/home/ubuntu/Documents/code/openpose/output/front__nude__grassy_background_rendered.jpg'
  #openpose_fname='/home/ubuntu/Documents/code/openpose/output/openpose_success!.jpg'
  show_overlaid_polygon_measures(openpose_fname, other_body_measurements)
  """

  #os.system('source /home/ubuntu/Documents/code/hmr/venv_hmr/bin/activate && cd /home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world && python2 hello_smpl.py 0 0 0 0 0 0 {0} 0 0 0 && blender'.format(chest_area)) # TODO: fiddle with chest area
  '''
#===================================================================================================================================

































































# Glossary:
'''
glossary (for grep-easy-find)
  Function definitions (function headers)
===============================================================
  as of Sun Mar 17 09:25:56 EDT 2019
===============================================================
    128:def load_json(json_fname):
    133:def measure(json_fname):
    136:def parse_ppl_measures(json_dict):
    172:def segments(polygon):
    178:def area_polygon(polygon):
    212:def estim8_w8(json_fname, front_fname, side_fname, height):
    238:def measure_body_viz(json_fname, front_fname, side_fname, cust_height):
    432:def perim_e(a, b, precision=6):
    495:def ellipse_circum_approx(a, b, precision=6):
    502:def ellipse_circum(a, b):
    554:def measure_chest(json_fname):
    575:def show_overlaid_polygon_measures(pic_filename___with_openpose_keypoints_, openpose_keypts_dict, N=4):
    601:def vert_info(vs):
    622:def cross_sec(verts, midpt_h, window=0.652, which_ax="z", faces=None):
    673:def dist(pt1, pt2, norm='L2'):
    680:def pixel_height(mask):
    684:def pix_h(mask):
    703:def orig_pix_h(img_fname):
    717:def normalize_mesh(vs, mode='HMR'): 
    802:def triang_walk(verts, faces, start_face, height, adjacents, bots_idx, tops_idx, which_ax='z', ax=2):
    827:  def walk_recurs(verts, height, adjacents, vert_idx_list, bots_idx, tops_idx, top_or_bot='top', which_ax='z', ax=2):
    1151:def mesh_cross_sec(verts, faces, height, which_ax="z"):
    1215:def adjacents(verts, faces):
    1229:def lines_intersection_w_plane(vert_0, vert_1, height, which_ax='z'):
    1254:def mesh_perim_at_height(verts, faces, height, window=19.952, which_ax='z', ax=2, plot=False):
    1265:  def line_seg_crosses_height(pt1,pt2,h,ax=2):  #ax=2 means that axis='z'):
    1295:def parse_obj_file(obj_fname):
    1328:def mesh_err(obj_fname, json_fname, front_fname, side_fname, cust_height):
    1481:def conv_hulls_perim(xy_pts):
    1514:def perim_poly(verts):
    1527:def scale(v, s):
    1534:def rot8_obj(v, rotation):
    1541:def shift_verts(v, del_x, del_y, del_z):
    1552:def to_1st_octant(v):
    1561:def tri_area(tri_3x3):
    1620:def test_measure():


  Chest:
    Measure the circumference of your chest at your nipple level. Hold the end of the measuring tape in the middle of your chest at your nipple level. Evenly wrap the measuring tape around your chest. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.


  Waist:
    Measure the circumference of your waist at your preferred waistline. Your preferred waistline is where you typically wear the waist of your pants. After exhaling, hold the beginning of the measuring tape in front of your body in the middle of your waistline. Evenly wrap the measuring tape around your waist. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.


  Hips:
    Measure the circumference of your hips at the level where your hips are widest. Hold the beginning of the measuring tape in front of your body in the middle of your hip line. Evenly wrap the measuring tape around your hips. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.


  Inseam:
    Measure your inseam from your crotch to the floor. Your crotch is the inner, uppermost point of leg. Hold the beginning of the tape measure at your crotch. Make sure you are holding it at the 0 mark. Pull the tape measure down to the floor where your foot is situated. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.



'''






































































'''
 Nathan (approx):
  height      =  75.    # inches
  weight      = 157.4   # pounds
  chest       =  34.    # inches (nipples)
  waist       =  30.    # inches (belly button)
  hips        =  32.    # inches (below hip bone)
  inseam      =  35.    # inches

  My Lee's Jeans are 29x36 inches (29 waist, 36 inseam).  The inseam measures almost EXACTLY 36 inches when I lay them out on the table and measure the literal seam from crotch to hem   Some jeans that mostly 
  If a pair of jeans should fit around the belly-button area (no need for a belt, then!  (we can sell customers on this)), we should be selling jeans specifically tailored for that length.  I'm not 100% sure whether jeans are supposed to fit this way, though.  Maybe we could try to strong-arm the jeans companies into selling at that size?
'''
# TODO: we have to define our measurements     OURSELVES, not just copy what www.bodyvisualizer.com says
#   let's start by taking some measurements and seeing what actually fits with the jeans/shirts we HAVE
