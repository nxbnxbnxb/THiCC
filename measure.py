from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#TODO? rename to mzr?
import json
import numpy as np
np.seterr(all='raise')
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import ConvexHull as ConvHull
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
def measure_body_viz(json_fname, front_fname, side_fname, cust_h):
# used to be called "chest_waist_hips_circum(json_fname, front_fname, side_fname, cust_h)"
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
  pr("masks_h: ",masks_h)
  pr("orig_h:  ",orig_h)

  # Change heights (ie. nip_h, hip_h) to be within the mask's "height scale" :
  # mask_scaling is here because deeplab spits out segmaps of shape (513,288)
  mask_scaling  = float(masks_h)/float(orig_h)
  nip_h        *= mask_scaling
  nip_h         = int(round(nip_h))
  pr("nip_h is \n",nip_h)
  belly_button_h*= mask_scaling # TODO: separate scale variable for this (  "float(masks_h)/float(orig_h)" )
  belly_button_h = int(round(belly_button_h))
  pr("belly_button_h is \n",belly_button_h)
  hip_h        *= mask_scaling
  hip_h         = int(round(hip_h))
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
  pr("after adjustment, \n  nip_h is \n    ",nip_h)
  chest_l=np.count_nonzero( side_mask[nip_h])           # "length," but what this really means is distance from back to nipple.
  waist_l=np.count_nonzero( side_mask[belly_button_h])  # "length," but what this really means is distance from back to nipple.
  hip_l=np.count_nonzero( side_mask[hip_h])  # "length," but what this really means is distance from back to nipple.
  real_h_scale=cust_h/pix_height
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
def dist(pt1, pt2, norm='L2'):
  # This function "dist()" should be easily vectorizable.

  # points as np arrays
  return math.sqrt(np.sum(np.square(pt1-pt2)))
#===================================================================================================================================
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








#===================================================================================================================================
  # NOTE: what's this "b" at the beginning of the line?   b'f 6310 1331 4688\n'  from 'rb' in with open(fname, 'rb') as f:       guessing it means binary or something like that.
  # with open(fname, 'rb') as f:

#===================================================================================================================================
def obj_err(obj_fname, json_fname, front_fname, side_fname, cust_h):
  # TODO: finalize docstring once function is closer 2 finalized
  '''
    Finds how far the mesh in the .obj file from the real person's measurements.  ("error")

    Gets hip, waist, and chest circumferences from an .obj file describing a human body.

    0.  Get hip height from openpose json keypoints
      a.  Scale hip height to real height
    1. Get stuff to calculate circumferences
     -a.  Make sure mesh is in proper "z=up" position  (rotations/pltshow()/KDTree, etc.)
      0.  Get back from belly button 
      a.  First shift the model to the (+,+,+) octant.
      b.  Then scale the .obj mesh (model) to cust_h
      c.  Find points at heights hip, waist, chest
        1.  Is it worth the time complexity investment in a  KDTree?    Not octree; KDTrees sort more like distance-finders whereas octrees are more about partitioning the space (~BSTree but Cartesian)
          a.  Almost certainly; tflow and openpose are by far the most complex, time-consuming dependencies
        1st, set point at (0,0,hip_h); query the KDTree().query() to find the 1st pt
        2nd, set another pt at (x_max, y_max, hip_h), call KDTree.query(pt2) to find the last endpt
        3rd, set another pt at (    0, y_max, hip_h), call KDTree.query(pt3) to find the 3rd endpt
        4th, set another pt at (x_max,     0, hip_h), call KDTree.query(pt4) to find the last endpt
        This (4 pts) takes a few more queries than using a triangle, but it's prob slightly easier to understand and debug than trying to get an equilateral triangle and do it that way.
        Then u gotta figure out how to do directed graph search on 4 pts in R3.
      d.
      e.  Be able to visualize mesh so I can rotate nathan_mesh.obj to the correct orientation (where z=up) before I can find points at chest_height

    2.  Calculate circumference
      2.  Left  path from belly to back
        a.  A* search
      3.  Right path from belly to back
      4.  Connect those 2 paths (right and left) into a polygon
      5.  Elliptic curves would probably be more accurate than said polygon, but more complex to code.

    3.  Or simpler, calculate the polygon perimeter directly and subtract some constant for up-down shift
    4. 
 
    ------
    Params
    ------
    vs: vertices

  '''
  # yet TODO:
  '''
    0.  Get the RIGHT shape to take perim of 
      b0.  Given the K points, take only their (x,y) values (ie. set z=0), poly=CHull(pts), perim=perim_poly(hull_edge_pts), and use perim.
      b1.  Do we HAVE to do the ellipse?
      c. generalize the assert

      d.  Make sure mesh is in proper "z=up" position  (rotations/pltshow()/KDTree, etc.);  TODO: generalize c.  Find points at heights hip, waist, chest
      e.  Be able to visualize mesh so I can rotate nathan_mesh.obj to the correct orientation (where z=up) before I can find points at chest_height.  Hook render_smpl() up to arbitrary .obj (verts) mesh?
        Then u gotta figure out how to do directed graph search on 4 pts in R3.
      d.

    2.  Calculate circumference
      2.  Left  path from belly to back
        a.  A* search
      3.  Right path from belly to back
      4.  Connect those 2 paths (right and left) into a polygon
      5.  Elliptic curves would probably be more accurate than said polygon, but more complex to code.
      6.  Perhaps we could take the np.mean() of the quadrilateral() and jagged() paths.

    3.  Or simpler, calculate the polygon perimeter directly and subtract some constant for up-down shift
    4. 

  '''



  #obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_mesh.obj'
  # Load .obj file
  verts=[]
  faces=[]
  with open(obj_fname, 'r') as f:
    for line in f.readlines():
      if line.startswith('v'):
        verts.append(line)
      elif line.startswith('f'):
        faces.append(line)
  vs=np.zeros((len(verts), 3)).astype("float64") # means verts
  heights=np.zeros(len(verts)).astype("float64")
  X,Y,Z=1,2,3
  for idx,v in enumerate(verts):
    v=v.split(' ')
    vs[idx]=v[X],v[Y],v[Z]

  # NOTES:  I think currently y is  "height," x is "width," and z is "depth"
  #               but we want z     "height," x    "width," and y is "depth"     (helpfully, this is ALSO how blender does it)
  #           This yz_swap solution below: (Wed Mar  6 13:49:35 EST 2019) is specifically tailored to:
  #             obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_mesh.obj'
  yz_swap=np.array([[   1,   0,   0],
                    [   0,   0,   1],
                    [   0,  -1,   0]]).astype('float64')
                    # NOTE: this 
  # TODO: somehow ensure this transformation doesn't turn our mesh "upside down."  Maybe use pltshow() combined with the cKDTree.  
  #   Funny, for the "approx mask" operation we'd really like to have that KDTree() "all-neighbors queries functionality".  https://stackoverflow.com/questions/6931209/difference-between-scipy-spatial-kdtree-and-scipy-spatial-ckdtree

  # Simple transformations:
  # Rotate
  vs=vs.dot(yz_swap)
  vs=to_1st_octant(vs)
  x_max = np.max(vs[:,0]); x_min = np.min(vs[:,0]); y_max = np.max(vs[:,1]); y_min = np.min(vs[:,1]); z_max = np.max(vs[:,2]); z_min = np.min(vs[:,2])
  x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
  pn(3); pr("AFTER ROTATION,  ======================================================================= : ");pn()
  pr("x_max:",x_max); pr("x_min:",x_min); pr("y_max:",y_max); pr("y_min:",y_min); pr("z_max:",z_max); pr("z_min:",z_min);pn(2)
  pr("x_len:",x_len); pr("y_len:",y_len); pr("z_len:",z_len);pn() 
  pn(9)
  #pr("vs.shape:",vs.shape) # (~6000,3)  ie. (BIG,3)
  # no good if in "Jesus pose" and wingspan is longer than height; we just want +z to be "up."
  assert z_len > y_len and z_len > x_len  # TODO remove this assert ...

  # Scale to real-life-sizes (inches):
  vs     = vs * cust_h / z_max
  x_max = np.max(vs[:,0]); y_max = np.max(vs[:,1]); z_max = np.max(vs[:,2])
  x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
  pn(3); pr("AFTER RESCALING, ======================================================================= : ");pn()
  pr("x_max:",x_max); pr("x_min:",x_min); pr("y_max:",y_max); pr("y_min:",y_min); pr("z_max:",z_max); pr("z_min:",z_min);pn(2)
  pr("x_len:",x_len); pr("y_len:",y_len); pr("z_len:",z_len);pn() 
  pn(9)
  '''
  vt=cKDTree(vs, copy_data=True) # TODo: make sure these names are consistent (ie. either all short like v_t or all descriptive like vert_tree)
  # Note: Testing whether model is oriented correctly:     It seems it is.
  cube=np.array([ [      0,      0,      0,],
                  [      0,      0, z_max ,],
                  [      0, y_max ,      0,],
                  [      0, y_max , z_max ,],
                  [ x_max ,      0,      0,],
                  [ x_max ,      0, z_max ,],
                  [ x_max , y_max ,      0,],
                  [ x_max , y_max , z_max ,],]).astype("float64")
  p0=vt.data[vt.query(cube[0])[1]]; p1=vt.data[vt.query(cube[1])[1]]; p2=vt.data[vt.query(cube[2])[1]]; p3=vt.data[vt.query(cube[3])[1]]; p4=vt.data[vt.query(cube[4])[1]]; p5=vt.data[vt.query(cube[5])[1]]; p6=vt.data[vt.query(cube[6])[1]]; p7=vt.data[vt.query(cube[7])[1]]

  #                                   RESULTS:
  pr("p0:",p0)                          # p0: [ 9.85292562 18.25485839  8.2885409 ]
  pr("p1:",p1)                          # p1: [24.11407082  7.41028286 72.78951963]
  pr("p2:",p2)                          # p2: [ 0.45696223 29.51314676  0.21197051]
  pr("p3:",p3)                          # p3: [25.9614062  14.3200455  74.71568264]
  pr("p4:",p4)                          # p4: [55.20888442 19.38908581 11.33892357]
  pr("p5:",p5)                          # p5: [30.59125137  6.16716044 72.10696399]
  pr("p6:",p6)                          # p6: [60.91727507 25.42253024  8.94314535]
  pr("p7:",p7)                          # p7: [29.14528445 15.35900081 74.65673826]
  '''

  vt=cKDTree(vs, copy_data=True) # TODo: make sure these names are consistent (ie. either all short like v_t or all descriptive like vert_tree)


  # Use openpose to get other measurements
  measures=measure_body_viz(json_fname, front_fname, side_fname, cust_h)
  pr('measures: ')
  p(measures); pn(3)
  chest_h=measures['chest_height_inches']

  # Todo: refactor decision: 0 vs. (x_min or y_min)?
  quad=np.array([ [      0,      0,chest_h,],
                  [      0, x_max ,chest_h,],
                  [ x_max , y_max ,chest_h,],
                  [ x_max ,      0,chest_h,],]).astype("float64")
  # NOTE: I chose to put the points of "quad" in this weird, illogical order b/c it makes calculating approximate perimeter easier later
  K=1; IDX=1
  pt0=vt.data[vt.query(quad[0])[IDX]]; pt1=vt.data[vt.query(quad[1])[IDX]]; pt2=vt.data[vt.query(quad[2])[IDX]]; pt3=vt.data[vt.query(quad[3])[IDX]]
  pn(69); pr("pt0:",pt0); pr("pt1:",pt1); pr("pt2:",pt2); pr("pt3:",pt3)
  calced_chest=dist(pt0,pt1)+dist(pt1,pt2)+dist(pt1,pt2)+dist(pt2,pt3)
  pr("Chest circumference in inches: ",calced_chest); pr("chest_h:",chest_h)
  err = calced_chest-measures['chest_circum_inches']
  #                  pt0: [21.74963849 11.08709837 46.94066765]
  #                  pt1: [23.13935033 16.36705111 47.54809897]
  #                  pt2: [32.86014707 13.87639756 46.46942123]
  #                  pt3: [31.42370296  6.02317215 58.62576271]


  # Stretch vertices in z direction:
  # Note: The bigger STRETCH is, the more likely we are to get only verts at the same z value.
  STRETCH=100.; Z=2
  svs=deepcopy(vs)
  svs[:,Z]*=STRETCH
  chest_h*=STRETCH
  #pn(9); pr("chest_h: ",chest_h); pr("STRETCH: ",STRETCH)
  svt=cKDTree(svs, copy_data=True) # TODo: make sure these names are consistent (ie. either all short like v_t or all descriptive like vert_tree)
  #pr(" PREVIOUSLY    :");pr("x_max: ",x_max); pr("y_max: ",y_max); pr("z_max: ",z_max); pn(9)
  x_max = np.max(svs[:,0]); y_max = np.max(svs[:,1]); z_max = np.max(svs[:,2])
  x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
  pr(" AFTER STRETCH :");pr("x_max: ",x_max); pr("y_max: ",y_max); pr("z_max: ",z_max); pn(9)
  s_quad=np.array([ [      0,      0,chest_h,],
                    [      0, x_max ,chest_h,],
                    [ x_max , y_max ,chest_h,],
                    [ x_max ,      0,chest_h,],]).astype("float64")
  K=70 # if K is higher, we get more of the potential polygon points.  But we also get more noise
  pt0=svt.data[svt.query(s_quad[0])[IDX]]; pt1=svt.data[svt.query(s_quad[1])[IDX]]; pt2=svt.data[svt.query(s_quad[2])[IDX]]; pt3=svt.data[svt.query(s_quad[3])[IDX]]
  pts_near_chest0=svt.data[svt.query(pt0,k=K)[1]] # Todo: rename
  pts_near_chest1=svt.data[svt.query(pt1,k=K)[1]]
  pts_near_chest2=svt.data[svt.query(pt2,k=K)[1]]
  pts_near_chest3=svt.data[svt.query(pt3,k=K)[1]]
  pts_near_chest=np.unique(np.concatenate((pts_near_chest0,pts_near_chest1,pts_near_chest2,pts_near_chest3),axis=0),axis=0)
  pr("pts_near_chest.shape:  ", pts_near_chest.shape)
  pr(K*4)                     # 80 total pts
  pr(pts_near_chest.shape[0]) # 24 unique pts
  pr("pts_near_chest:\n ",pts_near_chest);pn(3)
  pn(9); pr("pt0:",pt0); pr("pt1:",pt1); pr("pt2:",pt2); pr("pt3:",pt3)
  calced_chest=perim_poly((pt0,pt1,pt2,pt3))
  #pts_near_chest=pts_near_chest[] # should be only the xy pts (project the chest polygon onto the z=0 plane)
  pr("pts:\n{0}\n{1}\n{2}\n{3}".format(pt0,pt1,pt2,pt3))
  xy_pts=pts_near_chest[:,:pts_near_chest.shape[1]-1] # shape (n,2)
  hull=ConvHull(xy_pts)
  vertices = hull.vertices.tolist() + [hull.vertices[0]] # Todo: shorten
  pe(69);pn(9);pr("xy vertices: \n",xy_pts[vertices]);pn(9);pe(69)
  hull_edgepts=xy_pts[vertices]
  pr('hull_edgepts: \n',hull_edgepts)
  pr('hull_edgepts.shape:',hull_edgepts.shape)
  X=0; Y=1
  plt.scatter(hull_edgepts[:,X],hull_edgepts[:,Y]); plt.show()
  perim     = np.sum([euclidean(x, y) for x, y in zip(hull_edgepts, hull_edgepts[1:])])
  calced_chest=perim

  pr("calced_chest (wrong coords):",calced_chest); pn(3)

  #shrink_chest_again("pts0-3")
  pt0[2]/=STRETCH; pt1[2]/=STRETCH; pt2[2]/=STRETCH; pt3[2]/=STRETCH
  pr("pts:\n{0}\n{1}\n{2}\n{3}".format(pt0,pt1,pt2,pt3))
  calced_chest=dist(pt0,pt1)+dist(pt1,pt2)+dist(pt2,pt3)+dist(pt3,pt0)
  pr("calced_chest (RIGHT coords),   calculated with dist() manually:",calced_chest)
  # In the below line, I use the tuple (pt0,pt1,pt2,pt3) b/c immutability is always a good thing:
  calced_chest=perim_poly((pt0,pt1,pt2,pt3))
  # in the degenerate case (STRETCH is tooooo big), we get pt0==pt1==pt2==pt3.  Before then, we get a line (pt0==pt1 and pt2==pt3).
  # solution: stretch once 2 find the right quadrilateral, then do the A* search after a bigger stretch
  # easier solution: try to leave it with fairly small value of STRETCH.
  pr("calced_chest (RIGHT coords),   calculated with perim_poly():",calced_chest)
  '''
  pts:
  [  24.78644235   10.96347829 5032.98328219]
  [  24.0617395    13.36335661 5040.39916411]
  [  28.02287236   12.0112951  5030.3883187 ]
  [  29.74478009    6.43362254 5036.59004338]
  calced_chest (wrong coords): 38.04621572659287

  pts:
  [24.78644235 10.96347829 55.92203647]
  [24.0617395  13.36335661 56.00443516]
  [28.02287236 12.0112951  55.89320354]
  [29.74478009  6.43362254 55.96211159]
  calced_chest (RIGHT coords): 19.249215525477638        WAYYYYYYYYY too low
  '''

  # NOTES: went down to 38.04621572659287   (b4 STRETCH, was 40.22217968662046).  As of commit [TODO: insert commit number], the chest calculation is 27.196557168098682.   The real (empirical) measurement for Nathan's chest_circum is 34.575946247110544 inches.   Hm.... I was thinking it ought to be SMALLER than the real measurement.  I think it's because the STRETCH increases the magnitude of chest_circum enough to cancel out the decrease from it being quad instead of ~elliptical
  #                  pt0: [  24.78644235   10.96347829 5032.98328219]
  #                  pt1: [  24.0617395    13.36335661 5040.39916411]
  #                  pt2: [  28.02287236   12.0112951  5030.3883187 ]
  #                  pt3: [  29.74478009    6.43362254 5036.59004338]
  # Todo:

  pr("Chest circumference in inches: ", calced_chest)

  # TODO: make "err" calculation much much much more comprehensive
  err = calced_chest-measures['chest_circum_inches']
  pn(29)
  return err, measures, vs
#=========================== end obj_err() ==================

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
def shift_obj(v, del_x, del_y, del_z):
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
  # more like "move_2_origin(v)"
  '''
    v = vertices
  '''
  return shift_obj(v, -np.min(v[:,0]), -np.min(v[:,1]), -np.min(v[:,2]))
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
#===================================================================================================================================














#===================================================================================================================================
if __name__=="__main__":
  NATHANS_HEIGHT_INCHES=75
  json_fname  = '/home/n/Dropbox/vr_mall_backup/json_imgs_openpose/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____keypoints.json'
  side_fname  = '/home/n/Dropbox/vr_mall_backup/imgs/n8_side___jesus_pose_legs_closed/n8_side___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  front_fname = '/home/n/Dropbox/vr_mall_backup/imgs/n8_front___jesus_legs_closed/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE___.jpg'
  obj_fname='/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_mesh.obj'; pr("obj_fname: ",obj_fname)
  err,measures,vs=obj_err(obj_fname, json_fname, front_fname, side_fname,NATHANS_HEIGHT_INCHES)
  #pr(obj_err(obj_fname, json_fname, front_fname, side_fname,NATHANS_HEIGHT_INCHES)) #if __name__=="__main__": 
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
  Function definitions (function headers)

  As of Fri Mar  1 06:53:14 EST 2019,
    115: load_json(json_fname):
    120: measure(json_fname):
    123: parse_ppl_measures(json_dict):
    159: segments(polygon):
    165: area_polygon(polygon):
    199: estim8_w8(json_fname, front_fname, side_fname, height):
    225: chest_circum(json_fname, front_fname, side_fname, cust_h):
    340: fac(n):
    351: half_c_n(n):
    363: sequence(n,h):
    369: series(n,h):
    375: perim_e(a, b, precision=6):
    438: ellipse_circum_approx(a, b, precision=6):
    445: ellipse_circum(a, b):
    497: measure_chest(json_fname):
    518: show_overlaid_polygon_measures(pic_filename___with_openpose_keypoints_, openpose_keypts_dict, N=4):
    544: dist(pt1, pt2):
    548: pixel_height(mask):
    552: pix_h(mask):
    571: orig_pix_h(img_fname):
    598: test_chest_circ():


  Chest
    Measure the circumference of your chest at your nipple level. Hold the end of the measuring tape in the middle of your chest at your nipple level. Evenly wrap the measuring tape around your chest. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.

  Waist
    Measure the circumference of your waist at your preferred waistline. Your preferred waistline is where you typically wear the waist of your pants. After exhaling, hold the beginning of the measuring tape in front of your body in the middle of your waistline. Evenly wrap the measuring tape around your waist. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.

  Hips
    Measure the circumference of your hips at the level where your hips are widest. Hold the beginning of the measuring tape in front of your body in the middle of your hip line. Evenly wrap the measuring tape around your hips. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.

  Inseam
    Measure your inseam from your crotch to the floor. Your crotch is the inner, uppermost point of leg. Hold the beginning of the tape measure at your crotch. Make sure you are holding it at the 0 mark. Pull the tape measure down to the floor where your foot is situated. Note the measurement at the point where the tape measure meets at the 0 mark.
    Note: When taking your measurements, relax your muscles and stand with weight equally distributed on both feet. Make sure that the measuring tape is kept at an even horizontal level around your body.

'''
