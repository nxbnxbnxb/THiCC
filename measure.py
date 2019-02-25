import json
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio as ii
import os

import viz
from viz import pltshow
from seg import segment_local as seg_local, segment_black_background as seg_black_back
from utils import pn
import sympy as s

# TODO: make the earlier JSON-generation via openpose automated end-to-end like this.  Everything must be MODULAR, though

#shape=np.concatenate(trapezoid,np.mean(trapezoid,axis=0),axis='x')




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
def chest_circum_overview(json_fname, front_img, side_img):
  # BIG PICTURE FUNCTION.  Will almost certainly not come out quite this cleanly; we have to do bottom-up programming too.  But this is the Platonic ideal of how it should look.
  '''
    circumference (perimeter) of body at the nipple level

    params:
    ------
    json_fname is keypoints from openpose
  '''
  # TODO: scale for actual real-world height.  See how to do this from working code previously written.
  front_mask    = seg_local(front_img)
  side_mask     = seg_local( side_img)
  torso_len     = hip_h - shoulder_h

  # nipple height is where we measure the chest, according to www.bodyvisualizer.com ("chest")
  nip_h         = shoulder_h + (torso_len*1./3.) # double check sign!   (depends whether hip_h or shoulder_h is bigger)   
  # NOTE;  no part of the customers' arms are at the same height ("y value") as the customer's nipples.
  chest_w=np.count_nonzero(front_mask[nip_h])
  chest_l=np.count_nonzero(side_mask[nip_h])  # "length," but what this really means is distance from back to nipple.
  return ellipse_perim(chest_w, chest_l)
#===================================================================================================================================
def chest_circum(json_fname, front_fname, side_fname):
  # PROBLEM: everyone's torso is a tad different.
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
  '''

  # TODO:
  '''
    0.  Clean up this function ("chest_circum()")
    1.  Deal with slight up-down shifts of the body during stepping rotation by shifting the whole mask up or down
      a.  ie. we have 1) frontal photo and 2) side photo.  Front mask starts higher than side mask, so we shift side mask "up" by np.min(front_mask)-np.min(side_mask)
    2.  Find armpits like I wrote the function find_crotch() to do in "old_measure.py"
      a. how?
    3.  Standardize the rotation pose for all get_measurements() code. (ie. all Jesus pose)
    4.  Write in the code for get_Ellipse_circum()
    5.  Test this function
    6.  F
    7.
    8.
    9.
    10.
    11.
    12.
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
      e.
      f.
    4.
    5.
    6.
    7.
    8.
    9.
    10.
    11.
    12.
  '''
  measurements  = measure(json_fname)

  # NOTE: because of how "y" is set up, Hip['y'] is larger than LShoulder['y'], despite the fact that the hip is below the shoulder in real life.
  shoulder_h    = measurements['Neck']['y']
  hip_h         = measurements['MidHip']['y']
  torso_len     = hip_h - shoulder_h
  print("torso_len: \n",torso_len)
  print("hip_h: \n",hip_h)
  print("shoulder_h: \n",shoulder_h)
  print("torso_len: \n",torso_len)
  # nipple height is where we measure the chest, according to www.bodyvisualizer.com ("chest")

  NIP_IN_TORSO  = 0.31460 #0.31460 is approximately 28/89, derived empirically from ONE image of myself 
  # previous values: 1./3., 2./5.
  nip_h         = shoulder_h + (torso_len*NIP_IN_TORSO) 
  pn(3)
  front_mask    = np.rot90(seg_local(front_fname))
  side_mask     = np.rot90(seg_local( side_fname))
  pn(3)
  #origs_height  = orig_pix_h(front_fname) # not necessary
  orig_h        = np.asarray(ii.imread(front_fname)).shape[0]
  masks_h       = front_mask.shape[0]
  print("masks_h:",masks_h)
  print("orig_h:",orig_h)
  nip_h        *= float(masks_h)/float(orig_h)
  nip_h         = int(round(nip_h))
  print("nip_h is \n",nip_h)
  # TODO: pltshow() in parallel (ie. like hmr)

  # show nipple identified
  cutout,segmap=seg_black_back(front_fname)
  WHITE=255
  cutout[nip_h]=WHITE 
  pltshow(cutout)   # success on /home/n/N.jpg

  # test nipple identified from side view
  cutout,segmap=seg_black_back(side_fname)
  cutout[nip_h]=WHITE 
  pltshow(cutout)
  # Data:
  #   For these particular images, the side view is 7 units shifted "up" from the   front view
  #   NOTE: we ought to identify a rotation point where from the side view the arms are directly at the customer's sides.

  CONST=10
  #pltshow(front_mask[nip_h-CONST:nip_h+CONST,:])
  #pltshow(front_mask[int(shoulder_h*float(masks_h)/float(orig_h)):int(hip_h*float(masks_h)/float(orig_h)),:])
  #pltshow(front_mask)
  print("front_mask.shape = \n",front_mask.shape)
  print("side_mask.shape = \n",side_mask.shape)
  pltshow( side_mask)

  # NOTE;  no part of the customers' arms are at the same height ("y value") as the customer's nipples.
  chest_w=np.count_nonzero(front_mask[nip_h]) # TODO; rename more descriptively?
  chest_l=np.count_nonzero( side_mask[nip_h])  # "length," but what this really means is distance from back to nipple.
  # ellipse circumference
  return ellipse_circum(chest_w/2., chest_l/2.)
  # TODO: try to catch all bugs before they get too serious
#================================================= chest_circum() ======================================================================

#=======================================================================================================================================
def ellipse_circum(a, b):
  '''
  '''
  #major and minor axes
  # https://en.wikipedia.org/wiki/Ellipse
  # https://stackoverflow.com/questions/22560342/calculate-an-integral-in-python
  # TODO: generalize this s.t.  the bigger of a and b gets assigned as a and  the smaller as b
  # TODO: finish this function and integrate it into chest_circum().
  # TODO: conda install sympy into env "cat"
  # TODO: finish this function and integrate it into chest_circum().

  # perhaps use an approximation of ellipse_circ() instead (https://www.mathsisfun.com/geometry/ellipse-perimeter.html 
  #                                                         and/or wikipedia.org/wiki/Ellipse)

  if b > a:
    tmp=a; a=b; b=tmp # swap s.t. a is always the semi-major axis

  e=math.sqrt(1-(b**2)/(a**2))
  theta=s.symbols('theta')
  return s.integrate(
    s.sqrt(
      1-\
      e**2*s.sin(theta)**2),
    (theta, 0, math.pi/2.))
  # rename oval_perim()?  slightly shorter and less intimidating
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
  print("img_w_keypts.shape: \n",img_w_keypts.shape)
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
def L2_dist(pt1, pt2):
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
def test_chest_circ():
  # NATHAN
  json_fname  = '/home/n/Dropbox/vr_mall_backup/IMPORTANT/front__nude__grassy_background_keypoints.json'
  #'/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'
  front_fname = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/front__nude__grassy_background.jpg'
  side_fname  = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/unclothed_outside_delaware_____uniform_background_with_wobbling/000000143.jpg'
  chest_circum(json_fname,front_fname,side_fname)
#===================================================================================================================================














#===================================================================================================================================
if __name__=="__main__":
  test_chest_circ()
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
  print("mask.shape:\n",mask.shape);print('\n'*1)
  height_in_pixels=pixel_height(mask)
  nathans_gender='male'
  gender=nathans_gender

  json_fname='/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'
  chest_area, other_body_measurements=measure_chest(json_fname) # TODO: change the measurements to shoulders,
  print("chest_area:",chest_area)
  LShoulder = np.array([ other_body_measurements['LShoulder']['x'],  other_body_measurements['LShoulder']['y']]).astype('float64')
  RShoulder = np.array([ other_body_measurements['RShoulder']['x'],  other_body_measurements['RShoulder']['y']]).astype('float64')
  LHip      = np.array([ other_body_measurements['LHip']['x']     ,  other_body_measurements['LHip']['y']]     ).astype('float64')
  RHip      = np.array([ other_body_measurements['RHip']['x']     ,  other_body_measurements['RHip']['y']]     ).astype('float64')
  print("LShoulder,RShoulder,LHip,RHip:",LShoulder,RShoulder,LHip,RHip)
  if gender.lower()=='female':
    s2h_ratio_const   = 1/3.
    # TODO:  find "zero point" for shoulders to hips for a female.
  else:
    s2h_ratio_const   = 2/5.  # TODO: empirically figure out what this s2h_ratio_const ought to be
    zero_beta__shoulder_2_hips=3.481943933038265
    # I've empirically derived that Nathan's shoulders_hips_diff___inches in that one picture (/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/front__nude__grassy_background.jpg) is 3.48.   This is for the variable assignment "shoulders_hips_diff___inches= (L2_dist(LShoulder,RShoulder) - L2_dist(LHip,RHip)) / orig_imgs_height_in_pixels * customer_height"

  # TODO: fuck with s2h_ratio_const until it's right
  shoulders_hips_diff___inches= (L2_dist(LShoulder,RShoulder) - L2_dist(LHip,RHip)) / orig_imgs_height_in_pixels * customer_height
  # NOTE: we have to make sure the image is well-cropped (consistently cropped) s.t. the customer height is a consistent fraction of the total image height.  TODO TODO!
  print("shoulders_hips_diff___inches:\n",shoulders_hips_diff___inches)
  beta_shoulders_hips = (shoulders_hips_diff___inches - zero_beta__shoulder_2_hips) * s2h_ratio_const 
  print("beta_shoulders_hips:    {0}".format(beta_shoulders_hips))
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
































































