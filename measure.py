import json
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio as ii
import os

import viz

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
if __name__=="__main__":
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
#===================================================================================================================================
































































