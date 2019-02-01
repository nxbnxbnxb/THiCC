import json
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio as ii

import viz

# TODO: make the earlier json-generation via openpose automated end-to-end like this.  Everything must be MODULAR though

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

    TODO: refactor.  As of Fri Feb  1 10:39:12 EST 2019, what did I mean by this?
  '''
  measures=json_dict[u'people'][0][u'pose_keypoints_2d']
  measures_dict={}
  measures_dict["RShoulder"]= {'x':measures[2*3], 'y':measures[7],      'c':measures[8]}
  measures_dict["LShoulder"]= {'x':measures[5*3], 'y':measures[16],     'c':measures[17]}
  measures_dict["RHip"]     = {'x':measures[9*3], 'y':measures[9*3+1],  'c':measures[9*3+2]}
  measures_dict["LHip"]     = {'x':measures[12*3],'y':measures[12*3+1], 'c':measures[12*3+2]}
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
    Calculates the polygon's area.  Works for concave polygons too

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
  json_fname='/home/n/Documents/code/openpose/output/front__nude__grassy_background_keypoints.json'
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
    left_shoulder   = [measurements['LShoulder']['x'], measurements['LShoulder']['y']]
    right_shoulder  = [measurements['RShoulder']['x'], measurements['RShoulder']['y']]
    left_hip        = [measurements['LHip']['x']     , measurements['LHip']['y']]
    right_hip       = [measurements['RHip']['x']     , measurements['RHip']['y']]
    plt.plot( left_shoulder[::-1],  right_shoulder[::-1], 'k-', lw=2) # across clavicle 
    plt.plot( right_hip[::-1],      right_shoulder[::-1], 'k-', lw=2) # down right side
    plt.plot( left_hip[::-1],       right_hip[::-1],      'k-', lw=2) # across waist 
    plt.plot( [measurements['LShoulder']['x'] , measurements['RShoulder']['x']],
              [measurements['LShoulder']['y'] , measurements['RShoulder']['y']],  'k-', lw=2) # across clavicle
    plt.plot( [measurements['RHip']['x']      , measurements['RShoulder']['x']],
              [measurements['RHip']['y']      , measurements['RShoulder']['y']],  'k-', lw=2) # down right side
    plt.plot( [measurements['RHip']['x']      , measurements['LHip']['x']],
              [measurements['RHip']['y']      , measurements['LHip']['y']],  'k-', lw=2)
    plt.plot( [measurements['RShoulder']['x'] , measurements['LHip']['x']],
              [measurements['RShoulder']['y'] , measurements['LHip']['y']],  'k-', lw=2)
    """
    left_shoulder   = [measurements['LShoulder']['x'], measurements['LShoulder']['y']]
    right_shoulder  = [measurements['RShoulder']['x'], measurements['RShoulder']['y']]
    left_hip        = [measurements['LHip']['x']     , measurements['LHip']['y']]
    right_hip       = [measurements['RHip']['x']     , measurements['RHip']['y']]
    print('left_shoulder:\n',left_shoulder)
    print('right_shoulder:\n',right_shoulder)
    print('left_hip:\n',left_hip)
    print('right_hip:\n',right_hip)
    plt.plot( left_shoulder[::-1],  right_shoulder[::-1], 'k-', lw=2) # across clavicle 
    plt.plot( right_hip[::-1],      right_shoulder[::-1], 'k-', lw=2) # down right side
    plt.plot( left_hip[::-1],       right_hip[::-1],      'k-', lw=2) # across waist 
    plt.plot( left_hip[::-1],       left_shoulder[::-1],  'k-', lw=2) # down left side
    """
  #plt.plot([(70,100),(70,250)],'k-',lw=5)

  # maybe it's the float -> int conversion.  Nope, it's not the float->int.  It was that xs had to come 1st in the plt.plot() parameter list and ys 2nd, (plt.plot([pt1x,pt2x], [pt1y,pt2y], 'k-', lw=2),    not plt.plot([pt1x,pt1y], [pt2x,pt2y], 'k-', lw=2))

  #plt.plot( left_hip, left_shoulder,        'k-', lw=2) # down left side
  #print("left_hip:\n{0}\n\nleft_shoulder:\n{1}".format(left_hip, left_shoulder))

  plt.show()
  plt.close()
  return

#===================================================================================================================================
if __name__=="__main__":
  json_fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'
  chest_area, other_body_measurements=measure_chest(json_fname)
  print("chest_area:",chest_area)

  # draw relevant polygon on top of image
  openpose_fname='/home/n/Documents/code/openpose/output/front__nude__grassy_background_rendered.jpg'
  #openpose_fname='/home/n/Documents/code/openpose/output/openpose_success!.jpg'
  show_overlaid_polygon_measures(openpose_fname, other_body_measurements)
#===================================================================================================================================
































































