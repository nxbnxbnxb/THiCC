import json
import numpy as np
import math

# TODO: make the earlier json-generation via openpose automated end-to-end like this.  Everything must be MODULAR though

#shape=np.concatenate(trapezoid,np.mean(trapezoid,axis=0),axis='x')






































































#===================================================================================================================================
def load_json(json_fname):
  with open(json_fname) as json_data:
      data = json.load(json_data)
      return data
#===================================================================================================================================
def parse_ppl_measures(json_dict):
  '''
    NOTE: Extensible to whatever measurements we want later

    TODO: refactor
  '''
  measures=json_dict[u'people'][0][u'pose_keypoints_2d']
  measures_dict={}
  measures_dict["RShoulder"]= {'x':measures[2*3],'y':measures[7],'c':measures[8]}
  measures_dict["LShoulder"]= {'x':measures[5*3],'y':measures[16],'c':measures[17]}
  measures_dict["RHip"]     = {'x':measures[9*3],'y':measures[9*3+1],'c':measures[9*3+2]}
  measures_dict["LHip"]     = {'x':measures[12*3],'y':measures[12*3+1],'c':measures[12*3+2]}
  return measures_dict # NOTE: you gotta use the version of openpose I ran if you just return pose_keypoints_2d.
#===================================================================================================================================
def area_quadrilateral(quad):
  # https://stackoverflow.com/questions/1329546/whats-a-good-algorithm-for-calculating-the-area-of-a-quadrilateral
  # relies on quad being convex, but it always is for the chest of a person face-on
  triang0=quad[:3]
  triang1=quad[1:]
  print(quad)
  DOWN=0
  # vectors
  A=np.diff(triang0[:2],axis=DOWN)
  B=np.diff(triang0[1:],axis=DOWN)
  C=np.diff(triang1[:2],axis=DOWN)
  D=np.diff(triang1[1:],axis=DOWN)
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
  '''
  return np.cross(A-B,C-D)[0] # TODO: double-check that this indexing works.  also TODO: optimize if this is the time-consuming step
#===================================================================================================================================
def measure_chest(json_fname):
  '''
    Goal: we can automatically sell you the best-fitting shirt from a single picture of you in form-fitting clothing
    This should return the shirt size.
  '''
  json_fname='/home/n/Documents/code/openpose/output/front__nude__grassy_background_keypoints.json'
  measurements_json_dict=parse_ppl_measures(load_json(json_fname))
  x_LShoulder=measurements_json_dict['LShoulder']['x'];y_LShoulder=measurements_json_dict['LShoulder']['y']
  x_RShoulder=measurements_json_dict['RShoulder']['x'];y_RShoulder=measurements_json_dict['RShoulder']['y']
  x_LHip     =measurements_json_dict['LHip']['x']     ;y_LHip     =measurements_json_dict['LHip']['y']     
  x_RHip     =measurements_json_dict['RHip']['x']     ;y_RHip     =measurements_json_dict['RHip']['y']     
  quadrilateral=np.array([[x_LShoulder,y_LShoulder],
                          [x_RShoulder,y_RShoulder],
                          [x_LHip     ,y_LHip     ],
                          [x_RHip     ,y_RHip     ]]).astype('float64')
  chest_area_front= area_quadrilateral(quadrilateral)
  # TODO: correlate this with the shirt sizing.  Also, earlier in this process, we have to account for pixel-reality differences in the original images taken; pixel height needs to scale with height of the person
  return chest_area_front
#===================================================================================================================================
def triangles_area(x1,y1,x2,y2,x3,y3):
  '''
    The area of a triangle is equal to 1/2 * the magnitude of the cross product between 2 of its sides.
    The magnitude of the cross product is = len(side_1) * len(side_2) * sin(angle_btwn_side_1_and_side_2)
    Here I call "A" the length of the triangle's first  side,
                "B"                              second side,
            and "C" the length of triangle's     third  side
  '''
  A =math.sqrt((x1-x2)**2 + (y1-y2)**2)
  B =math.sqrt((x1-x3)**2 + (y1-y3)**2)
  C =math.sqrt((x2-x3)**2 + (y2-y3)**2)
  cos__theta_c = (A**2+B**2-C**2)/\
                 (2*A*B) # <== This is true by the law of cosines.  cos_theta_c means the angle opposite side C
  sin__theta_c = math.sqrt(1-cos__theta_c**2)
  return A*B*sin__theta_c/2.0
#===================================================================================================================================
if __name__=="__main__":
  fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/front__nude__grassy_background_keypoints.json'
  chest_area=measure_chest(fname)
  print("chest_area:",chest_area)
#===================================================================================================================================
































































