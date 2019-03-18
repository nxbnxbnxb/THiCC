import numpy as np
import imageio as ii
import glob
import os
import sys
import json
from pprint import pprint as p
#from scipy.spatial import cKDTree

import viz
from viz import pltshow
from utils import pif, np_img, mask_info
from d import debug
from save import save 


"""
  Glossary (glossary):

  As of Fri Mar  1 06:53:14 EST 2019,
    115: load_json(json_fname):
    120: measure(json_fname):
    123: parse_ppl_measures(json_dict):
    159: segments(polygon):
    165: area_polygon(polygon):
    199: estim8_w8(json_fname, front_fname, side_fname, height):
    225: chest_circum(json_fname, front_fname, side_fname, cust_height):
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

  Glossary as of Mon Feb 25 09:07:23 EST 2019:
    def find_toes(mask___face_on):
    def find_crotch(mask___portrait_view):
    def get_waist(mask,customers_height):
    def pixel_height(mask):
    def measure_leg(crotch,toes):
    def leg_len(mask,customers_height):
"""
"""
  Point is, we have height and weight (from customer) and inseam
    TODO:
      Chest
      Waist
      Hips

      exercise???  (if we get this at all, it should just be by askign.  Althgouh people will not necessarily know the answer off the top of their heads, they will prooooooooooooobably be able to estimate REASONabblllly well.
"""









# TODO: extend these functions to get ALL measurements of the person (ie. wingspan, leg length, waist)
#     TODO: using iPhone measure() app in conjunction with the code in this module, automate height-retrieval in inches and size customers accordingly



pr=print
#==============================================================
def find_toes(mask___face_on):
  '''
    Doesn't do this perfectly; just an estimation (Thu Jan 17 11:22:14 EST 2019)

    Ideas to improve it if the result isn't precise enough:
      Add dist_from_crotch() as a negative weight 
        ie. 
          1. Try to find the toe that is both closest to the left corner and farthest from the crotch, and
          2. Find        the toe that is closest to the right corner and farthest from the crotch)
  '''
  #     minimum distance from bottom right corner (one toe)
  # and minimum distance from bottom left corner  (the other toe)
  mask        = mask___face_on
  bot_left    = np.array([mask.shape[0]-1,0               ])
  bot_right   = np.array([mask.shape[0]-1,mask.shape[1]-1 ])
  locs        = np.array(np.nonzero(mask)).T
  distances   = np.sum(np.sqrt(np.abs(locs-bot_left )),axis=1)
  lefts_idx   = np.argmin(distances)
  distances   = np.sum(np.sqrt(np.abs(locs-bot_right)),axis=1)
  rights_idx  = np.argmin(distances)
  return {'left_toe':locs[lefts_idx].reshape(1,2), 'right_toe':locs[rights_idx].reshape(1,2)}

#==============================================================
def find_crotch(mask___portrait_view):
  # Given a photo with the customer's legs spread, trace the inside of the left leg up until you find the crotch.
  '''
    As of Thu Jan 17 09:15:12 EST 2019,
      This func assumes 0 noise in the mask, which is an unrealistic assumption
      Also assumes we never get any "hooks" in the left leg, ie. 
                                                              /
                                                             /                               \   <--- right leg
                                                             ^                                \ 
                                                            / \ <---- this is the hook I mean  \ 
                                                           /
                                                          /
  '''
  # Given a photo with the customer's legs spread, trace the inside of the left leg up until you find the crotch.
  mask=mask___portrait_view
  locs=np.nonzero(mask)
  toe_height=np.amax(locs[0])
  both_feet_height=float('inf')
  # return "crotch" as dict at the end
  crotch={};crotch['x_loc']=float("inf");crotch['height']=float("inf")
  left_leg_inner_x=0 # assume the person is facing away from us;   then their left is our left
  for height in range(toe_height,0,-1):
    if both_feet_height!=float('inf'):
      break  # we've found the starting point from the left leg
    else: # both_feet_height will change values once we find the pt where we see both legs separately
      in_1st=False
      looking_for_2nd=False
      for x in range(mask.shape[1]):
        if not in_1st and mask[height,x]:
          in_1st=True
        elif in_1st and not mask[height,x] and not looking_for_2nd:
          left_leg_inner_x=x
          looking_for_2nd=True
        elif looking_for_2nd and mask[height,x]:
          both_feet_height=height
          break # out of inner (for x in range...) for loop, not the "for height in range(...)" loop
  while float('inf') in crotch.values():
    # NOTE: could do an infinite loop on the wrong input
    low=max(np.nonzero(mask[:,left_leg_inner_x])[0])
    should_be_lower=max(np.nonzero(mask[:,left_leg_inner_x+1])[0])
    if should_be_lower > low:
      crotch['x_loc']=left_leg_inner_x
      crotch['height']=low
      # will cause the break
    else:
      left_leg_inner_x+=1
  return crotch
  # NOTE:  should be at (280,150).  It was.  Jan 29, 2019.
#==============================================================
def get_waist(mask,customers_height):
  '''
    This function assumes there are no arms down at the waist height.  ie. Jesus pose, mTailor pose, one of those.
    Waist, not inseam, for men's jeans

    customers_height is in inches
  '''
  pix_height=pixel_height(mask)
  crotch=find_crotch(mask)
  crotch_height=crotch['height']
  crotch_x=crotch['x_loc']
  crotch=np.array([crotch['height'], crotch['x_loc']]).astype('int64')
  pix_btwn_waist_and_crotch=24 # TODO: fiddle with.
  waist_height=crotch_height-pix_btwn_waist_and_crotch
  waist_in_pixels=int(np.count_nonzero(mask[waist_height])) # NOTE: can modify this to get only the middle section (strip)'s length if both arms go down and are just as high as the waist
  print("waist_in_pixels is {0}".format(waist_in_pixels))
  print("crotch is {0}".format(crotch)) # fine
  print("pix_height is {0}".format(pix_height))
  print("customers_height is {0}".format(customers_height))
  pltshow(mask)
  pltshow(mask[waist_height-10:waist_height+10,:])
  if debug:
    pltshow(mask[crotch_height-10:crotch_height+10, crotch_x-10:crotch_x+10])
    pltshow(mask[waist_height-5:waist_height+5,:])
  return waist_in_pixels/pix_height*customers_height  # NOTE NOTE NOTE NOTE NOTE not right!
  # TODO: get the belly-waist from the side view and approximate the waist measurement as an ellipse.
  #       but how do we locate the waist given the side view?  Openpose could do it, but that's another dependency.  We could take the midpoint of head and toe, but again, pretty fragile.
#==============================================================
def pixel_height(mask):
  locs=np.nonzero(mask)
  return np.amax(locs[0])-np.amin(locs[0])
#==============================================================
#==============================================================
def measure_leg(crotch,toes):
  TWO_LEGS=2
  return np.sum(np.sqrt(np.sum(np.square(crotch-toes),axis=1)))/TWO_LEGS
#==============================================================
#==============================================================
def leg_len(mask,customers_height):
  '''
    customers_height is in inches
  '''
  pix_height=pixel_height(mask)
  #print("pix_height:\n{0}".format(pix_height))  # not the problem

  crotch=find_crotch(mask)
  crotch=np.array([crotch['height'], crotch['x_loc']]).astype('int64')
  toes  =find_toes  (mask)
  toes  =np.concatenate((toes['left_toe'],toes['right_toe']),axis=0)
  '''
  #print("toes:\n{0}".format(toes))
  #print("crotch:\n{0}".format(crotch))  # NOTE: toes and crotch were fine.  the issue was somewhere in between these lines and the end

    pix_height:
    330

    toes:
    [[423 120]
     [409 168]]

     crotch:
     [288 137]

     leg_len_pixels:
     16.154910013534966
  '''
  pif("crotch-toes:\n{0}".format(crotch-toes))
  leg_len_pixels=measure_leg(crotch,toes)
  pif("leg_len_pixels:\n{0}".format(leg_len_pixels))
  pif("\n"*3)
  return leg_len_pixels/pix_height*customers_height
#==============================================================
if __name__=="__main__":
  mask_fname='mask.png'
  mask=np_img(mask_fname)
  mask_data=mask_info(mask)
  pr("mask_data:");p(mask_data)
  pr("mask.shape:",mask.shape)
  mask_2d=np.logical_and(mask[:,:,0],mask[:,:,1])
  mask_2d=np.logical_and(mask_2d,mask[:,:,2])
  pr("mask_2d.shape:",mask_2d.shape)
  crotch=find_crotch(mask_2d)
  pr("crotch:")
  p(crotch)
  pltshow(mask_2d)
#==============================================================




  # NOTE:  esp. in the future, be wary of how important it is to use np.greater()  (segmentation doesn't just return simple "true-false")
  # whole directory of masks:
  """
  folder='/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/'
  mask_fnames= sorted(glob.glob(folder+'*'),key=os.path.getmtime)
  print(mask_fnames)
  fronts_ctr=0
  for mask_fname in mask_fnames:
    if mask_fname.endswith("jpg") or mask_fname.endswith("png") or mask_fname.endswith("jpeg"):
      print(find_crotch(np.greater(np.asarray(ii.imread(mask_fname)),127)))
    if fronts_ctr == 60: # 60 is experimentally derived from this particular set of masks
      break
    fronts_ctr+=1


  # single image:
  """



  '''
  mask_fname  = sys.argv[1]   #"/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg"
  # 32, 38, 39, and 43 find the crotch to be higher  than the "nearby" images
  mask        = np.asarray(ii.imread(mask_fname))
  mask=np.greater(mask,127)
  NATHAN_HEIGHT=75 # inches
  print("inseam estimation (length in inches):   {0}".format(leg_len(mask,NATHAN_HEIGHT)))
  '''
  # NOTE:  there are no real units here;  it's all just a ratio that is normalized to Nathan's height and pants length
  #"""
#==============================================================
  # for Nathan segmentation (/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/2019_01_15____11:04_AM___/000000000.jpg),
  #
  #                       29.65615073829731     **** short
  #
  #
  # for file "./masks/2019_01_17____17\:57_PM_______inseam_36___male__/000000000.jpg":  leg_len() was 30.77904443651943
  #
  #   See prefix above.  It ain't just 000000000.jpg
  #
  # 000000000.jpg         30.77904443651943
  # 000000001.jpg         30.384982686788234    **** short
  # 000000002.jpg         30.428061480054698
  # 000000003.jpg          7.355339015509475      crossed legs
  # 000000004.jpg         32.046654629942765    **** big
  # 000000005.jpg         31.32609217992018
  # 000000006.jpg         31.86657296827472
  # 000000007.jpg         30.800175451288585
  # 000000008.jpg         34.446067858908876    **** big
  # 000000009.jpg          5.712278750981283      crossed legs
  # 000000010.jpg         32.53770678534222
  # 000000011.jpg          7.482261612732713      crossed legs
  # 000000012.jpg          7.630327382130935      crossed legs
  # 000000013.jpg         31.989151381902385
  # 000000014.jpg          6.805627029964921      crossed legs
  # 000000015.jpg         32.964500571537876
  # 000000016.jpg         28.52023566021052     **** short because the guy's turning to the side so his inseam looks lower than it actually is
  # 000000017.jpg         29.261865975820115
  # 000000018.jpg         33.05316375925231
  # 000000019.jpg          5.640038864533751      crossed legs
  # 000000020.jpg         34.87838629196392
  # 000000021.jpg          5.653397472957198      crossed legs
  # 000000022.jpg          6.675191520628813      crossed legs
  # 000000023.jpg         34.87838629196392
  #
  #
  #
  #  NOTE: this doesn't use openpose; will probably be more dependent on the customer posing a certain way, dependent on the segmentation coming out right, etc.
  #    slightly less robust to 
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
 










































































# from measure.py as of Fri Mar  8 10:35:19 EST 2019:
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

  """
  vt=cKDTree(vs, copy_data=True) # TODo: make sure these names are consistent (ie. either all short like v_t or all descriptive like vert_tree)

  # Todo: refactor decision: 0 vs. (x_min or y_min)?
  quad=np.array([ [      0,      0,chest_h,],
                  [      0, y_max ,chest_h,],
                  [ x_max , y_max ,chest_h,],
                  [ x_max ,      0,chest_h,],]).astype("float64")
  # NOTE: I chose to put the points of "quad" in this weird, illogical order b/c it makes calculating approximate perimeter easier later
  pr("quad:\n{0}   \n\n(we're getting the endpoints of the chest cross-section from this)\n".format(quad))
  K=1; IDX=1
  #  quad:                                                                                      NO GOOD.     
  # [[ 0.          0.         55.90405904]          pt0: [21.74963849 11.08709837 46.94066765]
  #  [ 0.         62.4291962  55.90405904]    ==>   pt1: [23.13935033 16.36705111 47.54809897]
  #  [62.4291962  30.58900293 55.90405904]          pt2: [32.86014707 13.87639756 46.46942123]
  #  [62.4291962   0.         55.90405904]]         pt3: [31.42370296  6.02317215 58.62576271]


  # Stretch vertices in z direction:
  # Note: The bigger STRETCH is, the more likely we are to get only verts at the same z value.
  STRETCH=  5.; Z=2
  svs=deepcopy(vs)
  svs[:,Z]*=STRETCH
  pn(1); pr("chest_h: ",chest_h); pr("STRETCH: ",STRETCH) 
  chest_h*=STRETCH
  pn(1); pr("chest_h: ",chest_h); pr("STRETCH: ",STRETCH) 
  # orig_chest_h ==   55.9            inches
  #   chest_h:      5590.405904059041

  svt=cKDTree(svs, copy_data=True) # TODo: make sure these names are consistent (ie. either all short like v_t or all descriptive like vert_tree)
  #pr(" PREVIOUSLY    :");pr("x_max: ",x_max); pr("y_max: ",y_max); pr("z_max: ",z_max); pn(9)
  x_max = np.max(svs[:,0]); y_max = np.max(svs[:,1]); z_max = np.max(svs[:,2])
  x_len=x_max-x_min; y_len=y_max-y_min; z_len=z_max-z_min
  if debug:
    pr(" AFTER STRETCH :");pr("x_max: ",x_max); pr("y_max: ",y_max); pr("z_max: ",z_max); pn(9)
  s_quad=np.array([ [      0,      0,chest_h,],
                    [      0, y_max ,chest_h,],
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
  hull=ConvexHull(xy_pts) #scipy.spatial.
  vertices = hull.vertices.tolist() + [hull.vertices[0]] # Todo: shorten
  pe(69); pn(9); pr("xy vertices: \n",xy_pts[vertices]); pn(9); pe(69)
  perim_edgepts=xy_pts[vertices]
  pr('perim_edgepts: \n',perim_edgepts)
  pr('perim_edgepts.shape:',perim_edgepts.shape)
  X=0; Y=1
  plt.title("Nathan\'s Chest cross section:   ConvexHull")
  plt.scatter(perim_edgepts[:,X],perim_edgepts[:,Y]); plt.show()
  perim     = np.sum([euclidean(x, y) for x, y in zip(perim_edgepts, perim_edgepts[1:])])
  calced_chest=perim

  # NOTES: went down to 38.04621572659287   (b4 STRETCH, was 40.22217968662046).  As of commit 692fc87, the chest calculation is 27.196557168098682.   The real (empirical) measurement for Nathan's chest_circum is 34.575946247110544 inches.   Hm.... I was thinking it ought to be SMALLER than the real measurement.  I think it's because the STRETCH increases the magnitude of chest_circum enough to cancel out the decrease from it being quad instead of ~elliptical
  #                  pt0: [  24.78644235   10.96347829 5032.98328219]
  #                  pt1: [  24.0617395    13.36335661 5040.39916411]
  #                  pt2: [  28.02287236   12.0112951  5030.3883187 ]
  #                  pt3: [  29.74478009    6.43362254 5036.59004338]
  # Todo:


  pr("Chest circumference in inches: ", calced_chest)
  """






































































  """
  #Mon Mar 11 09:32:05 EDT 2019
    # finding the perimeter of the .obj mesh at height:
  intersecting_faces=[]
  for edge in intersecting_edges:
    v0_idx=edge[0]
    v1_idx=edge[1]
    adj_faces_0=adjs[v0_idx]
    for face in adj_faces_0:
      if v1_idx in face:
        intersecting_faces.append(face)
  print("len(faces): ",len(faces))
  print("len(intersecting_faces): ",len(intersecting_faces))
  """


  '''
  #=========================================
  def faces_at_height(faces, height, verts, window, which_ax, ax):
    # This func is BROKE.
    #   len(faces)     :  13776
    #   len(faces_at_h):  32400
    # fast
    out_faces=[]
    for face in faces:
      for v_idx in face:
        if verts[v_idx][ax] < height + window and verts[v_idx][ax] > height - window:
          out_faces.append(face)
    return out_faces
  '''

  """
  print("len(faces)     : ",len(faces))      # len(faces): 32400
  def face_hash_table(faces, height):
    for i,face in enumerate(faces):
      # TODO:  put the face indices in a dictionary and use it to vastly speed up faces_attached()
      pass
  def faces_attached(edges, faces, verts):
  # TODO: fix this.
  #  real  1m54.993s
  #  user  1m57.233s
  #  sys 0m1.181s

  # TODO: fix this.  OMG it got WORSE when I used faces_at_height()!  faces must be HUGE.  I'll have to do something about this tonight/tomorrow
  #print("len(faces_at_h):",len(faces_at_h)) # len(faces_at_h): 32400
  #real 3m54.254s
  #user  3m57.056s
  #sys 0m1.073s

    # The Brute-force implementation is DIRT SLOW
    face_list=[]
    for edge in edges:
      v0=edge[0]
      v1=edge[1]
      for face in faces:
        if v0 in face and v1 in face:
          face_list.append(face)
    return face_list
  #====== end faces_attached(params) =======
  #faces=faces_attached(intersecting_edges, faces_at_h, verts)    # SLOW



#Mon Mar 11 14:38:17 EDT 2019
  # calculate mesh_resolution:   (the area of each triangle in the mesh, on average):
  mesh_resolution=0 # should end up at 0.652 by the end
  for f in fs:
    v0=vs[f[0]].reshape((1,3))
    v1=vs[f[1]].reshape((1,3))
    v2=vs[f[2]].reshape((1,3))
    tri=np.concatenate((v0,v1,v2),axis=0).astype('float64')
    mesh_resolution+=tri_area(tri)
  mesh_resolution/=len(fs) # avg area of triangles in the body mesh.
  #pr("mesh_resolution:  ",mesh_resolution) # for HMR,  mesh_resolution was 0.0001195638271130602 .
  """

#===================================================================================================================================
  # NOTE: what's this "b" at the beginning of the line?   b'f 6310 1331 4688\n'  from 'rb' in with open(fname, 'rb') as f:       guessing it means binary or something like that.
  # with open(fname, 'rb') as f:

