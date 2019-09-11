'''
#================================================================================
# From SMPLifyX directory:
#================================================================================



#================================================================================
# The following imports were blindly copied from main.py:
#   ( /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplifyx/smplify-x/smplifyx/main.py )
#================================================================================
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config    # this might be the only import I need.  
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False


#================================================================================
# Imports from fit_single_frame.py:
#================================================================================
try:
    import cPickle as pickle
except ImportError:
    import pickle
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
import cv2
import PIL.Image as pil_img

from optimizers import optim_factory
import numpy as np
import fitting
from human_body_prior.tools.model_loader import load_vposer





'''
#================================================================================
# My "render()" imports:
#   (nxb;   September 10, 2019)
#================================================================================
import pyrender ; pyr=pyrender;   pr=pyrender
import trimesh  ; tri=trimesh;    tm=trimesh

import time
import numpy as np
import math
from math import sin, cos
import matplotlib.pyplot as plt
# NOTE:   mpl.use('Agg')  'TkAgg', 'wxagg',  ?  TkAgg?  'mixed', 'agg', 'cairo', 'gtk3agg', 'gtk3cairo', 'nbagg', 'pdf', 'pgf', 'ps', 'qt4agg', 'qt4cairo', 'qt5agg', 'qt5cairo', 'svg', 'tkagg', 'webagg', 'wxagg',   (for the most recent list of possible backends for matplotlib, see:  https://matplotlib.org/3.1.1/api/index_backend_api.html)

#=================================================
def pn(n=0) : print('\n'*n)
#=================================================

#=================================================
def pe(n=89): print('='*n)
#=================================================

#=================================================
def print_dict_recurs(d, indent_lvl=2):
    for k,v in d.items():
        print (('  ')*indent_lvl+'within key '+str(k)+': ')
        if type(v)==type({}) or type(v)==type(OrderedDict()):
            print_dict_recurs(v, indent_lvl+1)
        elif type(v)==type([]):
            print_list_recurs(v, indent_lvl+1)
        else:
            print (('  ')*indent_lvl+'  value in dict: '+str(v))
#=================================================

#=================================================
def print_list(l):
    print_list_recurs(l, 0)
#=================================================
def print_list_recurs(l, indent_lvl):
    print (('  ')*indent_lvl+'printing list')
    for e in l:
        if type(e)==type({}) or type(e)==type(OrderedDict()):
            print_dict_recurs(e, indent_lvl+1)
        elif type(e)==type([]):
            print_list_recurs(e, indent_lvl+1)
        else:
            print (('  ')*indent_lvl+'  element in list: '+str(e))
#=================================================
 











#=========================================================
#=========================================================
#=========================================================
#=========================================================
#=========================================================
def prod():
  # TODO:   production code.
  # TODO:   production code.
  # TODO:   production code.
  # TODO:   production code.
  pass
#=========================================================
#=========================================================
#=========================================================
#=========================================================
#=========================================================








#=========================================================
def obj_2_png_trial(fname, png_path, filetype='obj'):
  mesh  = trimesh.load(fname)
  mesh  = pyrender.Mesh.from_trimesh(mesh)
  scene = pyrender.Scene()
  scene.add(mesh)
  # TODO===================================================================================================
  # TODO===================================================================================================
  # TODO===================================================================================================
  # TODO===================================================================================================
  #                         rot8  (45 around y,       111 around x?)
  # TODO===================================================================================================
  # TODO===================================================================================================
  # TODO===================================================================================================
  # TODO===================================================================================================

  # Camera:
  camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
  s = np.sqrt(2)/2
  '''
  camera_pose = np.array([
    [0.0, -s ,  s , 0.3   ],          # 0.3 moves us towards the butt (increase this number => "backwards" / "towards butt instead of belly" ("dorsal," I **think**))
    [1.0, 0.0, 0.0, 0.0   ],          #+0.0 moves us towards the mesh's head. (increase this number => "up" / "towards head" ("cranial," I **think**))
    [0.0,  s ,  s , 0.35  ],          #+0.0 moves us towards the mesh's head. (increase this number => "up" / "towards head" ("cranial," I **think**))
    [0.0, 0.0, 0.0, 1.0   ], ])
  '''
  #'''
  camera_pose = np.array([          #                                 camera_pose = np.array([         
    [0.0, -1.*s, 1.0*s, 0.3      ], #    # changes the lighting:        [0.0, -1*s, 0.5*s, 0.3      ], 
    [1.0, 0.0, 0.0,     0.5      ], #                                   [1.0, 0.0, 0.0, 0.0   ],
    [0.0, s, s,         0.35     ], #                                   [0.0, s, s, 0.35      ],
    [0.0, 0.0, 0.0,     1.0],    ]) #                                   [0.0, 0.0, 0.0, 1.0   ], ])       
  #'''    # https://en.wikipedia.org/wiki/Camera_matrix
  '''
  camera_pose = np.array([        # ORIGINAL.
    [0.0, -s, s, 0.3      ],
    [1.0, 0.0, 0.0, 0.0   ],
    [0.0, s, s, 0.35      ],
    [0.0, 0.0, 0.0, 1.0   ], ])
  '''
  light_pose  = np.array([        # ORIGINAL.
    [0.0, -s, s, 0.3      ],
    [1.0, 0.0, 0.0, 0.0   ],
    [0.0, s, s, 0.35      ],
    [0.0, 0.0, 0.0, 1.0   ], ])
  scene.add(camera, pose=camera_pose)

  # Lighting:
  light = pyrender.SpotLight(
    color=np.ones(3),  # white
    intensity=3.0, 
    innerConeAngle=np.pi/16.0,
    outerConeAngle=np.pi/6.0) # end `light = pr.SpotLight()`
  scene.add(light, pose=light_pose)
  # view()   (like "plt.show()" )
  pyrender.Viewer(scene, use_raymond_lighting=True)

  # render & save:
  r = pyrender.OffscreenRenderer(400, 400)
  color, depth = r.render(scene)
  plt.figure()
  plt.axis('off')
  plt.imshow(color)
  plt.savefig(png_path)
#=========================================================

#=========================================================
def obj_2_png(in_fname, out_fname, filetype='obj'):
  '''
  ===================================================================================================
    Saves render.png of     the given mesh.obj.
  ===================================================================================================
    ONLY WORKS on **VERY SIMPLE** .obj files that have all the 
      lines describing vertices ("v 0.01028526 0.88941276 0.04215282")
        BEFORE
      lines describing faces ("f 1 2 3")
  ===================================================================================================
  '''
  obj_2_png_trial(in_fname, out_fname, filetype='obj')
  """
  mesh=trimesh.load(fname)
  mesh = pyrender.Mesh.from_trimesh(mesh)
  scene = pyrender.Scene()
  scene.add(mesh)
  """
  #pyrender.Viewer(scene, use_raymond_lighting=True)   # this is just to show the coder (nxb) what the scene looks like.

  #return png_fname
#=========================================================


#=========================================================
def obj_2_rot8d_mesh(obj_fname, angle, angle_mode='degrees', axis='x'):
#=========================================================
  '''
    y 45 degrees    and 
        x 90?
    and z 90?
  '''
  # TODO:  extend to .ply files, other kinds of mesh files
  vfdict  = verts_and_faces_____from_fname(obj_fname, filetype='obj')
  v_matrix=np.array(vfdict['verts'])
  if angle_mode.lower() ==  'degrees':
    a = ( angle * math.pi)  / 180.
  else: #  mode         ==  'radians'
    a =   angle
  if    axis.upper()=='X':
    rot=np.array([
      [1.       ,     0.,      0.],
      [0.       , cos(a), -sin(a)],
      [0.       , sin(a),  cos(a)]])
  elif  axis.upper()=='Y':
    rot=np.array([
      [ cos(a)  ,     0.,  sin(a)],
      [ 0.      ,     1.,      0.],
      [-sin(a)  ,     0.,  cos(a)]])
  elif  axis.upper()=='Z':
    rot=np.array([
      [ cos(a)  ,-sin(a),      0.],
      [ sin(a)  , cos(a),      0.],
      [     0.  ,     0.,      1.]])
  else:
    raise Exception("Sorry, please specify a valid axis (x, y, or z)")

  rot8d_vs=(  rot.dot(v_matrix.T)  ).T
  return {
    'verts':  rot8d_vs ,
    'faces':  vfdict['faces']
  }
#=========================================================

#=========================================================
def write_customer_ready_mesh(input_obj_fname, out_obj_fname, angle=0.0, angle_mode='degrees', axis='x'):
  vf_dict=obj_2_rot8d_mesh(input_obj_fname, angle=angle, angle_mode=angle_mode, axis=axis)
  write_obj(out_obj_fname, vf_dict['verts'], vf_dict['faces'])
#=========================================================

#=========================================================
def write_obj(out_fname, verts, faces):
  '''
    Parameters:
      'verts':    I'm expecting 'verts' to be a numpy array of shape (n, 3)
      'faces':    list of lists, 0-indexed
  '''
  # TODO: error checking for filenames
  with open(out_fname, 'w+') as fp:
    for v in verts:
      fp.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
    for f in faces:
      f=np.array(f)+1 # 0-indexed to 1-indexed.
      fp.write('f {0} {1} {2}\n'.format(f[0], f[1], f[2]))
#=========================================================






#=========================================================
def verts_and_faces_____from_fname(fname, filetype='obj'):
  '''
  ===================================================================================================
    Returns dict of Vertices and faces
  ===================================================================================================
    ONLY WORKS on **VERY SIMPLE** .obj files that have all the 
      lines describing vertices ("v 0.01028526 0.88941276 0.04215282")
        BEFORE
      lines describing faces ("f 1 2 3")

    Returns:
      faces 0-indexed.
  ===================================================================================================
  ===================================================================================================
  '''
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  #==================================================================================================
  # NOTE:  this **might** be non-pythonic code:   (you should assume the user knows how to use the functions)
  assert type(filetype) ==  type("str")
  assert type(fname   ) ==  type("str")
  assert fname.endswith(filetype)

  with open(fname, 'r+') as fp:
    lines = fp.readlines()
  verts = []   # lines[:where_verts_end_and_faces_start]
  faces = []   # lines[where_verts_end_and_faces_start:]
  for i,line in enumerate(lines):
    vert_line = line.startswith('v')
    if vert_line:
      v=line.split(' ')[1:]      # "v 1.01 2.01 3.01" =====> ['1.01', '2.01', '3.01\n']
      # cut out the newline:
      v[-1] = v[-1][:-1]
      for vi,num in enumerate(v):
        v[vi] = float(num)
      verts.append(v)
    else: # face_line:
      f=line.split(' ')[1:]      # "f 1 2 3" =====> ['1', '2', '3\n']
      f[-1] = f[-1][:-1]
      for fi,num in enumerate(f): # ['1', '2', '3'] =====> [1, 2, 3]
        # ".obj" file indices start from **1**, **not** 0.  So to convert to 0, we have to subtract 1:
        f[fi] =  int(num)-1
      faces.append(f)
  return {
    'verts' : verts,
    'faces' : faces,
  }
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================























#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=="__main__":
  #=======================================
  #========== start timing: ==============
  #=======================================
  start = time.time()

  #nxb_reply='x 0'
  obj_filepath  = '/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes/000.obj'
  #obj_filepath  = '/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes/000_y_45.obj'
  outpath =\
    rot8d_obj_filepath=\
    '/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes/rot8d_000.obj'

  out_png_path  = '/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes/000.png'
  obj_2_png_trial(obj_filepath, out_png_path, filetype='obj')


  #===============================================================================================
  #===============================================================================================
  #===============================================================================================
  '''
  while not (nxb_reply=='Q'):
    # Parse:
    axis, angle = nxb_reply.split(" ")
    # Rot8():
    write_customer_ready_mesh(obj_filepath, outpath, angle=float(angle), axis=axis)

    # View():
    png_fname     = obj_2_png(rot8d_obj_filepath, filetype='obj')
    prompt='\n\nPlease reply "Q" if you want to quit. \nOtherwise, type an axis to rotate around, a space, and then an angle in degrees, like "x 45" or "z 60" \n\n'
    nxb_reply=input(prompt)
  #===============================================================================================
  #=========================== end loop   "while not (nxb_reply=='Q'):"===========================
  #===============================================================================================
    """
    #obj_2_verts_matrix(obj_filepath)
    s=obj_filepath
    outmesh_path= s[ 0            :s.rfind('/')+1 ]+\
      'rot8d_'  + s[s.rfind('/')+1:               ]
      # ie.   "/home/n/dir/name/rot8d_cust_mesh.obj"
    """
  '''



  #=======================================
  #           End Timing:
  #=======================================
  elapsed = time.time() - start
  time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
  print('Processing the data took: {}'.format(time_msg))










































































































































































































































































































  #==========================================
  #==========================================
  #==========================================
  #         from "main.py" :
  # (the full path (on nxb's laptop) is  :   
  #   "/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplifyx/smplify-x/smplifyx")
  #==========================================
  #==========================================
  #==========================================

  #==========================================
  # TODO:  initialize 'camera' parameters:
  #==========================================
  '''
  start = time.time()
  args=parse_config()
  output_folder = args.pop('output_folder')
  output_folder = osp.expandvars(output_folder)
  if not osp.exists(output_folder):
      os.makedirs(output_folder)

  # Store the arguments for the current experiment:
  conf_fn = osp.join(output_folder, 'conf.yaml')
  with open(conf_fn, 'w') as conf_file:
      yaml.dump(args, conf_file)
  #======================================
  # set variable "dtype:"   `dtype = ...`
  #======================================
  float_dtype = args['float_dtype']
  if float_dtype == 'float64':
      dtype = torch.float64
  elif float_dtype == 'float32':
      dtype = torch.float64
  else:
      print('Unknown float type {}, exiting!'.format(float_dtype))
      sys.exit(-1)

  #==========================================
  # Get faces and vertices of the mesh:
  #==========================================
  obj_filepath  = '/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes/000.obj'
  vf_dict       = verts_and_faces_____from_fname(obj_filepath)
  verts         = vf_dict['verts']
  faces         = vf_dict['faces']
  mesh          = tri.Trimesh(verts, faces, process=False)

  """
  ===================================================================================================
  ===================================================================================================
  ===================================================================================================
  ===================================================================================================
    nxb note,   [September 10, 2019]
      I'm abandoning this code.  I will leave it here as a reference for making pretty cameras work.
      Please consult the beautiful pyrender documentation before returning to this code:  [https://buildmedia.readthedocs.org/media/pdf/pyrender/latest/pyrender.pdf]
  ===================================================================================================
  ===================================================================================================
  ===================================================================================================
  ===================================================================================================
  """
  #==========================================
  # pyrender:    ()
  #==========================================
  material      = pyrender.MetallicRoughnessMaterial(
      metallicFactor=0.0,
      alphaMode='OPAQUE',
      baseColorFactor=(1.0, 1.0, 0.9, 1.0))
  mesh          = pyrender.Mesh.from_trimesh(
      mesh,
      material=material)
  # scene
  scene         = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                  ambient_light=(0.3, 0.3, 0.3))
  scene.add(mesh, 'mesh')

  #======================================
  # Create the camera object:
  #======================================
  focal_length = args.get('focal_length')
  camera = create_camera(focal_length_x=focal_length,
                         focal_length_y=focal_length,
                         dtype=dtype,
                         **args)
  camera=create_camera()

  #=====================================================================
  #=====================================================================
  #=====================================================================
  # Code after this point is in `fit_single_frame.py`    or **LATER**.
  #=====================================================================
  #=====================================================================
  #=====================================================================

  #=====================================================================
  #   A whole bunch of variables set in the  heading of "fit_single_frame.py" 's    function "def fit_single_frame()" :
  #       Probably most of them we don't need, but this is just the MVP and  I'm too lazy to go through and understand what each and every variable does.
  #=====================================================================
  result_fn='out.pkl'; mesh_fn='out.obj'; out_img_fn='overlay.png'; loss_type='smplify'; use_cuda=True; init_joints_idxs=(9, 12, 2, 5); use_face=True; use_hands=True; data_weights=None; body_pose_prior_weights=None; hand_pose_prior_weights=None; jaw_pose_prior_weights=None; shape_weights=None; expr_weights=None; hand_joints_weights=None; face_joints_weights=None; depth_loss_weight=1e2; interpenetration=True; coll_loss_weights=None; df_cone_height=0.5; penalize_outside=True; max_collisions=8; point2plane=False; part_segm_fn=''; focal_length=5000.; side_view_thsh=25.; rho=100; vposer_latent_dim=32; vposer_ckpt=''; use_joints_conf=False; interactive=True; visualize=False; save_meshes=True; degrees=None; batch_size=1; dtype=torch.float32; ign_part_pairs=None; left_shoulder_idx=2; right_shoulder_idx=5;
  W, H = 1024, 768
  #   These hardcoded width and height are from /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/_homogenus_____CLICKME/homogenus/samples/images/im1221.jpg
  #     (homogenus demo images, img1221).  
  #       I *THINK* it's from Leed Sports Dataset????  (AKA "LSD").  But I should double-check that at: https://www.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf
  args['visualize']=True
  with fitting.FittingMonitor(**args) as monitor:
    light_nodes = monitor.mv.viewer._create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)
  with torch.no_grad():
    camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5
  camera_center = camera.center.detach().cpu().numpy().squeeze()
  camera_transl = camera.translation.detach().cpu().numpy().squeeze()
  # Equivalent to 180 degrees around the y-axis. Transforms the fit to
  # OpenGL compatible coordinate system.
  camera_transl[0] *= -1.0

  camera_pose   = np.eye(4)
  camera_pose[:3, 3] = camera_transl

  camera        = pyrender.camera.IntrinsicsCamera(
      fx=focal_length, fy=focal_length,
      cx=camera_center[0], cy=camera_center[1])
  scene.add(camera, pose=camera_pose)


  r = pyrender.OffscreenRenderer(viewport_width=W,
                                 viewport_height=H,
                                 point_size=1.0)
  color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
  #==================================================
  #' '    could comment out.  idk if is important.
  #==================================================
  color = color.astype(np.float32) / 255.0
  #==================================================
  #' '
  #==================================================

  # plt show(pyrender_object):
  plt.figure()
  plt.imshow(color)
  plt.savefig('customer_render.png')
  plt.show()
  #color = color.astype(np.float32) / 255.0
  # Tell the code-monkey  how long rendering took.   (NXB is the only code-monkey as of   Sep 11  2019, at 01:18 A.M.. )
  elapsed = time.time() - start
  time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
  print('Processing the data took: {}'.format(time_msg))
  '''
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================























































