'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''

from smpl_webuser.serialization import load_model

import numpy as np
import sys
import subprocess as sp
from math import pi
from copy import deepcopy
from pprint import pprint as p
#=========================================================================
pr=print
def pn(n=0): print('\n'*n)
def pe(n=89): print('='*n)
#=========================================================================

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
  return shift_verts(v, -np.min(v[:,0]), -np.min(v[:,1]), -np.min(v[:,2]))
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
  data= (x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len)
  return data
#===================================================================================================================================
def orient_mesh(params):
  '''
    Standardizes mesh position and orientation.  Return value "vs" ought to have the mesh's head "point" in the +z direction, every mesh vertex should have (+x,+y,+z) values
    Assumes input mesh is in T-pose.

  Pseudocode:
    1.  Find the "smallest" x_len, y_len, or z_len
      a.
    2.  Orient s.t. that smallest is in the y direction.
    3.  Then main body is near the center of its axis   (x_mid)
    4.  Whereas arms are near the end of their axis (z_arms ~= z_max OR    z_arms ~= z_min){math.isclose()}
    5.
    6.
    7.
    8.
    9.
    10.

    -------
    Params:
    -------
      verts are np arrays.  means vertices of a mesh.

    return value "vs" ought to have the mesh's head "point" in the +z direction, every mesh vertex should have (+x,+y,+z) values
  '''
  #func orient_mesh(params)
  X,Y,Z=0,1,2
  verts=params['mesh']['verts'] # verts.shape==(n,3)
  x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len= vert_info(verts)
  print("x_min: {0}\nx_max: {1}\ny_min: {2}\ny_max: {3}\nz_min: {4}\nz_max: {5}\nx_len: {6}\ny_len: {7}\nz_len: {8}\n".format(x_min,x_max,y_min,y_max,z_min,z_max,x_len,y_len,z_len))
  which=np.argmin((x_len,y_len,z_len))
  print("which:",which)
  def switch_axes(params):
    verts___n_x_3, ax1, ax2=params['verts'],params['ax1'],params['ax2'],
    swapped=deepcopy(verts___n_x_3)
    tmp=swapped[ax1]; swapped[ax1]=swapped[ax2]; swapped[ax2]=tmp
    return swapped
  
  if which != Y:
    switch_params={'verts':verts, 'ax1':which, 'ax2':Y}
    verts=switch_axes(switch_params)
  verts=to_1st_octant(verts)
  params=deepcopy(params)
  params['verts']=verts
  return params
  




































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
#===================================================================================================================================
def blender_render_mesh(params):
  '''
    I had to hack around the fact that subprocess.call() doesn't support cmd line args  ( or was it blender --python itself that doesn't?)
    So I rewrite a tiny python file with the argument supplied here and call blender with that.
  '''
  params=deepcopy(params)
  mesh_fname=params['mesh_fname']
  import_script="/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world/blender_import_obj.py"
  # overwrite
  with open(import_script, 'w') as fp:
    fp.write('import bpy\n')
    fp.write('obj_path=\''+mesh_fname+'\'\n')
    fp.write('bpy.ops.import_scene.obj(filepath = obj_path, split_mode = "OFF")\n')
  try:
    sp.call(['blender','--python', import_script])
  except:
    funcname=  sys._getframe().f_code.co_name
    raise(Exception('{0} call failed.  Sorry.  No mesh for you.'.format(funcname)))
  return params
#======================================= end func blender_render_mesh(params) ======================================================
def write_smpl(params):
  '''
    Writes a smpl mesh.  Tries to write the mesh s.t.  the person is facing "forward" in +y, head is "up" in +z, and left-right doesn't matter.
  '''
  funcname=  sys._getframe().f_code.co_name; pe();print('entering ',funcname);pe()
  m=params['model']
  gender=params['gender']

  ## Write to an .obj file
  outmesh_path = \
    './{10}_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(
      m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]),
      gender.lower())

  # TODO: write orient() more generally.
  with open( outmesh_path, 'w') as fp:
      for v in m.r:
          fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

      for f in m.f+1: # Faces are 1-based, not 0-based in obj files
          fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

  ## Print message
  print('..Output mesh saved to: ', outmesh_path) 
  params=deepcopy(params)
  params['mesh_fname']= outmesh_path
  return params
#====================================================== end func write_smpl(params) =============================================================================
def smpl(info):
  '''
    returns facing "forward" in y, head is "up" in z, and left-right doesn't matter.
  '''
  ## Load SMPL model
  info  = model(info)
  info  = write_smpl(info)
  info  = blender_render_mesh(info) # comment out when not debugging

  mesh_fname=info['mesh_fname']
  m=info['model']
  out= {
    'mesh':{
      'verts':m.r, # why couldn't they have just ***king named it "m.v" like normal people...  Clearly it's "vertices," right ???
      'faces':m.f, },
    'mesh_fname': mesh_fname,
  }
  return out
#======================================= end func smpl(params) =====================================================================





# TODO:   refactor whol module into single function   "def mesh(info):  betas=info['betas'];   mesh_fname=info['fname'];   betas=info['betas'];   betas=info['betas'];   betas=info[' betas'];      mesh=(verts,faces) return {'mesh':mesh, 'mesh_fname':mesh_fname, 'betas':betas}
# TODO:   refactor into "def mesh(info):  betas=info['betas'];   mesh_fname=info['fname'];   betas=info['betas'];   betas=info['betas'];   betas=info[' betas'];      mesh=(verts,faces) return {'mesh':mesh, 'mesh_fname':mesh_fname, 'betas':betas}
# TODO:   refactor into "def mesh(info):  betas=info['betas'];   mesh_fname=info['fname'];   betas=info['betas'];   betas=info['betas'];   betas=info[' betas'];      mesh=(verts,faces) return {'mesh':mesh, 'mesh_fname':mesh_fname, 'betas':betas}
# TODO:   refactor into "def mesh(info):  betas=info['betas'];   mesh_fname=info['fname'];   betas=info['betas'];   betas=info['betas'];   betas=info[' betas'];      mesh=(verts,faces) return {'mesh':mesh, 'mesh_fname':mesh_fname, 'betas':betas}

#===================================================================================================================================
def model(info):
  # load right model
  gender=info['gender']
  if gender.lower()=='male':
    ## NOTE: on a new system, double-triple-check to make sure path is correct (I renamed them)
    fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    m = load_model(fname)
  elif gender.lower()=='female':
    fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    m = load_model(fname)

  # shape
  # Note:     This is just a guess, but I bet whatever the hell MPI Tuebingen did to the data before they pickled it allowed us to set the betas once & only once.  -nxb, (comment written    Wed Mar 27 10:26:40 EDT 2019)
  m.betas[:] = info['betas']

  # pose
  m.pose[:] = np.zeros((m.pose.size)).astype('float64')
  ## Rotates (like a backflip)
  m.pose[0] =  pi  # Note: I'm p sure these rotations (I think it was pose[:3] that are the rotations) are taken care of differently in HMR.
  info=deepcopy(info)
  info['model']=m
  return info
#========================================= end func model(params) ==================================================================
#===================================================================================================================================
def get_betas():
  '''
    cmd line args.   If you wanna get 'em a diff way later (ie. reading from file), mutate in here.
  '''
  betas = np.zeros((10)).astype('float64')
  # it's best to store the betas in a temporary variable so when we assign everything, SMPL calculates EVERYTHING in one step
  for i in range(1,len(sys.argv)):
    betas[i-1]=float(sys.argv[i])
  return betas
#===================================================================================================================================
def main():
  gender = 'male'
  betas=get_betas() # cmd line -nxb (as of Wed Mar 27 17:15:52 EDT 2019)
  smpl_params={
    'gender':gender,
    'betas':betas,
    }
  smpl(smpl_params)
#===================================================================================================================================
if __name__=="__main__":
  main()
#===================================================================================================================================
