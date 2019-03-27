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

## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( '../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl' )

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2
#m.betas[:] = np.random.rand(m.betas.size) * .03
m.betas[:] = np.full((m.betas.size),5)

## Write to an .obj file
outmesh_path = './hello_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

#===================================================================================================================================
def blender_render_mesh(mesh_fname):
  '''
    I had to hack around the fact that subprocess.call() doesn't support cmd line args  ( or was it blender --python itself that doesn't?)
    So I rewrite a tiny python file with the argument supplied here and call blender with that.
  '''
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
#===================================================================================================================================
blender_render_mesh(outmesh_path)

## Print message
print('..Output mesh saved to: ', outmesh_path) 
