gender = 'male'
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

## Make sure path is correct
## Load SMPL model
if gender.lower()=='male':
  m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
else:
  m = load_model('../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# shape
print("m.betas.size: ",m.betas.size)
betas = np.zeros((m.betas.size)).astype('float64')
# it's best to store the betas in a temporary variable so when we assign everything, SMPL calculates EVERYTHING in one step
for i in range(1,len(sys.argv)):
  betas[i-1]=float(sys.argv[i])
m.betas[:] = betas

# pose
m.pose[:] = np.zeros((m.pose.size)).astype('float64')
## Rotates (like a backflip)
m.pose[0] =  pi  # Note: I'm p sure these rotations (I think it was pose[:3] that are the rotations) are taken care of differently in HMR.


## Write to an .obj file
outmesh_path = \
  './{10}_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(
    m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]),
    gender.lower())
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
