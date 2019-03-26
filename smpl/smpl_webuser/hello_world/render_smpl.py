# This is what I was thinking I'd rebuild!   Aaargh I feel so dumb.   Tue Feb 19 08:43:06 EST 2019
#     So if we can get the sliders in here (ie. kivy or www.bodyvisualizer.com), our job is complete.
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
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''

from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
from smpl import write_smpl
from Jonah_hill___Thetas import Jonah_Hill

import numpy as np
import chumpy as ch
import sys
import math
from math import sin, cos, pi

#=========================================================================
#pr=print # python2 doesn't allow.
def pn(n=0): print('\n'*n)
def pe(n=89): print('='*n)
#=========================================================================

gender = 'male'
## Load SMPL model
if gender.lower()=='male':
  m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
else:
  m = load_model('../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

## Assign pose and shape parameters
m.pose[:] = np.zeros(m.pose.size).astype('float64') # m.pose.shape is (72,).  72 pose parameters   # All zeros is "Jesus pose."  #np.random.rand(m.pose.size) * .2
m.betas[:] = np.zeros(m.betas.size).astype('float64') 



#  There might be a reference for this (what each m.pose value means) somewhere.  But I doubt it.

#m.pose[1]
#m.pose[1] = 0.34 # rotates roughly around the axis from bellybutton to spinal cord.
#m.pose[2] = 0.33 # rotates rougly around the axis from foot to head
#=======================================
#m.pose[3] = -0.9  # left leg back (positive)
#m.pose[4] =  0.9 # left foot out (~marching band splayed out feet pose, positive)
#m.pose[5] =  0.9 # left foot away 
#=======================================
#m.pose[6] =  0.9  # values [6] thru [8] are identical, but for right leg, foot, etc.
#============ right leg ================
#m.pose[9]  =  0.9 # rotates torso down (bend over to touch toes)
#m.pose[10] = 0.9 # rotating to crack back
#m.pose[11] = 0.9 # torso again
#============ left leg ================
#m.pose[12]=0.9 # left knee
#m.pose[13]=0.9 # left leg rotates a diff way
#m.pose[14]=0.9 # left leg again
#============ right leg ================
#m.pose[15]=0.9 # right knee (tibia up into air)
#m.pose[18]=0.9 # bend down

#m.pose[27]=1.9 # head down
#m.pose[31]=3.9 # toes
#m.pose[36]=1.9 # also head down.  Why??  But it doesn't look like ALL of the poses repeat cyclically.  Just a few...  Or maybe I can't recognize the subtle differences between, say, m.pose[36] and m.pose[27].
#m.pose[39]=1.1 # a hand movement!  Good news, Chris Columbus; there IS land on the other side of the ocean.
#m.pose[42]=1.4   # right arm pitch? yaw? roll?

## Rotates (like a backflip)
m.pose[0] =  pi  # NOTE # I'm p sure these rotations (I think it was pose[:3] that are the rotations) are taken care of differently in HMR.
#m.pose[0] = 2.4
#m.pose[41]= -1.1  # m.pose[41] is shoulder-level rotation.
#m.pose[44]=  1.1  # m.pose[44] is shoulder-level rotation.
# more poses after, but I don't have to deal with 'em right now.

# Todo: put the following code snippet (sys.argv...) in hello_smpl.py.  It adapts to however many cmd line args you feed this module.
# Get betas from cmd line args
for i in range(1,len(sys.argv)):
  m.betas[i-1]=float(sys.argv[i])
# The above little chunk of command-line-args (sys.argv) parsing code should be reusable wherever we do SMPL stuff.  It adapts to however many cmd line args you put in.

## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480) # width and height

# Nathan Bendich:   Note to self: chumpy is a pain in the ass to mutate.  I think this is b/c it autodifferentiates
rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(m)*.9,
    light_color=np.array([1., 1., 1.]))



## Show it using OpenCV
import cv2
#from cv2 import CV_WINDOW_NORMAL
#cv2.namedWindow("main", CV_WINDOW_NORMAL)  # meant to resize the windows.  But I need Qt backend support to do this.  Would have to reinstall opencv for python, prob in a diff virtualenv or conda env
cv2.imshow('SMPL_{0}'.format(m.betas), rn.r)
print ('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()


# write mesh:
outmesh_path = \
  './{10}_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(
    m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]),
    gender.lower())
    # ie. male_02-20-300000.obj
with open( outmesh_path, 'w') as fp:
    for v in rn.v:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
pe(); print("m.betas:\n ",m.betas); pe()
print("new mesh saved at "+outmesh_path);pe()



## Could also use matplotlib to display
#import matplotlib as mpl
#mpl.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()





































































