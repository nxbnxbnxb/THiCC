NATHAN=True
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

import smpl_webuser.serialization
from smpl_webuser.serialization import load_model
import numpy as np
import sys
from numbers import Number

#===================================================================================================================================
def body_talk_male():
#===================================================================================================================================
  '''
    Generate lotsa bodies
  '''
  ## Load SMPL model (here we load the male model)
  ## TODO: Make sure path is correct
  MAN   = '../../models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
  m = load_model( MAN )

  ## Assign random pose and shape parameters
  m.pose[:]  = np.zeros(m.pose.size).astype('float64')
  m.betas[:] = np.zeros(m.betas.size).astype('float64')
  print("m.betas.size: {0}".format(m.betas.size))

  #print("len(sys.argv) is {0}".format(len(sys.argv)))
  print("sys.argv is {0}".format(sys.argv))
  for i in range(10):
    for beta in (-15,15):
      m.betas[i]=beta

      ## Write to an .obj file
      outmesh_path = './male_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]))
      with open( outmesh_path, 'w') as fp:
          for v in m.r:
              fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

          for f in m.f+1: # Faces are 1-based, not 0-based in obj files
              fp.write( 
              'f %d %d %d\n' %  (f[0], f[1], f[2]) )
      ## Print message
      print('..Output mesh saved to: ', outmesh_path)
    m.betas[i]=0 # reset so we can isolate the effect of each beta
  # end for i in range(10):
#===================================================================================================================================


#===================================================================================================================================
def custom_body(female=False, height=False, weight=False, chest=False, waist=False, hips=False, inseam=False, exercise=False):
#===================================================================================================================================
  '''
    These params (height, weight, , stick_insectism, flabbiness, broad_shoulderedness, etc.) should be provided as numerical values if you pass the custom_body() function any parameters
  '''


  """
  flabbiness            = False
  broad_shoulderedness  = False
  """
  # TODO: calculate these if people want more precision (by "precision," I really mean "how-much-does-the-model-look-exactly-like-me?")
 
  if female:
    avg_height  =  65 # NOTE: INCHES and POUNDS!  Must convert for SI units
    avg_weight  = 141 # pounds
    avg_chest   =  37
    avg_waist   =  30
    avg_hips    =  40
    avg_inseam  =  30
    avg_exercise=   1 # hours / week
  else: # male
    avg_height  =  70 # NOTE: INCHES!  Must convert for SI units
    avg_weight  = 180 # pounds
    avg_chest   =  40
    avg_waist   =  35
    avg_hips    =  41
    avg_inseam  =  31
    avg_exercise=   2 # hours / week
 
  if NATHAN:      # I measured myself.  Customers won't want to do this; it took awhile and I had to find a measuring tape and a mirror
    height      =  75.
    weight      = 157.4
    chest       =  34.
    waist       =  30.
    hips        =  32.
    inseam      =  35.

  if (not waist) and (height and weight):
    waist=weight/height # estimate waist measurement from height and weight.  More people roughly know their weight and height   than know their waist measurement off the top of their head

  #'''
  HEIGHT_SCALING  = 1 # TODO: tinker with this.  Do we need independent HEIGHT and WAIST scaling factors?
  WAIST_SCALING   = 1 # NOTE: not currently in use
  #'''
  # TODO: get rid of DWARFISM altogether?  We can just scale the body instead
  DWARFISM_SCALING= 0.0 # 0.15  # might want more complex scaling, like Gaussian  (I THINK 1 beta means one standard deviation away)
  STICK_SCALING   = 0.2
  SHORT_LEGS_FAT_SCALING  = 0.31 #0.25
  SHORT_LEGS_THIN_SCALING = 0.25
  CHUBBINESS_SCALING      = 0.15
  PEAR_NESS_SCALING       = 0.45
  BIG_BELLY_SMALL_HIPS_SCALING=0.5 # 6th beta  m.betas[6]
  # 7th 
  # 8th   not sure what 8th and 7th do that hasn't already been adjusted earlier
  BROAD_SHOULDERS_SMALL_HIPS_SCALING=0.5# 9th

  
  #NOTE:  these constants are in order for males (ie. dwarfism is the 0th beta, stick-insectism is the 1th beta, ... pear_ness is the 5th beta
  # TODO:  complete the m.betas[i] = ...    for all of these.  Sometimes the effects they have will be negligible, and we need to test their interdependencies as well (ie. if we make stick-insect-y and dwarf-y simultaneously, do we get thinner but with other proportions equal?).

  dwarfism        =(-(height-avg_height) - (waist-avg_waist)) * DWARFISM_SCALING # NOTE: if I do it this way, we need a waist measurement.  But it's better this way than just using weight; weight can be caused by big muscles, thick legs, lots of different stuff
  #dwarfism        =(-(height-avg_height)*HEIGHT_SCALING - (waist-avg_waist)*WAIST_SCALING) *DWARFISM_SCALING 
  # minus signs because of the way the beta sign is handled
  stick_insectism = ((height-avg_height) - (waist-avg_waist)) * STICK_SCALING
  #stick_insectism = ((height-avg_height)*HEIGHT_SCALING - (waist-avg_waist)*WAIST_SCALING) * STICK_SCALING
  short_legs_fat  = (-(inseam-avg_inseam) + (waist-avg_waist)) * SHORT_LEGS_FAT_SCALING
  short_legs_thin = (-(inseam-avg_inseam) - (waist-avg_waist)) * SHORT_LEGS_THIN_SCALING
  chubbiness      = ( (chest-avg_chest)   + (waist-avg_waist) + (hips-avg_hips) ) * CHUBBINESS_SCALING  # NOTE: this one might be the one that needs individual scaling factors in the middle there (ie. WAIST_SCALING, HEIGHT_SCALING)
  #pear_ness       = ( (chest-avg_chest)   - (waist-avg_waist) + (hips-avg_hips) ) * PEAR_NESS_SCALING  # NOTE: this one might be the one that needs individual scaling factors in the middle there (ie. WAIST_SCALING, HEIGHT_SCALING)
  print('dwarfism:',dwarfism)
  print('stick_insectism:',stick_insectism)
  pear_ness = False

  #   Load SMPL model
  if female:
    m = load_model( '../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl' ) # TODO: detect female / male-ness from photo
  else:
    m = load_model( '../../models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl' ) # TODO: detect female / male-ness from photo

  ## Assign pose and shape parameters
  m.pose [:] = np.zeros(m.pose.size).astype('float64')
  m.betas[:] = np.zeros(m.betas.size).astype('float64')
  #m.betas[:] = np.random.rand(m.betas.size) * .03

  # print("m.betas.size: {0}".format(m.betas.size)) #10


  # For women:
  '''
  # -10 on all means short and fat       10 on all means tall and skinny
  #m.betas[0] = 10
  # -10 on 0th means short and skinny,   10 on 0th means tall and fat
  # -10 on 1st means short and fat,      10 on 1st means tall and skinny

  # 5 on both DOES a tall person, but they're also p skinny.  Not what you might expect given the 1st PC does tall and fat
  #(-3, 3) is a starving person
  #( 3,-3) is a fat giant
  #(-3,-3) is fatter and a tiny bit shorter
  #( 3, 3) is very thin

  #  For the "2nd" PC, even 10 isn't big enough to show a huge difference
  #  I'm not sure if there's any "intuitive" meaning to the 2nd PC.

  # For "higher" PCs, look at the .blend file.  It's unlikely any of these is going to be our moonshot to success.

  #print("len(sys.argv) is {0}".format(len(sys.argv)))
  '''

  # For   men:
  '''
    Avg measurements, according to bodyvisualizer.com/male.html
      Height:     70 inches
      Weight:    180 pounds
      Chest:      40 inches
      Waist:      35 inches
      Hips:       41 inches
      Inseam:     31 inches
      Exercise:    2 hrs/week


    Male:

      1st PC ([0]) is short and small belly
      2nd PC ([1]) is tall  and big   belly
      3rd PC ([2]) is short-legs + fat
      4rd PC ([3]) is short-legs + skinny
  '''
  # TODO: confirm we know exactly what all these PCs mean.  TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
  print("cmd line args (sys.argv) =\n{0}".format(sys.argv))
  if female:
    shoulder_hip_ratio  = m.betas[5]
  else: # male
    if isinstance(dwarfism, Number): # not False, like I made the default in the parameter list
      m.betas[0]          = dwarfism
    else:
      short_and_thin      = m.betas[0] # TODO: kill these pointers?  (all the "else" clauses)    The point of 'em was so we could feed in measurements from outside like "short_and_thin = sys.argv[i]"
    if isinstance(stick_insectism, Number):
      m.betas[1]          = stick_insectism
    else:
      tall_and_thin       = m.betas[1]
    if isinstance(short_legs_fat, Number):
      m.betas[2]          = short_legs_fat
    else:
      short_legs_fat      = m.betas[2]
    if isinstance(short_legs_thin, Number):  # NOTE: honestly, might wanna kill the if statements too and just have it read in waist, etc. (normal, easy-for-people-to-understand measurements)
      m.betas[3]          = short_legs_thin
    else:
      short_legs_thin     = m.betas[3]
    if isinstance(chubbiness, Number):
      m.betas[4]          = chubbiness
    else:
      chubbiness          = m.betas[4]
    if isinstance(pear_ness, Number):
      m.betas[5]          = pear_ness
    else:
      pear_ness           = m.betas[5]

    broad_shoulders_small_hips  = m.betas[9]
  # end else: #( male)  NOTE: all above values are for MALE
  for i in range(10):
    if m.betas[i] == 0:
      m.betas[i] = float(sys.argv[i+1]) # up to 10 cmd line args
    else:
      pass
      # m.betas[5]:    for women, shoulder-broadness vs. hip-width.  Positive values mean broader shoulders and more weight in torso; Negative values mean wide hips
  print("m.betas are: \n{0}".format(m.betas))

  ## Write to an .obj file
  if female:
    outmesh_path = './female_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]))
  else:
    outmesh_path = './male_{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}.obj'.format(int(m.betas[0][0]),int(m.betas[1][0]),int(m.betas[2][0]),int(m.betas[3][0]),int(m.betas[4][0]),int(m.betas[5][0]),int(m.betas[6][0]),int(m.betas[7][0]),int(m.betas[8][0]),int(m.betas[9][0]))

  with open( outmesh_path, 'w') as fp:
    for vertex in m.r:
      fp.write( 'v %f %f %f\n' % ( vertex[0], vertex[1], vertex[2]) )

    for face in m.f+1: # Faces are 1-based, not 0-based in obj files
      fp.write( 
      'f %d %d %d\n' %  (face[0], face[1], face[2]) )

  ## Print message
  print('..Output mesh saved to: ', outmesh_path)
# end func definition custom_body()
#===================================================================================================================================
if __name__=="__main__":
  #body_talk_male()
  from gender import gender  # NOTE: "sex," technically; gender is the socially-constructed one
  if gender.lower()=='male':
    female=False
  else:
    female=True
  custom_body(female) # TODO: refactor to say "gender" instead of clumsy boolean

