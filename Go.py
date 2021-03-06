from hello_world import smpl
from utils import pn,pe

import numpy as np
import sys
from copy import deepcopy
if    sys.version_info.major==3: # this will probably change and break
  pr=print
elif  sys.version_info.major==2:
  pass # can't "shortcut pointer"  pr=print






# This outline isn't THE worst, but it'll probably end up being wrong around the edges.  How the f**k can I say whether I should tune @ the end if I haven't even tuned @ the beginning yet?

  # The biggest issue with my outline is most of the fit()ting looping has to happen WITHIN the fit_vertical() and fit_around() functions, not in the macro fit() function.
  #   Also, with coordinate ascent/descent, the error/loss is an integral PART of the upd8_vertical() function
    #   Each of the 1st 2 betas has to be tweaked to get this sh!t

# IDEA:   cache all the meshes (betas) & search the space that way.   A problem is there are 10^10 such meshes, even if we only go for integer betas s.t. -5 < beta < 4.






























# TODO refactor LATER.  But yes, everything oughta be a dictionary.  HashTables ftw.
      # try to avoid duplicating costly operations?  (ie videocut)

  # overview:
  #  height-ratio
       #  circumference  measurement  width
 
      #  Maybe  we  should  have  2  internal      while loops (1 for height-ratios, 1 other for circumferences) ,
      #   And then a bigger while loop

      #   mesh=fit_vert(frames)           # means "fit_vertical()"
      #   mesh=fit_around(frames)
      # we can leave the while loop, have it default-end after 1 iteration, then extend (if possible)

    # It's probably worth it to 1) do a few BNN runs each time we try to fit,   2) pick the best "general gist" of the betas, then   3) return the one that "looks most like" the customer.
    #   Basically, it's best to avoid getting caught in a local optima;  better to find the GLOBAL optimum.




    # For the final tweaking, it's probably best to "do openpose" on the SMPL mesh and locally stretch/shrink various vertex-regions.
    #   The best way to do this will probably be to make regularization() EXTREMELY punishing of outliers (ie. np.abs(MCE [cubic error]) and)

    #   Whatever tweaking we do will have to be smooth (ie. if we find the elbow and the wrist and want to stretch it,  we should 1) convolve the elbow->wrist length with a 2-D Gaussian (ie. y=Gaussian(x)), not   2) just multiply all the radial distances away from the centroid-line of the arm by 3.   )
    #     For smoothly varying the thickness of the forearm, it's worth looking into bicubic interpolation and other similar methods.  Of course, 1st you should try the bilinear-interpolation-esque methods; easier to code, easier to reason about in a group, etc.
    #     It's probably NOT AS IMPORTANT with the "length of the forearm"   as it is with the "radii of the forearm cross-sections"





















#===================================================================================================================================
def init_shape(info):
  '''
    initialize the shape parameters ("betas" in the SMPL papers) with HMR.
  '''

  '''                         # HMR:
  frames=info['frames']
  front=frames[0]
  #mesh=HMR(front)
  '''

  # I'm kind of liking the betas=np.random.random((10)).astype("float64") idea instead.  I don't have to read any HMR code, and we might sometimes get some very peculiar edge cases that end up with exactly the result what we want!    -nxb (Wed Mar 27 17:44:13 EDT 2019)
  # Sometimes all we need to do is "inject a little randomness," as our man Nakul always says.
  betas=\
    5 -\
    (10*np.random.random((10)).astype('float64'))
  params={
    'gender'  : 'male',
    'betas'   : betas,
  }
  pe(66); pr("betas:\n",betas); pe(66)

  smpl_output=smpl.smpl(params)
  mesh=smpl_output['mesh']
  params['mesh']=mesh

  return deepcopy(params)
#====================================================== end func init_shape(params) =============================================================================
"""
def body_part_penalty(info):
  betas=info['shape']
  return {'penalty':penalty}
#===================================================================================================================================
def fit_each_body_part(info):
  '''
    # For the final tweaking, it's probably best to "do openpose" on the SMPL mesh and locally stretch/shrink various vertex-regions.
    #   The best way to do this will probably be to make regularization() EXTREMELY punishing of outliers (ie. np.abs(MCE [cubic error]) and)

    #   Whatever tweaking we do will have to be smooth (ie. if we find the elbow and the wrist and want to stretch it,  we should 1) convolve the elbow->wrist length with a 2-D Gaussian (ie. y=Gaussian(x)), not   2) just multiply all the radial distances away from the centroid-line of the arm by 3.   )
    #     For smoothly varying the thickness of the forearm, it's worth looking into bicubic interpolation and other similar methods.  Of course, 1st you should try the bilinear-interpolation-esque methods; easier to code, easier to reason about in a group, etc.
    #     It's probably NOT AS IMPORTANT with the "length of the forearm"   as it is with the "radii of the forearm cross-sections"
  '''
  #info[]
  # loss
  #   The best way to do this will probably be to make regularization() EXTREMELY punishing of outliers (ie. np.abs(MCE [cubic error]) and)
  return {}
#===================================================================================================================================
def tune(info):
  '''
    Fix up the final details of the mesh
      ie. forearm length, forearm radii, hip heights,   anything little and probably the extreme features that natural variation of SMPL doesn't easily capture.
        More examples:
          My legs are freakishly long.
          My waist is freakishly small.
          I am freakishly tall.
  '''
  mesh=info['mesh']
  faces=mesh['faces']
  verts=mesh['verts']
  #verts=mesh['verts']  # openpose_measures
  #verts=mesh['verts']  # smpl_measures
  while loss > STOP:
    mesh=stretch(mesh,mesh_measures)['mesh']
    fit_each_info={ 'openpose_measures':openpose_measures,
                    'mesh_measures':mesh_measures,
                    'mesh':mesh,}
    fit_each_output=fit_each_body_part(fit_each_info)
  out={'mesh', mesh}
  #...
  return out
#===================================================================================================================================
#           VERTICALS
#===================================================================================================================================

#===================================================================================================================================
def vertical_penalty(info):
  return {}
#===================================================================================================================================

#===================================================================================================================================
def vertical_err(info):
  return {}
#===================================================================================================================================

#===================================================================================================================================
def upd8_vertical(info):
  mesh=info['mesh']
  faces=mesh['faces']
  verts=mesh['verts']
  mesh_measures=mesh['mesh_measures']
  openpose_measures=openpose['openpose_measures']
  # upd8
  return {'mesh': mesh}
#===================================================================================================================================

#===================================================================================================================================
#===================================================================================================================================
def fit_vertical(info):
  #info is a dict.  
  # working  hypothesis: It's almost always good to return dict and send dict in as params; more flexible than a list of parameters, even with optional/keyword params
  frames=info['frames']
  mesh=info['mesh']
  # TODO: copypaste "vertical_error" code from mesh_err().
  '''
    When we update, do we need to send around the whole mesh?  Is that better than just sending around betas?
      I don't THINK there are memory (size complexity) concerns.  Not 100% sure, though.

    vert_err=vertical_error(info)
    vert_err+=regularize(betas)
  '''
  while loss > STOP:
    upd8_vert_info={"mesh":mesh, frames}
    upd8d=upd8_vert(upd8_vert_info)
    mesh=upd8d['mesh']
  return {'mesh': {'faces':faces, 'verts':verts}}

#===================================================================================================================================
#           AROUNDS
#===================================================================================================================================
def around_penalty(info):
  return {}
#===================================================================================================================================

#===================================================================================================================================
def around_err(info):
  return {}
#===================================================================================================================================

#===================================================================================================================================
def upd8_around(info):
  mesh=info['mesh']
  faces=mesh['faces']
  verts=mesh['verts']
  mesh_measures=mesh['mesh_measures']
  openpose_measures=openpose['openpose_measures']
  # upd8
  return {'mesh': mesh}
#===================================================================================================================================

#===================================================================================================================================
def fit_around(info):
  #info is a dict.  Almost always good to return dict and send dict in as params.
  frames=info['frames']
  mesh=info['mesh']
  # TODO: copypaste code from mesh_err().
  while loss > STOP:
    mesh=upd8_around(mesh, measures)
    # how different mesh's circumferences are from   openpose-calcul8d circumferences
    loss=around_err(mesh)
    # regularization
    loss+=around_penalty({'shape':betas})
  return {'mesh': {'faces':faces, 'verts':verts}}
#===================================================================================================================================

#===================================================================================================================================
def real_shape(info):
  '''
    Takes an "almost mode" of the betas
      where we detect the most plentiful betas & return those
    Or we could just take an argmin of the losses
  '''
  betas_cache=info['betas_cache']
  which_beta=np.argmin(info['losses'])
  return {
    'shape':betas_cache[which_beta],
  }
  '''
    clusters = sklearn.kmeans(betas_cache)
    global_max=find_best_mean(clusters)
    scipy.stats.mode()
  '''
#===================================================================================================================================
"""
def fit(video):
  shape=init_shape(video)
  #STOP=FIDDLE_with()
  N=10 # fiddle with this too (N runs of mini-BNN)
  betas_cache=np.zeros((N,10,)).astype('float64') # rename cache?
  losses=np.zeros((N)).astype('float64')
  for run in range(N):
    while loss > STOP:    # start by trying to make this condition True after 1 iteration;    so we can visualize everything after just 1 run.
      vertical_output=fit_vertical(vertical_info)
  """
      mesh  = vertical_output['mesh']
      around_output=fit_around  (around_info)
      mesh  = around_output  ['mesh']
      loss  = mesh_err(mesh)+ regularization
    betas_cache[run]=around_output['betas']
    losses[run]=loss
    # end while loop
  # end for loop
  # TODO: the "rough" mode() rather than the actual mode
  real_shape=real_shape({'losses':losses,'betas_cache':betas_cache})
  tune_info={'mesh':mesh,'measures':measures}
  tune_output=tune(tune_info)
  mesh=tune_output['mesh']
  mesh= update()
  return mesh
  """

  '''                           NOTES:
  # 2 steps:
#  height-ratio
     #  circumference  measurement  width

    #  Maybe  we  should  have  2  internal      while loops (1 for height-ratios, 1 other for circumferences) ,
    #   And then a bigger while loop
    # It would be even simpler if we were able to get it 100% right the 1st time, ie:

    #   mesh=fit_vert(frames)           # means "fit_vertical()"
    #   mesh=fit_around(frames)
    # we can leave the while loop, have it default-end after 1 iteration, then extend (if possible)

    # try to avoid duplicating costly operations?  (ie videocut)

  # It's probably worth it to 1) do a few BNN runs each time we try to fit,   2) pick the best "general gist" of the betas, then   3) return the one that "looks most like" the customer.
  #   Basically, it's best to avoid local optima for the sake of finding the GLOBAL optimum.

  # For the final tweaking, it's probably best to "do openpose" on the SMPL mesh and locally stretch/shrink various vertex-regions.
  #   The best way to do this will probably be to make regularization() EXTREMELY punishing of outliers (ie. np.abs(MCE [cubic error]) and)
  #   Whatever tweaking we do will have to be smooth (ie. if we find the elbow and the wrist and want to stretch it,  we should 1) convolve the elbow->wrist length with a 2-D Gaussian (ie. y=Gaussian(x)), not   2) just multiply all the radial distances away from the centroid-line of the arm by 3.   )
  #     For smoothly varying the thickness of the forearm, it's worth looking into bicubic interpolation and other similar methods.  Of course, 1st you should try the bilinear-interpolation-esque methods; easier to code, easier to reason about in a group, etc.
  #     It's probably NOT AS IMPORTANT with the "length of the forearm"   as it is with the "radii of the forearm cross-sections"
  '''
#===================================================================================================================================


if __name__=="__main__":
  vid_fname='[TODO]'
  mesh=fit(vid_fname)
