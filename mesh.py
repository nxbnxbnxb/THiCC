
# implementation of:

# "Voronoi-based Variational Reconstruction of Unoriented Point Sets"
#     by Alliez, Cohen-Steiner, Tong, and Desbrun

# link: http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf

'''
  As of Dec. 30, 2018, PART of just the normal-finding took: 
    real  14m40.537s
    user  14m39.357s
    sys   0m  1.088s
  Unacceptably slow.  Try to get the c++ CGAL libraries working

'''

import numpy as np
np.seterr(all='raise')
from copy import deepcopy
from scipy.spatial import Voronoi
import scipy.spatial
import sys
from d import debug
from utils import pif
from datetime import datetime
from save import *


#=========================================================

def volume_tetra(tetra):
  '''
    returns the volume of a tetrahedron

    tetra.shape is (4,3)

      -----
      Notes
        formula from https://stackoverflow.com/questions/9866452/calculate-volume-of-any-tetrahedron-given-4-points
        Tested on a few tetrahedrons.  Can verify any given tetrahedron volume at https://keisan.casio.com/exec/system/1329962711
  '''
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  tetra_float=tetra.astype('float64')
  a=tetra_float[0]; b=tetra_float[1]; c=tetra_float[2]; d=tetra_float[3]
  return abs(np.dot(a-d, 
                  np.cross(b-d,c-d)))/6.0
# end func def of volume_tetra(tetra)
#=========================================================

def CoM_and_vol(vertices):
  '''
    get center of mass and volume of arbitrary CONVEX polyhedron

    Params:
    -------
    vertices is a 2-D np.array.
    vertices.shape == (n,3)
    
    
    Notes:
    ------

    An approximate way of doing this would be to fill a 3-D np array and use scipy.ndimage.measurements.center_of_mass()  :
      (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.measurements.center_of_mass.html)

    Tested this function a few times with regular polyhedra like cube, octohedron, icosahedron, and dodecahedron.
    At the moment, this actually WORKS.  Come back to this commit if things start breaking.
  '''
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  hull=scipy.spatial.ConvexHull(vertices)
  volume=hull.volume
  triangle_mesh_hull=vertices[hull.simplices] # (n,3,3)
  # triang mesh calculation taken from    https://stackoverflow.com/questions/26434726/return-surface-triangle-of-3d-scipy-spatial-delaunay/26516915
  inner_pt = np.mean(vertices[:2],axis=0).reshape((1,3))
  CoM=np.zeros((3,))
  pif("inner_pt is {0}".format(inner_pt))
  for triangle in triangle_mesh_hull:
    pif("triangle is: \n{0}".format(triangle))
    tetra=np.concatenate((inner_pt,triangle),axis=0)
    CoM_tetra=np.mean(tetra,axis=0)
    vol_tetra=volume_tetra(tetra)
    CoM+=(CoM_tetra*vol_tetra)
  return CoM/volume, volume
# end func def of CoM_and_vol(vertices):
#=========================================================

def norms(vertices):
  pass
def norms(vertices):
  '''
    get center of mass and volume of arbitrary CONVEX polyhedron

    Params:
    -------
    vertices is a 2-D np.array.
    vertices.shape == (n,3)
    
    
    Notes:
    ------

    An approximate way of doing this would be to fill a 3-D np array and use scipy.ndimage.measurements.center_of_mass()  :
      (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.measurements.center_of_mass.html)

    Tested this function a few times with regular polyhedra like cube, octohedron, icosahedron, and dodecahedron.
    At the moment, this actually WORKS.  Come back to this commit if things start breaking.
  '''
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  hull=scipy.spatial.ConvexHull(vertices)
  volume=hull.volume
  triangle_mesh_hull=vertices[hull.simplices] # (n,3,3)
  # triang mesh calculation taken from    https://stackoverflow.com/questions/26434726/return-surface-triangle-of-3d-scipy-spatial-delaunay/26516915
  inner_pt = np.mean(vertices[:2],axis=0).reshape((1,3))
  CoM=np.zeros((3,))
  pif("inner_pt is {0}".format(inner_pt))
  for triangle in triangle_mesh_hull:
    pif("triangle is: \n{0}".format(triangle))
    tetra=np.concatenate((inner_pt,triangle),axis=0)
    CoM_tetra=np.mean(tetra,axis=0)
    vol_tetra=volume_tetra(tetra)
    CoM+=(CoM_tetra*vol_tetra)
  return CoM/volume, volume
# end func def of CoM_and_vol(vertices):
#=========================================================

def centroid_and_vol(idx, vor):
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  region = vor.regions[idx]
  INF_BOUND=-1
  if INF_BOUND in region:
    return False
    # TODO: figure out how to handle infinite bound of vor region;   this should just mean that this region is a dummy point
  vertices = vor.vertices[region]
  pif("vertices are {0}".format(vertices))
  if len(vertices) == 0:
    return False
    # TODO:   Perhaps I should use a neighbor's precalculated normal in this case, maybe also decreasing the magnitude of the covariance tensor.
    # NOTE:   This behavior is explained in (https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/spatial.html#qhulltutorial; "Note here that due to similar numerical precision issues as in Delaunay triangulation above, there may be fewer Voronoi regions than input points"), but I am still not sure WHY this happens.  Shouldn't every input point have an associated Voronoi region, at least unless it is a duplicate?  I think it has something to do with the PARTICULAR algorithm qhull uses to calculate Vor_3d, rather than the mathematical definition of the voronoi region
  else:
    pif("vertices.shape:  {0}".format(vertices.shape))
    return CoM_and_vol(vertices)
# end func def of centroid_and_vol(idx, vor):
#=========================================================

def add_dummies(pt_cloud):
  '''
    Dummy pts as described in the Alliez paper

    Notes:
    ------
      Assumes pt_cloud is regular (ie. cube)
  '''
  # TODO:   make more dummies; has to be dense enough that it's impossible for a real data point's vor region to extend to infinity
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  end=max(pt_cloud.shape)-2; mid = int(round((end+2)/2)); start=1
  with_dummies = deepcopy(pt_cloud)
  # 8 corners of the cube
  with_dummies[ start, start, start]=True
  with_dummies[ start, start,   end]=True
  with_dummies[ start,   end, start]=True
  with_dummies[ start,   end,   end]=True
  with_dummies[   end, start, start]=True
  with_dummies[   end, start,   end]=True
  with_dummies[   end,   end, start]=True
  with_dummies[   end,   end,   end]=True
  # 6 faces
  with_dummies[ start,   mid,   mid]=True
  with_dummies[   end,   mid,   mid]=True
  with_dummies[   mid, start,   mid]=True
  with_dummies[   mid,   end,   mid]=True
  with_dummies[   mid,   mid, start]=True
  with_dummies[   mid,   mid,   end]=True
  return with_dummies
# end def of func add_dummies(pt_cloud)
#=========================================================
def ZERO_NORM():
  return np.eye(3).astype("float64")
def norms(vor):
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  norms=np.zeros((3,3, len(vor.regions)))
  Q=np.array([[2,1,1],
              [1,2,1],
              [1,1,2]]).astype('float64')/120.0  # check Alliez et al. paper
  for idx in range(len(vor.regions)):
    pt=vor.points[idx]
    region = vor.regions[idx]
    INF_BOUND=-1
    if INF_BOUND in region:
      continue  # when we've done this correctly, this will only happen for dummy (edge) points
    vertices = vor.vertices[region]
    if len(vertices) == 0:
      norms[:,:,idx] = ZERO_NORM()
      continue
    hull=scipy.spatial.ConvexHull(vertices)
    volume=hull.volume
    triangle_mesh_hull=vertices[hull.simplices] # (n,3,3)
    inner_pt = np.mean(vertices[:2],axis=0).reshape((1,3))
    CoM_vor=np.zeros((1,3)).astype('float64')
    ctr=5
    for triangle in triangle_mesh_hull:
      if ctr > 0:
        print(triangle) #[[x1, y1, z1],
        ctr-=1          # [x2, y2, z2],
      tetra=np.concatenate((inner_pt,triangle),axis=0)
      CoM_tetra=np.mean(tetra,axis=0)
      vol_tetra=volume_tetra(tetra)
      CoM_vor+=(CoM_tetra*vol_tetra)
    CoM_vor/=volume
    voros_cov=np.zeros((3,3)).astype('float64')  # THIS vorohedron's covariance
    for triangle in triangle_mesh_hull:
      tetra=np.concatenate((CoM_vor,triangle),axis=0)
      N=(triangle-np.vstack((CoM_vor,CoM_vor,CoM_vor)).astype('float64')).T
      voros_cov += np.linalg.det(N)*np.dot(N,Q,N.T)
      # this is enough for single voros' covariances, but now we wanna union the adjacent vorohedrons' covariances too  (loosely speaking; again, read the paper :P)
      # NOTE:  this doesn't compile yet.  It just shows an intermediate step that I wanted to document in git



      # TODO:  check the results of the Voronoi() function
      #   calculate N using each of the tetrahedrons' CoMs, translate with vorohedrons' CoMs, and thereby find the covariances
      #   "unioning" process also needs to be baked in.  

      # NOTE: As is, the whole thing is really really way too slow.  There's gotta be a faster way

if __name__=='__main__':
  model = np.load('skin_nathan_.npy').astype('bool')
  model = add_dummies(model)

  locs=np.nonzero(model); locs=np.array(locs).T.astype('int64')
  # vor step of Alliez et al.   (for estimating normals from point cloud)
  vor=Voronoi(locs)
  normals=norms(vor)







# calling Voronoi() on 831259 on locs,  shape==(513, 513, 513)
  #
  # time: 
  #
  #   real  1m3.208s
  #   user  1m2.044s
  #   sys   0m1.344s
  #







































