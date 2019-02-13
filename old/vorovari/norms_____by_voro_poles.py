
# This code is Bendich's riff on an implementation of "Voronoi Poles" normal-vector-finding:

# Reference material:
#   "Voronoi-based Variational Reconstruction of Unoriented Point Sets"
#       by Alliez, Cohen-Steiner, Tong, and Desbrun
#     NOTE: nickname for the paper: Voro-Vari
#   "mesh.py" (in subdir "tests/" under github main repo as of January 10, 2019)

# link: http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf

import numpy as np
np.seterr(all='raise')
from copy import deepcopy
import scipy.spatial
import sys
from d import debug
from utils import pif
from datetime import datetime
from save import *
from numpy import linalg as LA  # TODO: shorten every np.linalg.[insert funcall name]() to LA.[funcall name]() 

# below imports are mainly to prevent "np.core.numeric.ComplexWarning":
import warnings
warnings.filterwarnings("error")

# NOTE:   np.eigh() or np.eigvalsh() may work faster/more reliably for a symmetric (or Hermitian: conjugate symmetric) matrix
#
#
#
#
#
#
#
# TODO:   more explicit type casting:  ie.  .astype('float64'), .astype('int64')
#         check the results of the Voronoi() function
#
#
#
#
#
#
#
#


#=========================================================
def add_dummies(pt_cloud):
  '''
    Dummy pts as described in the Alliez paper

    Notes:
    ------
      Assumes pt_cloud is regular (ie. cube)
  '''
  # TODO: better way to solve the dummy problem is to put the model in a larger numpy array and "insert" it into the center.  Jan. 9, 2019
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
def norms(vor):
  '''
    Calculates the norms at each real (not dummy) point  from the input data pt_cloud.

    Parameters
    ----------
    vor is an object returned by the scipy.spatial.Voronoi() function;  you'll have to call Voronoi(pts) on your own pt_cloud to be able to use this function.  The initial pts have to be a (n,3) numpy array of the SKIN of a person, not every voxel in their body whole 
    returns tuple containing:
      (0th) an np array of shape (n,3), where n is the number of input points plus the number of dummies.  Normal vectors to each pt in the input pt_cloud
      (1st) an np array of shape (n),   where n is the number of input points plus the number of dummies.  Confidences in each normal

    Notes
    -----
    When in doubt, consult the source:   http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf
    I use plenty of shorthand from the Alliez 2007 paper; it keeps the variable names from getting unwield-ily long and cumbersome

    As of Jan. 2, 2019, I call a 3d Voronoi region a "Vorohedron" b/c it's shorter than "Voronoi polyhedron"
    Vorohedra are always convex (https://en.wikipedia.org/wiki/Voronoi_diagram):
      "[if] the space is a finite-dimensional Euclidean space (our case), each site is a point, there are finitely many points and all of them are different, then the Voronoi cells are CONVEX polytopes." (-Wikipedia)
      proof: (https://math.stackexchange.com/questions/269575/does-voronoi-tessellation-in-3d-always-produce-convex-polyhedrons)
  '''
  # TODO: refactor into separate, descriptive functions
  func_name      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+func_name)

  # CONSTANTS
  # TODO: consistently put names like INF_BOUND into everything
  INF_BOUND=-1

  # Below are the data structures we're storing the calculated normal vectors, confidences, centroids (CoMs), volumes, and covariances (covars) in.  "Norms," "normals," "covars," and "covariances" all refer to the same thing.
  # voro_vols, voro_CoMs, and voro_covars all record individual Vorohedrons' measurements, not those of unions of multiple vorohedra
  norms       = np.zeros((vor.npoints,3)   ).astype('float64')
  confidences = np.zeros( vor.npoints      ).astype('float64') # NOTE: equivalent to "pointness of this point's vorohedron"
  print('\n'*3); print("="*99)
  print("starting to calculate normal vectors")
  print("="*99); print('\n'*3)

    # NOTE: The reindexing below (region_idx != pt_idx) is important.  Voronoi() doesn't index vor.regions with the same indices as the vor.points.  This was an early error I made (January 2, 2018).  
    #       Indexing bugs have been pretty common in my code development so far, as have reshaping (broadcasting) bugs and transpose bugs (ie. M.T).  Many levels of nesting almost always kill my understanding of the details
    #       Why is len(vor.regions) > len(vor.points)?  As far as I can tell, vor.regions often has to have an empty region.  Unsure why
  for pt_idx in range(len(vor.points)):
    region_idx      = vor.point_region[pt_idx]; region          = vor.regions[region_idx]; pt              = vor.points [pt_idx] # imho, "pt" is enough specificifity here.  Otherwise, we could have called it "pt_from_pt_cloud" or "skin_pt"
    voro_verts=vor.vertices[region]
    if INF_BOUND in region:
      continue  # when we've added dummy pts correctly, this will only happen for dummy (edge) points
    elif len(voro_verts)==0: # Don't use this pt as data in consideration of the norms  
      continue  # to next point in skin's pt_cloud
      # Sometimes qhull returns empty lists of vertices.  I believe this is because of coplanar points, as qhull uses Delaunay triangulation to calculate the Vorohedrons.  The current default behavior is to leave the norms as <0,0,0> in this case, which is how they were initialized in norms = np.zeros(...).astype(...)
      #   More details on why I believe this can be found [here](https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html), under the heading "Coplanar points" (As of Jan. 3, 2018)
    else: # calculate normal
      hull=scipy.spatial.ConvexHull(voro_verts)
      #hull.volume    is a thing precalculated here   in case you decide you want it
      max_dist=float('-inf');pole=None
      # NOTE: I thought about throwing a KDTree() in here to help me sort through the vorohedron's vertices, but there aren't too many of them
      for vert in voro_verts:
        dist=LA.norm(vert-pt)
        if dist>max_dist: 
          max_dist=dist
          pole=vert
      norms[pt_idx]=pole-pt  # we might wanna invest in calculating the CoM of the vorohedron instead of just doing the "-pt" minus "pole-pt"
      confidences[pt_idx]=LA.norm(norms[pt_idx])   /hull.volume
  # end "for pt_idx in range(len(vor.points)):" }
  pif(  'Exiting '+func_name); #print('Exiting '+func_name)
  return norms, confidences
# end func def of   norms(vor):
#=========================================================
def main(model, norms_save_filename, confidences_save_filename):
  model = add_dummies(model)
  locs=np.nonzero(model); locs=np.array(locs).T.astype('int64')
  print("locs with dummies: \n{0}".format(locs))
  print("num_pts: \n{0}".format(locs.shape[0]))
  # vor step of voro_poles
  vor                   = scipy.spatial.Voronoi(locs)
  normals, confidences  = norms(vor)
  if save:
    np.save(norms_save_filename, normals)
    np.save(confidences_save_filename, confidences)
# end func def of   main():
#=========================================================
def five_pt_plane():
  plane=np.zeros((19,19,19)).astype('bool')
  plane[ 7,10,12]=True # x==4 because we think there may be a calculation problem if we only do coplanar points
  plane[12,10, 8]=True
  plane[10,10,10]=True
  plane[10, 8,12]=True
  plane[10,12, 8]=True
  return plane
#=========================================================
def plane():
  plane=np.zeros((19,19,19)).astype('bool')
  for x in range(6,13):
    for y in range(6,13):
      for z in range(6,13):
        if x==y:
          plane[x,y,z]=True
  return plane
#=========================================================
if __name__=='__main__':
  model = np.load('skin_nathan_.npy').astype('bool')
  main(model, 'norms_nathan_.npy')
#=========================================================







# calling Voronoi() on 831259 on locs,  shape==(513, 513, 513)
  #
  # time: 
  #
  #   real  1m3.208s
  #   user  1m2.044s
  #   sys   0m1.344s
  #


























































