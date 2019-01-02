
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
# TODO:  check the results of the Voronoi() function
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
#=========================================================
def anisotropy(matrix):
  '''
    Basically the "pointiness" of the normal
    Described in Alliez 2007
  '''
  # TODO:   there are plenty of weird exceptions that might be thrown surrounding zeros, weird inputs, etc.
  if type(matrix) == type(np.zeros((2,2))):
    pif("matrix.shape is {0}".format(matrix.shape))
  else:
    pif(matrix)
  assert len(matrix.shape)==2 and matrix.shape[0]==matrix.shape[1]
  eigs=np.abs(np.linalg.eigvals(matrix))
  if np.any(eigs):
    return 1-abs(np.min(eigs)/np.max(eigs))  # IRL should never be 1; that would mean an infinitesimally thin slice of volume
  else:  # zero matrix
    return 0
#=========================================================
def norms(vor):
  '''
    Calculates the norms at each real (not dummy) point  from the input data pt_cloud.

    Parameters
    ----------
    vor is an object returned by the scipy.spatial.Voronoi() function;  you'll have to call Voronoi(pts) on your own pt_cloud to be able to use this function.  The initial pts have to be a (n,3) numpy array of the SKIN of a person, not every voxel in their body whole 
    returns np array of shape (3,3,n), where n is the number of input points.  I know a 3x3 matrix isn't a normal vector in the traditional physics sense, but I bet Alliez et al. have a reason to use these 3x3s instead of 3x1s normal vectors

    Notes
    -----
    When in doubt, consult the source:   http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf
    I use plenty of shorthand from the Alliez 2007 paper; it keeps the variable names from getting unwield-ily long and cumbersome

    As of Jan. 2, 2019, I call a 3d Voronoi region a "Vorohedron" b/c it's shorter than "Voronoi polyhedron"
    Vorohedra are always convex (https://en.wikipedia.org/wiki/Voronoi_diagram):
      "[if] the space is a finite-dimensional Euclidean space (our case), each site is a point, there are finitely many points and all of them are different, then the Voronoi cells are CONVEX polytopes." (-Wikipedia)
      proof: (https://math.stackexchange.com/questions/269575/does-voronoi-tessellation-in-3d-always-produce-convex-polyhedrons)
  '''
  # TODO: Thoroughly consider whether it's better to call everything covariances or norms?
  # TODO:   more explicit typing:  ie.  .astype('float64'), .astype('int64')
  # TODO: refactor into separate, descriptive functions
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  # CONSTANTS
  # TODO: consistently put names like BELOW into EVERYTHING
  Q=np.array([[2,1,1],
              [1,2,1],
              [1,1,2]]).astype('float64')/120.0  # referenced in Alliez 2007 paper
  K=50
  INF_BOUND=-1
  FIRST_2_PTS=2
  BESIDE=0
  BELOW=1
  UNDER=0
  DOWN=0
  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices

  # caches
  norms       = np.zeros((len(vor.regions),3,3) ).astype('float64')#    <-- "norms" will be the final return value;   stands for "normal vectors'
  voro_covars = np.zeros((len(vor.regions),3,3) ).astype('float64')
  voro_vols   = np.zeros( len(vor.regions)      ).astype('float64')
  voro_CoMs   = np.zeros((len(vor.regions),3  ) ).astype('float64')
  # voro_vols, voro_CoMs, and voro_covars are all INDIVIDUAL measurements (not those of unions of multiple vorohedra)
  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  TODO:  consider again

  for pt_idx in range(len(vor.regions)):
    pt              = vor.points [pt_idx]
    region          = vor.regions[pt_idx]
    if INF_BOUND in region:
      continue  # when we've added dummy pts correctly, this will only happen for dummy (edge) points
    neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
    neighbors       = vor.points[neighbors_idxes]
    neighbor_idx    = 0 # calculate for self?  should below be <= k or just < k?  TODO: check that the kdtree will return a pt from within the tree.  I'm 90% sure it will, but plz double-check
    aniso_max       = float('-inf')
    aniso_curr      = float('-inf')
    while neighbor_idx < K and anisotropy(norms[pt_idx]) < 0.9:
      macro_neighbor_idx=neighbors_idxes[neighbor_idx]
      # TODO:   if we terminate by neighbor_idx==K, take the covar w/ max anisotropy
      # TODO:  switch order of if and else:  (dad says "elses" should be shorter)
      if np.any(voro_covars[macro_neighbor_idx]):
        unions_vol  = np.sum(voro_vols[neighbors_idxes[:neighbor_idx+1]])  # NOTE:  +1 because if neighbor_idx was 0 this wouldn't return anything and if it were K-1 you'd get all the neighbors_indices
        unions_CoM  = np.sum(voro_vols[neighbors_idxes[:neighbor_idx+1]].reshape((neighbor_idx+1,1))*voro_CoMs[neighbors_idxes[:neighbor_idx+1]]) / unions_vol
        assert unions_CoM.shape == (3,)
        p_i         = voro_CoMs[neighbors_idxes[:neighbor_idx+1]]  - np.tile(unions_CoM.reshape((1,3)),(neighbor_idx+1,BELOW)) # p_i*p_i.T is symmetric w.r.t. unions_CoM - voros.   So it doesn't matter which order we do here
        m_i         = voro_vols[neighbors_idxes[:neighbor_idx+1]]
        unions_covar= np.sum(  voro_covars[neighbors_idxes[:neighbor_idx+1]]  -  np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)),axis=2) # calculation from Alliez paper (covars of unions of vorohedra)
        # TODO:   is it faster to use a running covariance rather than calculating it this way?  I bet the running covariance IS faster, though it'll resemble the paper less than this way does
        aniso_curr        = anisotropy(unions_covar) # TODO:  all this stuff in the "else:" branch
        if aniso_curr > aniso_max:
          aniso_max = aniso_curr
          norms[pt_idx]=unions_covar
        neighbor_idx     += 1
        continue
        # TODO:  consider whether it's easier to use a for loop  (ie. (for i in range(BIG))):    so the "continue"s automatically advance the index, but then we gotta throw a "break" in there if anitotropy > 0.9
      # end if (covars were stored):  (calculate the vorohedron's covar by 1st calcul8ing   volume and CoM

      else:  # calculate volumes, CoMs, and covars for this vorohedron for the 1st time
        vertices = vor.vertices[region]
        if len(vertices) == 0:
          # don't use this pt as data in consideration of the norms  (the failure of qhull to compute the Voronoi region   probably means there's some weirdness with coplanar points)
          neighbor_idx   += 1
          continue
        hull=scipy.spatial.ConvexHull(vertices)
        voro_vols[macro_neighbor_idx]=hull.volume
        vol_voro=voro_vols[macro_neighbor_idx] # NOTE:  this pointer allows us to mutate the bigger array without repeatedly typing the cumbersome indices
        # NOTE:  I did the above operation in 2 lines because otherwise I don't get a pointer to the nparr that has a short name.
        triangle_mesh_hull=vertices[hull.simplices] #(n,3,3)
        inner_pt  = np.mean(vertices[:FIRST_2_PTS],axis=UNDER).reshape((1,3)) # new tetrahedron vertex.   In the beginning this is good enough because we're guaranteed Vorohedrons are convex
        CoM_voro  = voro_CoMs[macro_neighbor_idx] # NOTE:  this pointer allows us to mutate the bigger array without repeatedly typing the cumbersome indices
        # calc vol of tetrahedron    so we can calc CoM of each tetrahedron   so we can calc CoM of whole Vorohedron
        for triangle in triangle_mesh_hull:
          ''' print(triangle) => [[x1, y1, z1],
                                  [x2, y2, z2],
                                  [x3, y3, z3]]             '''
          tetra     = np.concatenate((inner_pt,triangle),axis=UNDER)
          CoM_tetra = np.mean(tetra,axis=DOWN)
          vol_tetra = volume_tetra(tetra)
          CoM_voro += CoM_tetra*vol_tetra
        CoM_voro   /= vol_voro
        cov_voro    = voro_covars[macro_neighbor_idx]  # THIS vorohedron's covariance.   NOTE:  this pointer allows us to mutate the bigger array without repeatedly typing the cumbersome indices
        # vorohedron's covar is the sum of the internal tetrahedrons' covars.
        for triangle in triangle_mesh_hull:
          tetra=np.concatenate((CoM_voro.reshape((1,3)),triangle),axis=UNDER)
          N=(triangle  - np.vstack((CoM_voro,CoM_voro,CoM_voro)).astype('float64')).T
          cov_voro += np.linalg.det(N)*np.dot(N,Q,N.T)
        unions_vol  = np.sum(voro_vols[neighbors_idxes[:neighbor_idx+1]])
        pif("voro_vols[neighbors_idxes[:neighbor_idx+1]].shape is {0}".format(voro_vols[neighbors_idxes[:neighbor_idx+1]].shape))
        pif("voro_CoMs[neighbors_idxes[:neighbor_idx+1]].shape is {0}".format(voro_CoMs[neighbors_idxes[:neighbor_idx+1]].shape))
        unions_CoM  = np.sum(voro_vols[neighbors_idxes[:neighbor_idx+1]].reshape((neighbor_idx+1,1))*voro_CoMs[neighbors_idxes[:neighbor_idx+1]],axis=DOWN) / unions_vol
        assert unions_CoM.shape == (3,)
        p_i         = voro_CoMs[neighbors_idxes[:neighbor_idx+1]]  - np.tile(unions_CoM.reshape((1,3)),(neighbor_idx+1,BELOW)).T # p_i*p_i.T is symmetric w.r.t. unions_CoM - voros.   So it doesn't matter which order we do here
        m_i         = voro_vols[neighbors_idxes[:neighbor_idx+1]]
        pif("p_i.shape is {0}".format(p_i.shape))
        pif("m_i.shape is {0}".format(m_i.shape))
        pif("voro_covars[neighbors_idxes[:neighbor_idx+1]].shape is {0}".format(voro_covars[neighbors_idxes[:neighbor_idx+1]].shape))
        pif("np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)).shape is {0}".format(np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)).shape))
        unions_covar= np.sum(  voro_covars[neighbors_idxes[:neighbor_idx+1]]  -  np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)),axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
        # TODO:   is it faster to use a running covariance rather than calculating it this way?  I bet the running covariance IS faster, though it'll resemble the paper less than this way does
        pif("unions_covar.shape is {0}".format(unions_covar.shape))
        pif()
        aniso_curr        = anisotropy(unions_covar) # TODO:  all this stuff in the "else:" branch
        if aniso_curr > aniso_max:
          aniso_max = aniso_curr
          norms[pt_idx]=unions_covar
              #norms[pt_idx]+=cov_voro #TODO:   delete
        # TODO:  adjust for volume of union region
      # end else: (covars weren't stored)
      neighbor_idx+=1
    # TODO:  if neighbor_idx == K: return covar with max anisotropy  
    # end while anisotropy() < 0.9 and   neighbor_idx < K:
  return norms
  # TODO: normalize each normal vector.   Question for Alliez et al.: Doesn't normalization reduce the degree to which one can be "sure" of the pointiness at each point in the pt cloud?
  # NOTE: As of Dec. 31, 2018, the whole thing is really waaaaay too slow.  TODO: speed it up.  There's gotta be a faster way
# end func def of   norms(vor):
#=========================================================
def main():
  model = np.load('skin_nathan_.npy').astype('bool')
  model = add_dummies(model)

  locs=np.nonzero(model); locs=np.array(locs).T.astype('int64')
  # vor step of Alliez et al.   (for estimating normals from point cloud)
  vor=Voronoi(locs)
  # TODO:  check the results of the Voronoi() function
  normals=norms(vor)
  print(normals.shape)
# end func def of   main():
#=========================================================
if __name__=='__main__':
  main()
#=========================================================







# calling Voronoi() on 831259 on locs,  shape==(513, 513, 513)
  #
  # time: 
  #
  #   real  1m3.208s
  #   user  1m2.044s
  #   sys   0m1.344s
  #







































