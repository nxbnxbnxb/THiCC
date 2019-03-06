
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
# TODO:  check the results of the Voronoi() function
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

# TODO:   np.eigh() or np.eigvalsh() may work faster/more reliably for a symmetric (or Hermitian: conjugate symmetric) matrix
#
#
#
#
#
#
#
#
#
# TODO:  check whether we should take the max of abs(np.linalg.eigvals()) or the maximum POSITIVE eigval()    (when picking the maximum eigenvalue that gives us the anisotropy, etc.)  I THINK max abs makes more sense, but I'm not 100% sure
#
#


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
def ZERO_NORM():
  return np.eye(3).astype("float64")
#=========================================================
def max_eig_vect(covar):
  '''
    returns the eigenvector associated with the largest eigenvalue of the 3x3 input matrix.

    -----
    Notes:
      should be generalizable to any 3x3 matrix, but we don't particularly care right now
  '''
  # NOTE:  on Jan. 8, 2018, this call gave the Warning   "mesh.py:328: ComplexWarning: Casting complex values to real discards the imaginary part."  TODO: figure out why, workaround it.
  vals, vects = LA.eig(covar)
  return vects[:,vals.argmax()]
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
    returns np array of shape (n,3,3), where n is the number of input points.  I know a 3x3 matrix isn't a normal vector in the traditional physics sense, but I bet Alliez et al. have a reason to use these 3x3s instead of 3x1s normal vectors

    Notes
    -----
    When in doubt, consult the source:   http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf
    I use plenty of shorthand from the Alliez 2007 paper; it keeps the variable names from getting unwield-ily long and cumbersome

    As of Jan. 2, 2019, I call a 3d Voronoi region a "Vorohedron" b/c it's shorter than "Voronoi polyhedron"
    Vorohedra are always convex (https://en.wikipedia.org/wiki/Voronoi_diagram):
      "[if] the space is a finite-dimensional Euclidean space (our case), each site is a point, there are finitely many points and all of them are different, then the Voronoi cells are CONVEX polytopes." (-Wikipedia)
      proof: (https://math.stackexchange.com/questions/269575/does-voronoi-tessellation-in-3d-always-produce-convex-polyhedrons)
  '''
  # TODO:   more explicit type casting:  ie.  .astype('float64'), .astype('int64')
  # TODO: refactor into separate, descriptive functions
  header      = 'Models()\'s {0} function'.format(sys._getframe().f_code.co_name); pif('Entering '+header)

  # CONSTANTS
  # TODO: consistently put names like BELOW into EVERYTHING
  Q=np.array([[2,1,1],
              [1,2,1],
              [1,1,2]]).astype('float64')/120.0  # referenced in Alliez 2007 paper
  K=5#50
  INF_BOUND=-1
  FIRST_2_PTS=2
  BESIDE=0
  BELOW=1
  UNDER=0
  DOWN=0
  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices

  # Data structures we're storing the calculated centroids (CoMs), volumes, and covariances (covars) in.  "Norms," "normals," "covars," and "covariances" all refer to the same thing.
  norms       = np.zeros((vor.npoints,3)   ).astype('float64')
  final_covars= np.zeros((vor.npoints,3,3) ).astype('float64')#    <-- "final_covars" will be the final return value;   stands for "normal vectors'
  confidences = np.zeros( vor.npoints      ).astype('float64') # anisotropies
  voro_covars = np.zeros((vor.npoints,3,3) ).astype('float64')
  voro_vols   = np.zeros( vor.npoints      ).astype('float64')
  voro_CoMs   = np.zeros((vor.npoints,3  ) ).astype('float64')
  # voro_vols, voro_CoMs, and voro_covars all record individual Vorohedrons' measurements, not those of unions of multiple vorohedra
  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  
  # TODO:  consider use of KDTree again  (high space complexity)
  # NOTE:   Jan. 3, 2018; memory usage is pretty reasonable with 85215 skin data points from a shape==(513, 513, 513) model from .png files with max dimension (in numpy) 513

  first = True
  # TODO: make sure in EVERY exit case, we do everything we need to get the norm, the covariances, and the confidence, etc.   Check every "continue" statement, every "break," etc.
  pif("len(vor.points) is \n{0}".format(len(vor.points)))
  pif("len(vor.regions) is \n{0}".format(len(vor.regions)))  # these 2 are fine
  dummy_pt_count=0
  for pt_idx in range(len(vor.points)):
    region_idx      = vor.point_region[pt_idx]
    # NOTE: This reindexing (region_idx != pt_idx) is important.  Voronoi() doesn't index vor.regions with the same indices as the vor.points.  This was an early error I made (January 2, 2018).  
    #       Indexing bugs have been pretty common in my code development so far, as have reshaping (broadcasting) bugs.  Many levels of nesting almost always kill my understanding of the details
    #       Why is len(vor.regions) > len(vor.points)?  As far as I can tell, vor.regions often has to have an empty region.  Unsure why
    region          = vor.regions[region_idx]
    pt              = vor.points [pt_idx]
    if INF_BOUND in region:
      dummy_pt_count+=1
      pif("{0}th dummy pt".format(dummy_pt_count)) # fine
      continue  # when we've added dummy pts correctly, this will only happen for dummy (edge) points
    neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
    neighbor_idx    = 0
    aniso_max       = float('-inf')
    aniso_curr      = float('-inf')

    # TODO:  consider whether it's easier to use a for loop  (ie. (for i in range(BIG))):    so the "continue"s automatically advance the index, but then we gotta throw a "break" in there if anitotropy > 0.9
    # NOTE:  the following while loop calculates the covariance of the union of this point with neighboring points
    while neighbor_idx < K and aniso_max < 0.9:
      macro_neighbor_idx=neighbors_idxes[neighbor_idx]
      union_so_far= neighbors_idxes[:neighbor_idx+1]  # NOTE:  the +1 is there for a reason.  Consider if neighbor_idx was 0; this wouldn't return anything without the +1.  And if neighbor_idx were K-1 you'd still be missing a neighbors without the +1
      # TODO:   if we terminate by neighbor_idx==K, take the covar w/ max anisotropy.  I THINK this is done.
      # I had as a todo:  "switch order of if and else:  (dad says "elses" should be shorter)."   
      # BUT this way makes the if statement "if np.any()," which is much easier to understand than "if not np.any()"
      if np.any(voro_covars[macro_neighbor_idx]):
        # TODO:  neighbors_idxes[:neighbor_idx+1] ==> cumulative_neighbors_idxes  (something shorter, but to this effect)
        unions_vol  = np.sum(voro_vols[union_so_far])
        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  * voro_CoMs[union_so_far],  axis=DOWN) / unions_vol
        assert unions_CoM.shape == (3,)
        p_i         = (voro_CoMs[macro_neighbor_idx] - unions_CoM).reshape((3,1))  # we want p_i.dot(p_i.T) to have shape (3,3)[3x3 matrix].   (3,1)x(1,3) ==> (3,3)
        # p_i*p_i.T is symmetric with respect to unions_CoM - voros_CoM.   So it doesn't matter which order we do here
        m_i         = voro_vols[macro_neighbor_idx]
        shifts      = np.tile((m_i*p_i.dot(p_i.T)),(neighbor_idx+1,1,1)).astype('float64')
        unions_covar= np.sum( voro_covars[union_so_far] - shifts,  axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
        # shifts.shape                      is (n, 3, 3)
        # voro_covars[union_so_far].shape   is (n, 3, 3)
        # unions_covar.shape                is    (3, 3)

        assert len(shifts.shape)==3 and   shifts.shape[1] == shifts.shape[2] == 3
        assert shifts.shape == voro_covars[union_so_far].shape
        assert unions_covar.shape == (3,3)
        # TODO:   is it faster to use a running covariance rather than calculating it this way?  I bet the running covariance IS faster, though it'll resemble the paper less than this way does
        aniso_curr        = anisotropy(unions_covar)
        if aniso_curr > aniso_max:
          aniso_max = aniso_curr
          final_covars[pt_idx]=unions_covar
        neighbor_idx     += 1
        continue  # next iteration of the inner "while" which calculate the covariance of the union of points nearby the pt we've iterated to in the outer "for" loop
      # end "if np.any(voro_covars[macro_neighbor_idx]):"   (covars were calculated in a previous loop):
      else:  # if covars haven't been calculated yet, calculate vorohedron's covariance.    To do that, first we need to calc its volume and centroid
        if first:
          print ("calculating first covariance")
          first=False
        vertices = vor.vertices[region]  # TODO: double-check indexing
        if len(vertices) == 0: # Don't use this pt as data in consideration of the norms  
        # Sometimes qhull returns empty lists of vertices.  I believe this is because of coplanar points, as qhull uses Delaunay triangulation to calculate the Vorohedrons
        #   More details on why I believe this can be found [here](https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html), under the heading "Coplanar points" (As of Jan. 3, 2018)
          final_covars[pt_idx]=np.eye(3) # TODO: make sure this is the right thing to do here.  I'm STILL not 100% sure why qhull sometimes returns [] under vor.regions.  Once we figure it out, we can treat the code appropriately
          neighbor_idx   += 1
          continue # next iteration of the inner "while" which calculate the covariance of the union of points nearby the pt we've iterated to in the outer "for" loop
        hull=scipy.spatial.ConvexHull(vertices)
        voro_vols[macro_neighbor_idx]=hull.volume
        vol_voro=voro_vols[macro_neighbor_idx]  # NOTE:  this pointer 'vol_voro' allows us to mutate the bigger array without repeatedly typing the cumbersome indices
                                                #       I did the above operation in 2 lines because otherwise I don't get a pointer to the nparr that has a short name.
        triangle_mesh_hull=vertices[hull.simplices] # triangle_mesh_hull.shape == (n,3,3)   NOTE: can this be right?  how would this variable "vertices" include ALL the triangles necessary to make a mesh surrounding this vorohedron???
        # TODO: double-check the shape of the above triang_mesh_hull.  I think my earlier "understanding" that "triangle_mesh_hull.shape == (n,3,3)" was wrong.
        # TODO: rename "triangle_mesh_hull" ==> "triangs" or something shorter.  problem is "mesh" and "hull" help describe this object
        assert len(triangle_mesh_hull.shape)==3 and   triangle_mesh_hull.shape[1] == triangle_mesh_hull.shape[2] == 3
        inner_pt  = np.mean(vertices[:FIRST_2_PTS],axis=UNDER).reshape((1,3)) # new tetrahedron vertex.   In the beginning this is good enough because we're guaranteed Vorohedrons are convex
        CoM_voro  = voro_CoMs[macro_neighbor_idx] # NOTE:  this pointer allows us to mutate the bigger array without repeatedly typing the cumbersome indices.  Initially CoM_voro is full of zeros
        # next we 1. find the volume of each tetrahedron,  so we can 2. find the CoM of each tetrahedron  so we can 3. find the CoM of the whole Vorohedron
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

        # The vorohedron's overall covariance is the sum of its internal tetrahedrons' covariances.  See http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf, Appendix A
        for triangle in triangle_mesh_hull:
          tetra=np.concatenate((CoM_voro.reshape((1,3)),triangle),axis=UNDER)
          N=(triangle.T  - np.vstack((CoM_voro,CoM_voro,CoM_voro)).astype('float64')).T
          cov_voro += np.linalg.det(N)*np.dot(N,Q,N.T)  # TODO: ensure these 3 pointers (including cov_voro, but also the 2 others like it)    actually mutate the stored values
        unions_vol  = np.sum(voro_vols[union_so_far])
        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  *voro_CoMs[union_so_far],axis=DOWN) / unions_vol
        assert unions_CoM.shape == (3,)
        #p_i         = voro_CoMs[neighbors_idxes[:neighbor_idx+1]]  - np.tile(unions_CoM.reshape((1,3)),(neighbor_idx+1,BELOW)).T # p_i*p_i.T is symmetric w.r.t. unions_CoM - voros.   So it doesn't matter which order we do here
        #m_i         = voro_vols[neighbors_idxes[:neighbor_idx+1]]   # TODO: double-check the dimensions here.  I'm guessing my more recent edit is right and the older one is wrong, but I'm not 100% sure
        p_i         = (voro_CoMs[macro_neighbor_idx] - unions_CoM).reshape((3,1))  # we want p_i.dot(p_i.T) to have shape (3,3)[3x3 matrix].   (3,1)x(1,3) ==> (3,3)
        m_i         = voro_vols[macro_neighbor_idx]
        pif("np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)).shape is {0}".format(np.tile((m_i*np.dot(p_i, p_i.T)),(neighbor_idx+1,1,1)).shape))
        shifts      = np.tile((m_i*p_i.dot(p_i.T)),(neighbor_idx+1,1,1)).astype('float64')
        unions_covar= np.sum(  voro_covars[union_so_far] - shifts,axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
        # TODO:                                       is axis=0 right or axis=2?  In the earlier line I wrote 2 and I was probably more awake when I wrote that
        assert unions_covar.shape == (3,3)
        # TODO:   is it faster to use a running covariance rather than calculating it this way?  I bet the running covariance IS faster, though it'll resemble the paper less than this way does
        aniso_curr        = anisotropy(unions_covar) # TODO:  all this stuff in the "else:" branch
        if aniso_curr > aniso_max:
          aniso_max = aniso_curr
          final_covars[pt_idx]=unions_covar
      # end else: (this block calculated the covariance, CoM, and volume to store in the np arrays declared before the loop (voro_covars, voro_vols, voro_CoMs))
      neighbor_idx+=1
      # tail condition: while loop is about to terminate
    # TODO:  ensure that   "if neighbor_idx == K: return covar with max anisotropy".  I'm pretty sure we did this
    # end inner loop "while anisotropy() < 0.9 and   neighbor_idx < K:"
    assert aniso_max != float('-inf')  # if it still is, do, uh....  idk, more debugging
    # NOTE:  these confidences and norms are calcul8d after the termination of the while loop.  So for each continue within the while loop, we don't have to get a confidence or a norm
    confidences[pt_idx] = aniso_max
    # TODO: if this eig calc is slowing everything down, move norms[pt_idx]=max_eig_vect() outside the for loop and vectorize eig() calculation as mentioned [here](https://stackoverflow.com/questions/19468889/vectorize-eigenvalue-calculation-in-numpy)
    import warnings
    warnings.filterwarnings("error")
    try:
      norms[pt_idx]       = max_eig_vect(final_covars[pt_idx])
    except np.core.numeric.ComplexWarning:
      print("ComplexWarning")
      print("final_covars[pt_idx] = \n{0}".format(final_covars[pt_idx]))
      print("pt                  is \n{0}".format(pt    ))
      print("pt_idx              is \n{0}".format(pt_idx))
      print("neighbor_idx        is \n{0}".format(neighbor_idx))
      print("aniso_max           is \n{0}".format(aniso_max))
      print("final_covars[pt_idx] = \n{0}".format(final_covars[pt_idx]))
      # NOTE: this happens a lot for pretty much any input
  # end "for pt_idx in range(len(vor.points)):" }

  pif(  'Exiting '+header)
  print('Exiting '+header)
  return norms
  # TODO: normalize each covariance.
  # NOTE: As of Dec. 31, 2018, the whole thing is really waaaaay too slow.  TODO: speed it up.  There's gotta be a faster way
# end func def of   norms(vor):
#=========================================================
def main(model, savefilename):
  model = add_dummies(model)
  locs=np.nonzero(model); locs=np.array(locs).T.astype('int64')
  print("locs with dummies: \n{0}".format(locs))
  print("locs.shape: \n{0}".format(locs.shape))
  # vor step of Alliez et al.   (for estimating normals from point cloud)
  vor=scipy.spatial.Voronoi(locs)
  # TODO:  check the results of the Voronoi() function
  normals=norms(vor)
  if save:
    np.save(savefilename, normals)
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
if __name__=='__main__':
  #model = np.load('skin_nathan_.npy').astype('bool')
  #main(model, 'norms_nathan_.npy')
  main(five_pt_plane(), 'norms_plane_5_pts.npy')
#=========================================================







# calling Voronoi() on 831259 on locs,  shape==(513, 513, 513)
  #
  # time: 
  #
  #   real  1m3.208s
  #   user  1m2.044s
  #   sys   0m1.344s
  #


























































