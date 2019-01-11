
import scipy.spatial as sp
from norms_____by_voro_poles import *
from mesh import *

#=========================================================
def seven_pt_plane():
  plane=np.zeros((29,29,29)).astype('bool')
  plane[15,15,15]=True
  plane[15,15, 6]=True
  plane[15,15,24]=True  # oughta be stretched out in this direction so we get a clear elongation
  plane[15,12,15]=True
  plane[15,18,15]=True
  plane[12,15,15]=True
  plane[18,15,15]=True
  return plane
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



'''
def main():
  model=add_dummies(seven_pt_plane)
  locs=np.nonzero(model)
  vor=sp.Voronoi(locs)
  vor.points
'''

if __name__=="__main__":
  model=add_dummies(seven_pt_plane())
  locs=np.nonzero(model); locs=np.array(locs).T.astype('float64')
  locs+=(np.random.random(locs.shape).astype('float64')  *0.1)
  vor=sp.Voronoi(locs)
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
  for pt_idx in range(len(vor.points)):
    region_idx = vor.point_region[pt_idx]; region = vor.regions[region_idx]; pt = vor.points[pt_idx]
    voro_verts=vor.vertices[region]
    if pt_idx==12:
    #input("pt_idx is {0} and  \npt is ".format(pt_idx, pt))  #NOTE: hack-y way of pausing (ghetto debugger)
      neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
      neighbor_idx    = 0
      aniso_max       = float('-inf')
      aniso_curr      = float('-inf')
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
            print("unions_covar is {0}".format(unions_covar))
        # end else: (this block calculated the covariance, CoM, and volume to store in the np arrays declared before the loop (voro_covars, voro_vols, voro_CoMs))
        neighbor_idx+=1
