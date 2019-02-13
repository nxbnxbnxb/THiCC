# "Voronoi-based Variational Reconstruction of Unoriented Point Sets"
#     by Alliez, Cohen-Steiner, Tong, and Desbrun

# link: http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf

from mesh import *
# above includes numpy, scipy.spatial, etc.

"""
#=========================================================
def plane(slope=1.0, axis='x', n_pts=1000):
  '''
    Returns an inclined plane.  For the purposes of testing norms made by mesh.py (the norm-generating process of the Alliez paper: http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf)
    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
  '''
  plane = np.zeros((3,n_pts)).astype('float64')
  X=0; Y=1; Z=2
  which_pt=0
  if axis=='x':
    # NOTE:  can vectorize this for-loop code  TODO: (but we don't really care about performance on such a tiny function; it shouldn't take too long)
    for which_pt in range(n_pts):
      for y in range(-n_pts, n_pts, 2):
        plane[which_pt, Y]=y
        plane[which_pt, Z]=slope*y
    plane[:,X] = np.random.random(n_pts)*2*n_pts - n_pts
    magnitude  = n_pts/2.0
    noise =  magnitude * np.random.random((3,n_pts)).astype('float64')\
           - madnitude/2.0  # centered on zero
    plane+=noise
  # TODO: if axis=='y':    , if axis=='z':,  slope
  return plane
#=========================================================
"""
def noisy_plane(slope=1.0, axis='x', n_pts=1000):
  '''
    Returns an inclined plane.  For the purposes of testing norms made by mesh.py (the norm-generating process of the Alliez paper: http://www.lama.univ-savoie.fr/pagesmembres/lachaud/Memoires/2012/Alliez-2007.pdf)
    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
  '''
  plane = np.zeros((3,n_pts)).astype('float64')
  X=0; Y=1; Z=2
  which_pt=0
  if axis=='x':
    # NOTE:  can vectorize this for-loop code  TODO: (but we don't really care about performance on such a tiny function; it shouldn't take too long)
    for which_pt in range(n_pts):
      for y in range(-n_pts, n_pts, 2):
        plane[which_pt, Y]=y
        plane[which_pt, Z]=slope*y
    plane[:,X] = np.random.random(n_pts)*2*n_pts - n_pts
    magnitude  = n_pts/2.0
    noise =  magnitude * np.random.random((3,n_pts)).astype('float64')\
           - madnitude/2.0  # centered on zero
    plane+=noise
  # TODO: if axis=='y':    , if axis=='z':,  slope
  return plane
#=========================================================
def test___norms():
  surface = noisy_plane()
  vor=Voronoi

#=========================================================
def sphere_shell():
  shell=np.zeros((19,19,19)).astype('bool')
  for i in range(19):
    for j in range(19):
      for k in range(19):
        if round((i-9)**2 + (j-9)**2 + (k-9)**2) == 25:
          shell[i,j,k]=True
  return shell
#=========================================================
def sphere_shell_probabilistic():
  shell=np.zeros((19,19,19)).astype('bool')
  for i in range(19):
    for j in range(19):
      for k in range(19):
        if round((i-9)**2 + (j-9)**2 + (k-9)**2) == 25:
          if np.random.random() < 0.9:
            shell[i,j,k]=True
  return shell
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

if __name__=="__main__":
  #np.save("sphere_shell.npy", sphere_shell())   only needed to do once

  '''
  # sphere:
  main(sphere_shell_probabilistic(), 'norms_sphere.npy')
  '''
  #np.save("plane.npy", plane())   #only needed to do once
  main(plane(), 'norms_plane.npy')


