import numpy as np
np.seterr(all='raise')
from skimage import measure
from skimage.draw import ellipsoid

# show model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#============================================================
def mesh_from_pt_cloud(model_np_arr):
  '''
    Marching Cubes algo; rudimentary point cloud ---> mesh algorithm
    The format we've been using works just fine as input;
      3D np arr (ie. arr.shape==(513,513,513)) where each arr[i,j,k] is either 0 (no body) or 1 (body)

    Timing: model of shape (513,513,513)
      real  0m4.610s
      user  0m3.060s
      sys 0m0.369s

      wow, fast!
  '''
  verts, faces, normals, values = measure.marching_cubes_lewiner(model_np_arr, 0)
  return verts, faces
#============================================================
def save_mesh(pt_cloud, faces_filename, verts_filename):# TODO: move this func from m_cubes to somewhere more sensible; we ought to be able to generalize the mesh-generation beyond mcubes in case we want to later do voro-vari, SMPL, etc.  (some other mesh-generating technique)
  verts, faces, normals, values = measure.marching_cubes_lewiner(pt_cloud, 0)
  np.save('faces_nathan_.npy', faces)
  np.save('verts_nathan_.npy', verts)
#============================================================

if __name__=='__main__':
  body_model=np.load('body_nathan_.npy')
  #============================================================
  save_mesh(body_model, 'faces_nathan_.npy', 'verts_nathan_.npy')






















































# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).  or blender (NBendich, Dec. 26, 2018)

'''

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

# TODO:  change to match curr    data's dimensions
ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 20)  # b = 10
ax.set_zlim(0, 32)  # c = 16

plt.tight_layout()
plt.show()
plt.close()
'''

