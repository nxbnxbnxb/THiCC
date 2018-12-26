import numpy as np
from skimage import measure
from skimage.draw import ellipsoid

# show model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

body_model=np.load('body_nathan_.npy')
#============================================================
verts, faces, normals, values = measure.marching_cubes_lewiner(body_model, 0)
#============================================================

np.save('faces_nathan_.npy', faces)
np.save('verts_nathan_.npy', verts)






















































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

