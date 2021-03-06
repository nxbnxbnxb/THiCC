import numpy as np
from scipy.spatial import Delaunay

#points = np.random.rand(30, 2) # 30 points in 2-d
#points = np.random.rand( 4, 2) #  4 points in 2-d
points = np.array([ [- 1.   , - 1   ,],
                    [  2.   ,   0.  ,],
                    [  0.   ,   2.  ,],
                    [  0.5  ,   0.5 ,]   ]) # concave 4-gon
tri = Delaunay(points)
# NOTE:  triangles are in:  tri.vertices!

# Make a list of line segments: 
# edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
#                 ((x1_2, y1_2), (x2_2, y2_2)),
#                 ... ]
edge_points = []
edges = set()

def add_edge(i, j):
    """Add a line between the i-th and j-th points, if not in the list already"""
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add( (i, j) )
    edge_points.append(points[ [i, j] ])

# loop over triangles: 
# ia, ib, ic = indices of corner points of the triangle
for ia, ib, ic in tri.vertices:
    add_edge(ia, ib)
    add_edge(ib, ic)
    add_edge(ic, ia)

print("edges: \n",edges)
print()
print("edge_points: \n",edge_points)
print()

# plot it: the LineCollection is just a (maybe) faster way to plot lots of
# lines at once
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

lines = LineCollection(edge_points)
plt.figure()
plt.title('Delaunay triangulation')
plt.gca().add_collection(lines)
plt.plot(points[:,0], points[:,1], 'o', hold=1)
plt.xlim(-1, 2)
plt.ylim(-1, 2)
lines = LineCollection(edge_points)
plt.show()

"""
# -- the same stuff for the convex hull

edges = set()
edge_points = []

for ia, ib in tri.convex_hull:
    add_edge(ia, ib)

lines = LineCollection(edge_points)
plt.figure()
plt.title('Convex hull')
plt.gca().add_collection(lines)
plt.plot(points[:,0], points[:,1], 'o', hold=1)
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.show()
"""
