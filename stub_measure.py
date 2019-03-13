#=====================================================================
def mesh_cross_sec(verts, faces, height, which_ax="z"):
  '''
    Takes a cross section of an .obj mesh
    The plane that creates the cross section is at "height" along "which_ax"

    verts are the pts of the mesh
    faces are the triangles
  '''
  NUM_DIMS=1; XYZ=3
  assert verts.shape[NUM_DIMS]==XYZ
  X=0; Y=1; Z=2
  if    which_ax.lower()=='x': ax=X
  elif  which_ax.lower()=='y': ax=Y
  elif  which_ax.lower()=='z': ax=Z

  adjs=adjacents(verts, faces)

  # Sort low to high (small to big).  Makes the operation faster
  sort_indices=np.argsort(verts[:,ax])
  sorted_verts=verts[sort_indices]
  inverted=invert_indices(sort_indices)
  #np.greater(sorted_verts[:,ax],height) # NOTE: there must be a way to do this directly without iterating cumbersome-ly

  # Triangle walk straddling the height
  for i,loc in enumerate(sorted_verts[:,ax]):
    if loc > height:
      vert_1st_under_idx= sorted_verts[:,ax][i-1]
      vert_1st_over_idx = sorted_verts[:,ax][i]
      orig_idx_over     = inverted[i]
      orig_idx_under    = inverted[i-1]
      print("orig_idx_over  : ",orig_idx_over)
      print("orig_idx_under : ",orig_idx_under)
      these_faces       = adjs[orig_idx_over]
      for face in these_faces:
        for vert in face:
          if vert[ax] < height:
            print("vert:        : ",vert)
            print("orig_idx_over: ",orig_idx_over)
            #vert
            #triang_walk(vert[],vert[ax])
      break
    # [2,2,2], [1,1,1], [0,0,0],
    # 6888: [ array([6887, 6886, 6888]),
    #         array([6888, 6886, 6889]),
    #         array([5238, 6887, 6888]),
    #         array([6889, 3953, 6888]),
    #         array([5238, 6888, 5239]),
    #         array([5239, 6888, 3953])],

  # Find bounds
  top = midpt_h + ( window/2.); bot = midpt_h - ( window/2.)
  found_bot=False; found_top=False
  for i,loc in enumerate(sorted_verts[:,ax]):
    if (not found_bot) and (loc > bot):   # should happen first
      found_bot=True; bot_idx=i
    elif (not found_top) and (loc > top): # before this guy
      found_top=True; top_idx=i
  targ_verts=sorted_verts[bot_idx:top_idx]

  # Only take xy values
  cross_sec=targ_verts[:,:Z]
  plt.title("Vertices in SMPL mesh at loc {0} along the \n{1} axis\nwith {2} points".format(midpt_h, which_ax, len(targ_verts)))
  plt.scatter(cross_sec[:,0],cross_sec[:,1]); plt.show()
  return cross_sec
#=====================================================================
def invert_indices(sort_indices):
  backwards=np.zeros(sort_indices.shape).astype("int64")
  for i,e in enumerate(sort_indices):
    backwards[e]=i
  return backwards
#=====================================================================
def adjacents(verts, faces):
  # .obj mesh
  adjs={i:[] for i in range(len(verts))}
  for face_idx,face in enumerate(faces): # 1,  v1 v2 v3
    v0=face[0]; v1=face[1]; v2=face[2]
    adjs[v0].append(face)
    adjs[v1].append(face)
    adjs[v2].append(face)
  return adjs
#=====================================================================


