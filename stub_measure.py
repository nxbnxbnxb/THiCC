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

#=====================================================================
def invert_indices(sort_indices):
  backwards=np.zeros(sort_indices.shape).astype("int64")
  for i,e in enumerate(sort_indices):
    backwards[e]=i
  return backwards
#=====================================================================

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


#=====================================================================
def mesh_err():
  calced_chest_circum , _ = mesh_perim_at_height(verts, faces, chest_h, which_ax='z')
  calced_hip_circum   , _ = mesh_perim_at_height(verts, faces, hip_h  , which_ax='z')
  calced_waist_circum , _ = mesh_perim_at_height(verts, faces, waist_h, which_ax='z')

  #crotch ratio: {'height': 255/433 down from the top, 'x_loc': 120/221 from the left to the right}
  CROTCH_LR_RATIO=120/211.
  crotch_depth=31.31119602
  calced_crotch_2_head_circum, crotch = mesh_perim_at_height(verts, faces, crotch_depth, which_ax='x', plot=True)
  print("calced_crotch_2_head_circum: ",calced_crotch_2_head_circum)
  min_height=np.inf
  real_crotch=None
  real_crotch_depth=crotch_depth
  start = crotch_depth-0.1
  end   = crotch_depth+0.1
  print("start = {0}     and end = {1}".format(start, end))
 
  # TODO NOTE This perfect-crotch-search takes too long.  We have to adapt it somehow to find a saddle point in z.   Or to only find the highest min point instead of calculating the whole ConvexHull and perimeter every time.  But before we make it fast, we prob have to check that we actually WANT the crotch so precisely.
  # Note: refactor into separate get_crotch()
  for d in np.linspace(start,end,99):
    calced_crotch_2_head_circum, crotch = mesh_perim_at_height(verts, faces, d, which_ax='x')
    if calced_crotch_2_head_circum < min_height:
      min_height=waist
      real_crotch_depth=d
      real_crotch=crotch
  print("min_height:",min_height)
  print("real_crotch_depth:",real_crotch_depth)
  print("real_crotch:",real_crotch) # NOTE: real_crotch: [27.65771812 12.4297897  31.31119602]
                                    #                    [27.66326531 12.42601213 31.31183592]
  bots_idx=np.argmin(verts[:,2])
  toe=verts[bots_idx]
  print('toe:',toe)
  print('toe.shape:',toe.shape)
  toe_to_head_perim, _ = mesh_perim_at_height(verts, faces, toe[0], which_ax='x', plot=True)
  print('toe_to_head_perim:',toe_to_head_perim)
  #toes=np.min(verts)
  BUMP=2.3428187919463   #1.8715
  crotch_depth-=BUMP # 28.1285.   Ought to be somewhere like 27.6571812080537
  #crotch_depth=x_max-crotch_depth
  pn(9);pe();pe();pe();pe();pr(" "*24+"about to calculate crotch")
  pr(" "*26+"crotch_depth:",crotch_depth);pe();pe();pe();pe();pn(9)
  calced_crotch_2_head_circum, _ = mesh_perim_at_height(verts, faces, crotch_depth, which_ax='x')
  pr("calced_crotch_2_head_circum:", calced_crotch_2_head_circum) # real is ~    calcul8d is ~102.02471693093469 inches
  calced_crotch_2_head_circum, _  = mesh_perim_at_height(verts, faces, real_crotch_depth, which_ax='x', plot=True)
  # ToDO: use real_crotch_depth to calculate inseam.  Is this good enough?  (Better/ worse than finding the toes & calcul8ing the inseam by dist_btwn(crotch, toe)
  # TOdO: generalize the crotch-finding calculation, hook up the openpose shit (hip, chest, waist, etc.  heights)     end-to-end
 

  #
  pr("calced_crotch_2_head_circum:" , calced_crotch_2_head_circum) # real is ~    calcul8d is ~102.02471693093469 inches.    Lower is 101.51852706256683
  pr("calced_chest_circum:  " , calced_chest_circum)   # real is ~34 inches (ConvexHull)
  pr("calced_hip_circum  :  " , calced_hip_circum)     # real is ~32 inches (ConvexHull)
  pr("calced_waist_circum:  " , calced_waist_circum)   # real is ~30 inches (ConvexHull)
#=====================================================================






































