from smpl import *

#===================================================================================================================================
def write_mesh(verts, faces, fname):
  # overwrite
  with open( fname, 'w') as fp:
    for v in verts:
      fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
    for f in faces+1: # Faces are 1-based, not 0-based in obj files
      fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
# end func write_mesh(verts, faces, fname):
#===================================================================================================================================
def parse_obj_file(obj_fname):
  verts=[]
  faces=[]
  with open(obj_fname, 'r') as f:
    for line in f.readlines():
      if line.startswith('v'):
        verts.append(line)
      elif line.startswith('f'):
        faces.append(line)
        # Note: faces start indexing at 1
  verts_arr=np.zeros((len(verts), 3)).astype("float64") # 'vs' means verts
  fs=np.zeros((len(faces), 3)).astype("int64"  ) # 'fs' means faces
  X,Y,Z=1,2,3
  for idx,vert in enumerate(verts):
    vert=vert.split(' ')
    verts_arr[idx]=vert[X],vert[Y],vert[Z]
    #case={}.get(x,DEFAULT_RET_VAL)
    #generating_method='HMR' or 'NNN' or 'VHMR' or ...
  #pr("measures:"); #p(measures) #print("chest_h:", chest_h) # measured, 55.9 inches is ~correct

  # read in faces
  for idx,f in enumerate(faces):
    f=f.split(' ')
    fs[idx]=np.array([int(f[1]),int(f[2]),int(f[3])]).astype("int64")-1
    # see .obj file for why these conventions are like they are.  each face is described as a sequences of 3 vertices:  ie. "f 1 99 4"
    # in .obj files, face indexing starts at 1  (face 1 might be 1 2 3 rather than 0 1 2, referring to vertices 1, 2, and 3)
  faces=fs
  verts=verts_arr
  return verts, faces
#=============== end parse_obj_file(params) =============== 
#===================================================================================================================================
def test_orient():
  # so, for whatever f***ing reason,   blender does not use "xyz" like you would expect from the .obj file's vertices.  Seems like "y" is actually the 3rd column of numbers for some reason.  So what do we really want orient_mesh() to do?  As long as we get the vertices in a consistent format for the eventual BNN, I THINK nothing else matters.
  mesh_params={
    'gender':'male',
    'betas':get_betas()}
  params=smpl(mesh_params)
  mesh_fname=params['mesh_fname']
  print("mesh_fname:",mesh_fname)
  params  = blender_render_mesh(params) # comment out when not debugging
  verts,faces=parse_obj_file(mesh_fname)
  params['mesh']['verts']=verts
  params['mesh']['faces']=faces
  print(params)
  params=orient_mesh(params)
  """
  mesh_fname='male_4400000000.obj'
  verts, faces=parse_obj_file(mesh_fname)
  params={
    'mesh_fname':mesh_fname,
    'mesh': {
      'verts':verts,
      'faces':faces,
    }
  }
  print("vert_info : ",vert_info(verts))
  params  = blender_render_mesh(params) # comment out when not debugging
  params=orient_mesh(params)
  verts=params['mesh']['verts']
  print("vert_info : ",vert_info(verts))
  write_mesh(verts, faces, params['mesh_fname']) # allows updated blender render
  params  = blender_render_mesh(params) # comment out when not debugging
  # ========= demarcate ==============
  gender = 'male'
  betas=get_betas() # cmd line -nxb (as of Wed Mar 27 17:15:52 EDT 2019)
  smpl_params={
    'gender':gender,
    'betas':betas,
    }
  params=smpl(smpl_params)
  """
#============================================ end func test_orient(params) =========================================================

#===================================================================================================================================
if __name__=="__main__":
  test_orient()
#===================================================================================================================================















































































