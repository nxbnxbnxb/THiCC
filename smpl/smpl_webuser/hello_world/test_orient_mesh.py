from smpl import *

#===================================================================================================================================
def test_orient():
  gender = 'male'
  betas=get_betas() # cmd line -nxb (as of Wed Mar 27 17:15:52 EDT 2019)
  smpl_params={
    'gender':gender,
    'betas':betas,
    }
  params=smpl(smpl_params)
  params  = blender_render_mesh(params) # comment out when not debugging
  params=orient_mesh(params)
  params  = blender_render_mesh(params) # comment out when not debugging

#===================================================================================================================================
if __name__=="__main__":
  test_orient()
#===================================================================================================================================
