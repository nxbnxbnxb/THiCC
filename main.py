import model
import m_cubes
import os
 

#================================================================
def save_mesh__from_video(vid_fname,final_mesh_fname,root_img_dir,root_mask_dir,mesh_filetype='blender'):
  '''
  '''
  # TODO: make this more end-to-end rather than saving and loading a million times
  #os.system("./cut_up_video.sh") # NOTE: I did this so I can use cv2; for some reason conda isn't letting me; so this shell script changes into a virtualenv where cv2 is working
  # TODO: get return code from os.system() and do error catching
  #model.save_masks_from_imgs(root_img_dir,root_mask_dir)
  pt_cloud=model.pt_cloud_from_masks(root_mask_dir)
  """
  if mesh_filetype.lower()=='blender':
    faces_fname='faces.py' # TODO: double-check the desired file ending for these files: .py? .txt?
    verts_fname='verts.py' # TODO: put these intermediate filename variables in the function parameter list?  This is doable now b/c we made them variables rather than completely hard-coding them.
    m_cubes.save_mesh(pt_cloud,faces_fname,verts_fname)
    # TODO: finally, make the .blend file after here
    utils.blend_it(final_mesh_fname,faces_fname,verts_fname)  # TODO: don't be cute?  So says Uncle Bob
  elif mesh_filetype.lower()=='unity':
    # do blender steps
    # use web API to convert
    pass # TODO TODO TODO TODO TODO TODO TODO TODO
  """
  return
  # TODO: move this func from m_cubes to somewhere more sensible; we ought to be able to generalize this 
#==== end func def of  save_mesh__from_video(vid_fname,final_mesh_fname,root_img_dir,root_mask_dir,mesh_filetype='blender') ====
def main():
  vid_fname='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/unclothed_outside_delaware_____uniform_background_with_wobbling.mp4'
  out_mesh='nathan.blend'
  root_img_dir='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/'
  root_mask_dir='/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks/'
  mesh_filetype='blender'
  save_mesh__from_video(
    vid_fname,
    out_mesh,
    root_img_dir,
    root_mask_dir,
    mesh_filetype)
  return

if __name__=="__main__":
  main()
  # TODO: incremental testing

