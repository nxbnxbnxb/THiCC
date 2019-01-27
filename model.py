import  numpy as np
np.seterr(all='raise')
from    copy import deepcopy
from os import listdir as ls
import datetime
import imageio as ii


# our files/modules.  
import seg
from viz      import * # TODO: narrow down so fewer naming conflicts
from d        import debug 
from save     import save
from utils    import pad_all, prepend_0s
from rot8     import rot8
from hollow   import hollow
from m_cubes  import mesh_from_pt_cloud, save_mesh
# TODO: consolidate all of 'em into utils.py?  or a few, maybe called:
#
#   mesh.py
#   mask.py
#   seg.py
#
# alt:
#   utils.py
#
#
#


#   NOTE:  z is "up" ("up" as in, think about a human being "upright"; the head is "up")

#===================================================================================================================================
def pif(s):
    if debug: print(s)
#===================================================================================================================================
class BadInputException(RuntimeError):
    pass
#===================================================================================================================================
class DimensionMismatchException(RuntimeError):
    pass
#===================================================================================================================================
def save_masks_from_imgs(root_img_dir,root_mask_dir,img_file_extension='jpg'):
  import glob
  import os
  if '/' == root_img_dir[-1]:
    root=root_img_dir+'*'
  else:
    root=root_img_dir+'/*'  # '*' means grab all dirs

  list_of_files = glob.glob(root)
  latest_img_dir = max(list_of_files, key=os.path.getctime)+'/' # https://docs.python.org/2/library/os.path.html.  lots of varieties, but getmtime() suits our purposes for now.  Tue Jan 15 10:16:14 EST 2019
  #oldest first
  if '/' == latest_img_dir[-1]:
    cmd=latest_img_dir+'*'
  else:
    cmd=latest_img_dir+'/*' # '*' means grab all dirs
  img_filenames = sorted(glob.glob(cmd), key=os.path.getmtime)
  timestamp=datetime.datetime.now().strftime('%Y_%m_%d____%H:%M_%p') # p for P.M. vs. A.M.
  curr_mask_dir=root_mask_dir+timestamp+"___"
  make="mkdir "+curr_mask_dir
  print("latest_img_dir is:\n{0}".format(latest_img_dir))
  print("about to:  "+make)
  user_reply=input("continue? y/n")
  if user_reply.upper()=='Y':
    os.system(make)
    #print(img_filenames) #00000.jpg, 00001.jpg, etc.
    for i,img_fname in enumerate(img_filenames):
      if img_fname.endswith(img_file_extension):
        segmap= seg.segment_from_local(img_fname)
        if debug:
          pltshow(np.rot90(segmap,k=1))
        if save:
          mask_fname=curr_mask_dir+'/'+prepend_0s(str(i))+'.'+img_file_extension
          print(mask_fname)
          ii.imwrite(
                  mask_fname,
                  np.rot90(segmap,k=1))
    return True  # TODO: standardize the return-values-upon-success for all functions in all code (in C/C++ it's '0')
  else:
    print("halted")
    return False
  # TODO:  clean up all the old image files laying around ('rm' them)
#===== end func def of   save_masks_from_imgs(root_img_dir,root_mask_dir,file_extension='jpg'): ===== }


#===================================================================================================================================
def mask(model, mask, axis='x'):
  '''
    Returns a new 3-D np array
      (the result of masks the 3-d np array "model" all the way through using the mask)
  '''
  model_copy = deepcopy(model)
  if not type(mask ) == type(np.ones((2,2))):
    raise BadInputException("the mask parameter was supposed to be a numpy array")
  if not type(model) == type(np.ones((2,2))):
    raise BadInputException("the model parameter was supposed to be a (3-D) numpy array")
  if not len(model.shape)==3:
    raise BadInputException("the model np array parameter was supposed to have 3 dimensions")
  if not len(mask.shape )==2:
    raise BadInputException("the mask  np array parameter was supposed to have 2 dimensions")
  models_height = model.shape[2]; masks_height=mask.shape[1]; models_width  = model.shape[0]; masks_width =mask.shape[0]
  if not models_width == masks_width and models_height==masks_height:
    raise DimensionMismatchException(" the models width and masks width have to be equal.   So do the heights")

  '''
  mask_3d = np.dstack((mask,mask))  # "model.shape[2]" many times
  '''
  # NOTE: actual code starts HERE:
  # TODO:  vectorize this instead of using a for loop
  #       do vstack or array_copy or just use clever broadcasting
  model_depth=model.shape[1]
  pif('model_depth is {0}'.format(model_depth))
  for i in range(model_depth):
    model_copy[:,i,:] = np.logical_and(model[:,i,:], mask)
  print()
  UINT8_MAX=np.iinfo('uint8').max; MID=int(round(UINT8_MAX/2.))
  model_copy[np.greater(model_copy, 0)] = MID  # TODO: make sure there aren't negative values in model that are meant to be "on" voxels
  return model_copy
#====================================  end func def of mask(model, mask, axis='x'):  ====================================================
def test_human():
  # mask 1
  import seg; mask_front___URL = "http://columbia.edu/~nxb2101/180.0.png"; mask_2d = seg.segment_from_URL(mask_front___URL); max_dim = max(mask_2d.shape); shape=(max_dim, max_dim, max_dim); shape_2d=(max_dim, max_dim); UINT8_MAX=np.iinfo('uint8').max; MID=int(round(UINT8_MAX/2.))
  #mask_1___fname = "/home/u/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/masks___human_front_and_side/180.0.jpg"
  mask_2d   = pad_all(mask_2d, shape_2d)
  model     = np.full(shape, MID).astype('uint8')
  #model     = np.ones(shape).astype('bool')
  model     = mask(model, mask_2d)
  print ("before rot8();   \n\n")
  if debug:
    show_cross_sections(model, axis='y', freq=250)

  angle     = 90.0  # 90.0  # 72.9
  model     = rot8(model, angle)
  print ("right after rot8();   \n\n")
  if debug:
    show_all_cross_sections(model, freq=20)

  # mask 2
  mask_side___URL ="http://columbia.edu/~nxb2101/90.0.png"
  mask_2d = pad_all(seg.segment_from_URL(mask_side___URL), shape_2d)
  model     = mask(model, mask_2d)
  print ("after 2nd masking:    \n\n")
  SKIN=False
  if SKIN:
    skin=hollow(model)
  if save:
    np.save('body_nathan_.npy', model)
    if SKIN:
      np.save('skin_nathan_.npy', skin )
    save_mesh(model, 'faces_nathan_.npy', 'verts_nathan_.npy')
  else: # not save:
    verts,faces=mesh_from_pt_cloud(model)

  if debug:
    show_all_cross_sections(model, freq=5)
    if SKIN:
      show_all_cross_sections(skin , freq=5)
  return model
#====================================   end func def of test_human():   =======================================================

#===================================================================================================================================
if __name__=='__main__':
  test_human()
#===================================================================================================================================















































































