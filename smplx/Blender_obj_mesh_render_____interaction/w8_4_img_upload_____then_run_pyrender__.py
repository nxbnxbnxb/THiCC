# NOTE:  if you need old functions from the other SMPLify-X and OpenPose, just **RECOPY** them.

'''
  TODO: 
    1.  Automatically   empty gsutil bucket "gs://vrdr_bucket_1/"   and     
      /home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket   .      Currently I have to do it manually, but once we get 1,000+ customers, image file overflow will start to be a serious problem.

'''


# TODO:   remove unnecessary `import` statements.
# All these `import` statements work in    the conda environment "SMPLX" (python3).
import time
import datetime
import subprocess; sp = subprocess
# "sp" vs. "subprocess" is  open-ended;   you don't HAVE to use the "sp" shortcut, but it's there for us.
import difflib
from collections import OrderedDict as OrD
import os
import sys











#===================================================================================================
def str1_minus_2(s1, s2):
  '''
    TODO: improve the efficiency.  We end up making a list and all other sorts of nonsense.
      Luckily, in the use case I want this function for ***right now***, the diff should be a short filename.
    Basically does "s1 - s2."


    NOTE: fails on unusual characters (non-English letters or numbers)



    Example:
      "a" == str_diff("Malaga", "Malag")
  '''
  ascii_ints_diffs = [li for li in difflib.ndiff(s1, s2) if li[0] != ' ']  # NOTE:  If this line of code confuses you, please see https://stackoverflow.com/questions/17904097/python-difference-between-two-strings, answer 2 "ndiff"
  diffs= OrD()
  for i,diff in enumerate(ascii_ints_diffs):
    diff_sp = diff.split(' ')
    # if s1 has this character, but s2 ***DOESN'T***.
    s1_has_this_char____but_s2_doesnt=  diff_sp[0]=='-'
    if s1_has_this_char____but_s2_doesnt:
      diffs[i] = diff_sp[1]
      #diffs[i] = chr(diff_sp[1]) # This line is the correct version **iff**  `s1-s2` contains odd ASCII characters like vowels with umlauts above them (http://www.asciitable.com/)
  out=''
  for i,diff in diffs.items():
    out+=diff
  return out
#===================================================================================================



















#===================================================================================================
def bucket_diff(
  local_rsync_dir_path='/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/keypoints', 
  file_types=['json']):
  #===================================================================================================
  '''
    Returns the filename of the most recently synced (uploaded) "gsutil rsync"ed file in 'local_rsync_dir_path'.
    As of September 8, 2019,   Compares the **CURRENT** contents of the gcloud bucket to the previous contents of the 

    file_types is a python [sequence](https://docs.python.org/3/library/stdtypes.html).  It may include only a single element, ie. : file_types=['json']    or multiple, ie. :  file_types=('png', 'jpeg', 'jpg')

  1.
  # TODO: error handling.
  '''
  #===================================================================================================


  cmd=['ls', '-tr', local_rsync_dir_path]

  # 
  # Initial "ls -ltrAh /path/to/imgdir/"      (like a "do-while" loop) :
  bucket_proc = sp.Popen(cmd, stdout=sp.PIPE) 
  # Description of "sp.Popen():"      https://docs.python.org/3/library/subprocess.html   and   https://stackoverflow.com/questions/89228/calling-an-external-command-in-python/92395#92395 .
  prev_bucket_contents, err = bucket_proc.communicate()
  encoding='utf-8'
  prev_bucket_contents= prev_bucket_contents.decode(encoding)
  # To understand "bytes_obj.decode(...)", see   https://stackoverflow.com/questions/606191/convert-bytes-to-a-string 
  assert not err
  # TODO: error handling.

  secs_waited=0
  # NOTE:  we break out of this "while True:" loop with a very particular `return` condition
  #===================================================================================================
  while True:
    time.sleep(1)
    if secs_waited%30==0:
      print("Been waiting "+str(secs_waited)+" seconds for a"+file_types[0]+ " file.") 
    secs_waited+=1

    bucket_proc = sp.Popen(cmd, stdout=sp.PIPE)
    bucket_contents, err = bucket_proc.communicate()
    bucket_contents=bucket_contents.decode(encoding)

    assert not err
    # TODO: error handling.
    if err:
      pass

    else: #(if not err):
      new_file_uploaded  = not (bucket_contents == prev_bucket_contents)
      #==============================================================================
      if new_file_uploaded:
        # Basically, the next line of code is "img_fname  =  curr - prev"  (the next line of code is "str1_minus_2(curr, prev)")
        img_fname=str1_minus_2(bucket_contents, prev_bucket_contents)
        img_fname=img_fname[:-1]      # <===== This line `img_fname=img_fname[:-1]` cuts out the newline character.
        img_path=local_rsync_dir_path+img_fname

        # check file ending
        right_filetype=False
        for ft in file_types:
          if img_fname.endswith(ft):
            right_filetype=True

        if right_filetype:
          return img_path # <------ ======================== NOTE: THIS IS THE PATH OUT OF THE LOOP. NOTE: ========================
        else: #if (not right_filetype):
          #prev_bucket_contents=bucket_contents   
          # I was gonna do `prev_bucket_contents=bucket_contents`, but it actually doesn't matter.  
          # We leave "why doesn't it matter?" as an exercise to the reader.  -nxb, Sun Sep  1 23:44:15 EDT 2019
          continue
      #============================================================== end "if new_file_uploaded:" ===============================================================
    prev_bucket_contents=bucket_contents
  #===============================================================end "while True:"===============================================================

  raise Exception("I don't know why you would ever hit this line of code.    Something is wrong, and it's probably Nathan's fault, not yours.")
#=========================================== end function definition of "bucket_diff(...args...):"===========================================











#==================================================================================================================
def get_customer_obj_fname():
  '''
    Returns the filename of the picture of the customer     that was uploaded via the simple webpage.


    @Precondition:
      I assume the fact that the local image and json directories (that are rsynced with the gcloud storage buckets) are always ***EMPTY***   before the customer uploads the picture of themselves.


  1.
    TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
    We wanna try to avoid making a high-detail large-file-size mesh be sent too many times.
    Can we do a function "upload_img_directly_from_gs_util_storage_bucket()" ?
    Or do we have to copy the image locally BEFORE we run SMPLify-X on it?   I could ***probably*** do everything remotely  (on gcloud storage/bucket)  if I installed the GCloud API, but we're very much in "rapid-prototyping-startup-MVP mode" right now.   ***Besides***, the best way we can make the end product ***fastest*** is by using the ***profiler*** at the end, rather than trying to predict which part of the process is going to bottleneck it for the customer.

  2.
    TODO NOTE:  we have to sync "gs://vrdr_bucket_1/" to '/home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket' MORE OFTEN than every minute (that's what crontab is currently doing)

  3.
    TODO: error handling.
  '''
  return bucket_diff(
    local_rsync_dir_path='/home/n/___vrdr_customer_obj_mesh_____gsutil_bucket/meshes',
    file_types=['obj']
  )
#=========================================== end function definition of "get_customer_OP_JSON_fname():"===========================================


#===================================================================================================
def blender_render(obj_filepath):
#===================================================================================================
  # launch blender
  blender_script_name='./xdotool_blender_obj_import_camera_set.sh'
  blender_success = not sp.call([blender_script_name, obj_filepath])
  if not blender_success:
    raise Exception("Somewhere in the shell script "+blender_script_name+"Blender failed to render the customer's body.") # import .obj file
#===================================================================================================
#========================== end function def of "blender_render()"==================================
#===================================================================================================












#===================================================================================================
if __name__=="__main__":
  '''
    Rapid prototyping:   M.V.P.
  '''
  #===================================================================================================
  # Find uploaded openpose_keypoints.json from the    process running on docker container on openpose-ubuntu16-1     (what's the terminology, technically?  "docker image?",  "docker instance?",  "docker container?", )
  #===================================================================================================
  customer_obj_path = get_customer_obj_fname()
  print("="*99)
  print(" "*22+"Customer .obj Path : ")
  print(" "*15+customer_obj_path)
  print("="*99)

  #===================================================================================================
  # Call "blender_render()" on the customer_mesh.obj to   get a 2-D image of their body's 3-D mesh:
  #===================================================================================================
  blender_success   = blender_render(customer_obj_path)
  if not blender_success:
    raise Exception("There was an error when trying to run blender on  "+cust_data_dir  + "\n with output directory " + str(SMPLify_X____output_dir))

  #===================================================================================================
  # Get the filename of the picture that the customer uploaded of themselves   so we can locate the .obj mesh file to send to Nathan (nxb)'s laptop so blender can render a png so we can show the customer a mesh of their body.
  #===================================================================================================
  img_upload_dir    = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/images/'
  # In the following lines of code,    I'm relying on the fact that the  local image and json directories (that are rsynced with the gcloud storage buckets) are always ***EMPTY***   before the customer uploads the picture of themselves.
  ls_new_upload_cmd = ['ls', img_upload_dir]
  cust_img_proc     = sp.Popen(ls_new_upload_cmd, stdout=sp.PIPE)
  cust_img_fname_encoded, ls_img_err  = cust_img_proc.communicate()
  if  ls_img_err:
    raise Exception("There was an error when trying to run command " + str(ls_new_upload_cmd))
  encoding          = 'utf-8'
  cust_img_fname    = cust_img_fname_encoded.decode(encoding)
  cust_img_fname_no_filetype          = cust_img_fname[:cust_img_fname.rfind('.')  ]         # ie.  "dog" instead of "dog.jpg":

  #===================================================================================================
  # The following line of code  assumes there's only 1 person in the customer's uploaded image:
  obj_mesh_file_path=   SMPLify_X____output_dir+  '/meshes/'+ cust_img_fname_no_filetype+ '/000.obj'
  # ie. obj_mesh_file_path=   '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/SMPLify-X___results/000.obj'
  #customer_json_path
  obj_file_bucket_name='gs://obj_meshes/' 
  cp_obj_mesh_to_gcloud_bucket_cmd=['gsutil', 'cp', obj_mesh_file_path, obj_file_bucket_name]
  success = not sp.call(cp_obj_mesh_to_gcloud_bucket_cmd)
  if not success:
    raise Exception("There was an error when trying to copy the customer's .obj mesh file   to a gcloud storage bucket.  \nThe command attempted is "+ str(cp_obj_mesh_to_gcloud_bucket_cmd) )
  #"gsutil copy obj_mesh_file_path obj_file_bucket_name"
  #===================================================================================================
#==================================================================================================================
#======================================= end 'if __name__=="__main__":' ===========================================
#==================================================================================================================



























#===================================================================================================
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
#===================================================================================================
#===================================================================================================
#===================================================================================================
'''

  TODO:     Put all the directories as the    input parameters to the function "SMPLifyX(..., dir2='...', dir3='...', dir4='/path/to/dir4/')

'''
#===================================================================================================
#===================================================================================================
#===================================================================================================
