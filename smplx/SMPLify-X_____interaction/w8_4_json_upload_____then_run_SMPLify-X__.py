"""
  NOTE: This should be run from the root directory :
      "/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/"
"""



'''
  TODO: 
    1.  Automatically   empty gsutil bucket "gs://vrdr_bucket_1/"   and     
      /home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket   .      Currently I have to do it manually, but once we get 1,000+ customers, image file overflow will start to be a serious problem.

'''


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
def mkdir():
  '''
    Makes a VALID temporary directory on the filesystem and returns a string with a that directory's name.
    Initial intention behind this function:
      Makes a temp directory on Gcloud Compute Engine (GCE)   ***exclusively*** for the purposes of briefly making a directory    to avoid name collisions.
  '''
  dir_name='tmp'
  made_dir=False # 0
  i=0
  # This ***whole*** loop is JUST to deal with the case where the user already has *ALL* the following directories in the current working directory: ("tmp/", "tmp1/", "tmp2/", "tmp3/", "tmp4/", ...)
  #================================================================================
  while not made_dir:
    made_dir_failed = sp.call(['mkdir', dir_name+str(i)])
    '''
      The return value 
      (0 for success,  or 
       1 for "this directory with that name already exists")
      is how "subprocess.call()" tells you the shell process succeeded.
    '''
    i+=1
    made_dir = not made_dir_failed 
  #================================================================================

  return dir_name+ str(i-1) +  '/'
  # `dir_name+str(i-1)`    is the correct name of the actual directory name  because  "i" got incremented 1 too many times during the "while not made_dir:" loop
#====================================== end function definition of "mkdir():"======================================













#===================================================================================================
def bucket_diff(
  local_rsync_dir_path='/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/keypoints', 
  file_types=['json']):
  '''
    Returns the filename of the most recently synced (uploaded) "gsutil rsync"ed file in 'local_rsync_dir_path'.
    As of September 8, 2019,   Compares the **CURRENT** contents of the gcloud bucket to the previous contents of the 

    file_types is a python [sequence](https://docs.python.org/3/library/stdtypes.html).  It may include only a single element, ie. : file_types=['json']    or multiple, ie. :  file_types=('png', 'jpeg', 'jpg')

  1.
  # TODO: error handling.
  '''



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
def get_customer_OP_JSON_fname():
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
    local_rsync_dir_path='/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/keypoints/',
    file_types=['json']
  ) # the json is already where it should be for SMPLify-X.
#=========================================== end function definition of "get_customer_OP_JSON_fname():"===========================================


















#==================================================================================================================
def get_customer_imgfname():
  '''
    Returns the filename of the picture of the customer     that was uploaded via the simple webpage.

  1.
  # TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  # We wanna try to avoid making a high-detail large-file-size mesh be sent too many times.
  # Can we do a function "upload_img_directly_from_gs_util_storage_bucket()" ?
  # Or do we have to copy the image locally BEFORE we run SMPLify-X on it?   I could ***probably*** do everything remotely  (on gcloud storage/bucket)  if I installed the GCloud API, but we're very much in "rapid-prototyping-startup-MVP mode" right now.   ***Besides***, the best way we can make the end product ***fastest*** is by using the ***profiler*** at the end, rather than trying to predict which part of the process is going to bottleneck it for the customer.

  2.
  # TODO NOTE:  we have to sync "gs://vrdr_bucket_1/" to '/home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket' MORE OFTEN than every minute (that's what crontab is currently doing)

  3.
  # TODO: error handling.
  '''
  return bucket_diff(
    local_rsync_dir_path='/openpose/data/vrdr_customer_imgs________gsutil_bucket/images/',
    file_types=['png', 'jpg', 'jpeg']
  )
#=========================================== end function definition of "get_customer_imgfname():"===========================================


#==================================================================================================================
def SMPLifyX(
  main              = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/smplifyx/main.py',
  conf_file         = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/cfg_files/fit_smplx.yaml',
  cust_data_dir     = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data',
  output_dir        = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/SMPLify-X___results',
  smplx_models_dir  = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/from_MPII_downloads/smplx/models',
  vposer_dir        = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/from_MPII_downloads/vposer_v1_0',
  pkl_filepath      = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/smplx_parts_segm.pkl'
  ):
#==================================================================================================================
  '''
    Run the "smplify-x" demo, just like in "run.sh" :

      Contents of "run.sh" :

    OUTPUT_FOLDER=/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/SMPLify-X___results
    DATA_FOLDER=/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/homogenus/samples
    MODEL_FOLDER=/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/from_MPII_downloads/smplx/models
    VPOSER_FOLDER=/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/from_MPII_downloads/vposer_v1_0

    python3 smplifyx/main.py --config cfg_files/fit_smplx.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder $OUTPUT_FOLDER \
        --visualize="False" \
        --model_folder $MODEL_FOLDER \
        --vposer_ckpt $VPOSER_FOLDER \
        --part_segm_fn smplx_parts_segm.pkl # &&\
  '''

  SMPLifyX_cmd=[
    'python3',          main,
    '--config',         conf_file,
    '--data_folder',    cust_data_dir,
    '--output_folder',  output_dir,
    '--visualize="False"',
    '--model_folder',   smplx_models_dir,
    '--vposer_ckpt',    vposer_dir,
    '--part_segm_fn',   pkl_filepath
  ]
  SMPLify_X_success = not sp.call(SMPLifyX_cmd)
  funcname  = sys._getframe().f_code.co_name
  if not SMPLify_X_success:
    raise Exception("There was an error when attempting to run   '" + main + "' \n in function " + funcname + ".")
  return True     # "True" means the function was successful.  Otherwise we'll throw ("raise")  the Exception.

#==================================================================================================================
#================================== end function definition of " SMPLifyX(): " ====================================
#==================================================================================================================








#==================================================================================================================
def openpose():
  '''
===================================================================================================
    Call the openpose binary 'openpose.bin' on images.
===================================================================================================
    Returns a dictionary of strings   pointing to where the 
      1.  openpose.json and   
      2.  images rendered with keypoints
      3.
      4.
      5.
    are.
===================================================================================================
  '''
  # The value of "OP_bin_loc" is locally determined  (if you're on a different machine, OP_bin_loc='/the/path/to/openpose.bin/on/your/machine/openpose.bin' )
  #   For the first iteration writing this, I'm on docker on the GCloud Compute Engine   Virtual Machine called "openpose-ubuntu16-1."  (Today.Date == September  7 2019.)
  """
  run_OP_cmd= '/openpose/OP_demo.sh'
  success = not sp.call([
    run_OP_cmd
  ])
  if not success:
    raise Exception("There was an error when attempting to run   '"+run_OP_cmd+"' .")
  return {
    "for debugging, see ": run_OP_cmd
  }
  """
  now=datetime.datetime.now()

  OP_bin_path   = '/openpose/build/examples/openpose/openpose.bin'
  in_img_path   = '/openpose/data/images/'
  out_img_path  = '/openpose/output/OP_keypoints_imgs_{0:%Y_%B_%d___%H:%M}/'.format(now)
  success= not sp.call(['mkdir', out_img_path])
  if not success:
    raise Exception("There was an error when attempting to create directory '"+out_img_path+"' .")
  out_json_path = '/openpose/output/keypoints_json_{0:%Y_%B_%d___%H:%M}/'.format(now)
  success= not sp.call(['mkdir', out_json_path])
  if not success:
    raise Exception("There was an error when attempting to create directory '"+out_json_path+"' .")
 
  success = not sp.call([
    OP_bin_path, 
    '--image_dir', in_img_path,
    '--face', '--hand',
    '--write_json', out_json_path,
    '--write_images', out_img_path,
    '--write_images_format', 'jpg',
    '--display', '0',
    '--render_pose', '1'
  ])
  if not success:
    raise Exception("There was an error while calling   openpose.bin on the image.  We THOUGHT openpose.bin was located at "+OP_bin_path+" on this machine.")
  return {
    "out_img_dir" : out_img_path,
    "out_json_dir": out_json_path
  }
#==================================================================================================================



















#==================================================================================================================
def cp_2_img_dir(src_path):
  # copy to image dir
  img_dir='/openpose/data/images'
  success= not sp.call(['cp', src_path, img_dir])
  if not success:
    raise Exception("There was an error when copying file "+src_path+" to destination directory '"+img_dir+"' .")
  data_dir=img_dir[:img_dir.rfind('/')]
  data_dir+='/'
  return data_dir
#==================================================================================================================



#===================================================================================================
if __name__=="__main__":
  '''
    Rapid prototyping:   M.V.P.
  '''
  #===================================================================================================
  # Find uploaded openpose_keypoints.json from the    process running on docker container on openpose-ubuntu16-1     (what's the terminology, technically?  "docker image?",  "docker instance?",  "docker container?", )
  #===================================================================================================
  customer_json_path=get_customer_OP_JSON_fname()
  print("="*99)
  print(" "*22+"Customer JSON Path : ")
  print(" "*15+customer_json_path)
  print("="*99)

  #===================================================================================================
  # Call SMPLify-X on the customer image and cust_img_openpose_keypoints.json to   get a 3-D mesh of their body:
  #===================================================================================================
  cust_data_dir     = '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data'
  SMPLify_X____output_dir= '/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/SMPLify-X___results'
  SMPLifyX_success = SMPLifyX(
    cust_data_dir = cust_data_dir,
    output_dir    = SMPLify_X____output_dir)
  if not SMPLifyX_success:
    raise Exception("There was an error when trying to run SMPLify-X on  "+cust_data_dir  + "\n with output directory " + str(SMPLify_X____output_dir))

  #===================================================================================================
  # Get the filename of the picture that the customer uploaded of themselves   so we can locate the .obj mesh file to send to Nathan (nxb)'s laptop so blender can render a png so we can show the customer a mesh of their body.
  #===================================================================================================
  img_upload_dir='/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/images/'
  # In the following lines of code,    I'm relying on the fact that the  local image and json directories (that are rsynced with the gcloud storage buckets) are always ***EMPTY***   before the customer uploads the picture of themselves.
  ls_new_upload_cmd=['ls', img_upload_dir]
  cust_img_proc=sp.Popen(ls_new_upload_cmd, stdout=sp.PIPE)
  cust_img_fname_encoded, ls_img_err  = cust_img_proc.communicate()
  if  ls_img_err:
    raise Exception("There was an error when trying to run command " + str(ls_new_upload_cmd))
  encoding='utf-8'
  cust_img_fname= cust_img_fname_encoded.decode(encoding)
  cust_img_fname_no_filetype=cust_img_fname[:cust_img_fname.rfind('.')  ]         # ie.  "dog" instead of "dog.jpg":

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

















  #cp_json_from_rsync_bucket_to_SMPLifyx_directory___cmd = ['cp', customer_json_path, '','','']              # NOTE:  should we also copy the   SMPL-X-mesh-overlaid-on-original-customer-image.png?

  #success = not sp.call(cp_json_from_rsync_bucket_to_SMPLifyx_directory___cmd)
  #if not success:
    #cmd=''
    #for word in cp_json_from_rsync_bucket_to_SMPLifyx_directory___cmd:
      #cmd+=' '+word
    #raise Exception("'cp' command within python3   'subprocess.call' failed.  \n  Was trying to execute the command '"+str(cmd)  +"'")
  

  #=============================================================================================
  #==================================== BEGIN COMMENT-OUT. {====================================
  #=============================================================================================
  '''
  local_customer_img_path  = cp_2_img_dir(customer_img_path)
  OP_results  = openpose()
  dated_json_dir=OP_results['out_json_dir']
  #./build/examples/openpose/openpose.bin --image_dir data/imgs/ --face --hand --write_json ./output --write_images ./output --write_images_format jpg --display 0 --render_pose 1
  json_dir='/openpose/output/keypoints_json/'
  success = not sp.call(['cp', dated_json_dir+'*', json_dir])
  if not success:
    raise Exception("There was an error when attempting to copy json from  directory '"+dated_json_dir+"' to '"+json_dir+"'.")
  #======================================================================================================================================================================================================
  # TODO.
  #smplifyx(customer_img_path)  # This can't be opening any GUIs.  TODO TODO TODO   NOTE  TODO TODO TODO.
  #======================================================================================================================================================================================================
  '''
#===================================================================================================




































































































#===================================================================================================
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
# old debugging shit:
#===================================================================================================
#===================================================================================================
#===================================================================================================
'''
  In "wait_4_customer_img()" :
      # This was `img_fname` : -rw-r--r--1cat_macys_vrcat_macys_vr0Sep202:35sigmoid.png_.gstmp     (solved this bug with the "if not ..." check right below this line)
      # NOTE:  here I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  
      #        ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
      # NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )
    #======================================================================================================================================================================================================
      # Basically, the next line of code is "img_fname  =  curr - prev"  (the next line of code is "str1_minus_2(curr, prev)")
      # this was the "diff"  :   -rw-r--r--1cat_macys_vrcat_macys_vr0Sep202:35sigmoid.png_.gstmp
      # I put the sync in editcron. (/etc/crontab)        ("sync" more technically means    `gsutil rsync`.   For some reason this (["gsutil", "ls", bucket_name])  was being weird to call within python3 )
      #bucket_name="gs://vrdr_bucket_1/"
        #["gsutil", "ls", bucket_name]   # this (["gsutil", "ls", bucket_name]) was giving us trouble for some reason.
      # OLD comment.  Only left here briefly so I remember why I changed the code to its current state. -nxb, Sun Sep  1 21:38:26 EDT 2019    
      #   The old comments:   NOTE:  here, I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
      #                       NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )
      # I'm probably making some assumptions to the effect of    "we didn't just change directories somehow, and './' from os.getcwd() is the same dir as the one where I locally "gsutil copied" the image to in the first place.   -nxb, Sun Sep  1 20:23:02 EDT 2019

      #failed=sp.call(['rm', '-rf', tmp_dir_name])   # NOTE:   can also do "succeeded = not sp.call(...)"


        ===================================================================================================
        ===================================================================================================
        ===================================================================================================
        ===================================================================================================
           This is the value of the variable 'img_path' within the function 'get_customer_imgfname():' 
        /openpose/data/vrdr_customer_imgs________gsutil_bucket/imagesn8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg

        NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE 
        ===================================================================================================
        /openpose/data/vrdr_customer_imgs________gsutil_bucket/imagesn8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg

        cp: cannot stat '/openpose/data/vrdr_customer_imgs________gsutil_bucket/imagesn8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg'$'\n': No such file or directory
        Traceback (most recent call last):
          File "w8_4_img_upload_____then_run_OPose__.py", line 280, in <module>
            cp_success  = cp_2_img_dir(customer_img_path)
          File "w8_4_img_upload_____then_run_OPose__.py", line 264, in cp_2_img_dir
            raise Exception("There was an error when copying file "+src_path+" to destination directory '"+data_dir+"' .")
        UnboundLocalError: local variable 'data_dir' referenced before assignment
        ===================================================================================================
        ===================================================================================================
        ===================================================================================================
             This is the value of the variable 'img_path' within the function 'get_customer_imgfname():' 
          /openpose/data/vrdr_customer_imgs________gsutil_bucket/images/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg

          /openpose/data/vrdr_customer_imgs________gsutil_bucket/images/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg

          cp: cannot stat '/openpose/data/vrdr_customer_imgs________gsutil_bucket/images/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg'$'\n': No such file or directory
          Traceback (most recent call last):
            File "w8_4_img_upload_____then_run_OPose__.py", line 280, in <module>
              cp_success  = cp_2_img_dir(customer_img_path)
            File "w8_4_img_upload_____then_run_OPose__.py", line 264, in cp_2_img_dir
              raise Exception("There was an error when copying file "+src_path+" to destination directory '"+img_dir+"' .")
          Exception: There was an error when copying file /openpose/data/vrdr_customer_imgs________gsutil_bucket/images/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg
           to destination directory '/openpose/data/images' .
        ===================================================================================================
        ===================================================================================================
        ===================================================================================================
          Starting OpenPose demo...
          Configuring OpenPose...
          Starting thread(s)...
          Auto-detecting all available GPUs... Detected 1 GPU(s), using 1 of them starting at GPU 0.

          Error:
          Prototxt file not found: models/face/pose_deploy.prototxt.
          Possible causes:
            1. Not downloading the OpenPose trained models.
            2. Not running OpenPose from the same directory where the `model` folder is located.
            3. Using paths with spaces.

          Coming from:
          - /openpose/src/openpose/net/netCaffe.cpp:ImplNetCaffe():53
          - /openpose/src/openpose/net/netCaffe.cpp:ImplNetCaffe():89
          Building synchronization state...
          - /openpose/include/openpose/wrapper/wrapperAuxiliary.hpp:configureThreadManager():1131
          - /openpose/include/openpose/wrapper/wrapper.hpp:exec():424
          Starting synchronization...
          Traceback (most recent call last):
            File "w8_4_img_upload_____then_run_OPose__.py", line 276, in <module>
              OP_results  = openpose(local_customer_img_path)
            File "w8_4_img_upload_____then_run_OPose__.py", line 228, in openpose
              raise Exception("There was an error while calling   openpose.bin on the image.  We THOUGHT openpose.bin was located at "+OP_bin_path+" on this machine.")
          Exception: There was an error while calling   openpose.bin on the image.  We THOUGHT openpose.bin was located at /openpose/build/examples/openpose/openpose.bin on this machine.



      Copying gs://vrdr_bucket_1/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg...
      / [1 files][  1.2 MiB/  1.2 MiB]                                                
      Operation completed over 1 objects/1.2 MiB.                                      

















  TODO:     Put all the directories as the    input parameters to the function "SMPLifyX(..., dir2='...', dir3='...', dir4='/path/to/dir4/')












       File "w8_4_json_upload_____then_run_SMPLify-X__.py", line 302
           '--visualize="False",

===================================================================================================

	Copying gs://openpose_json/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE_____keypoints.json...debug2: channel 0: window 999323 sent adjust 49253

/ [1 files][  3.5 KiB/  3.5 KiB]                                                
Operation completed over 1 objects/3.5 KiB.                                      
===================================================================================================
                      Customer JSON Path : 
               /home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/customer_data/keypoints/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE_____keypoints.json
===================================================================================================
Traceback (most recent call last):
  File "w8_4_json_upload_____then_run_SMPLify-X__.py", line 436, in <module>
    output_dir    = SMPLify_X____output_dir)
  File "w8_4_json_upload_____then_run_SMPLify-X__.py", line 307, in SMPLifyX
    SMPLify_X_success = not sp.call(SMPLifyX_cmd)
  File "/opt/anaconda3/envs/SMPLX/lib/python3.7/subprocess.py", line 323, in call
    with Popen(*popenargs, **kwargs) as p:
  File "/opt/anaconda3/envs/SMPLX/lib/python3.7/subprocess.py", line 775, in __init__
    restore_signals, start_new_session)
  File "/opt/anaconda3/envs/SMPLX/lib/python3.7/subprocess.py", line 1453, in _execute_child
    restore_signals, start_new_session, preexec_fn)
TypeError: expected str, bytes or os.PathLike object, not tuple




'''
#===================================================================================================
#===================================================================================================
#===================================================================================================
