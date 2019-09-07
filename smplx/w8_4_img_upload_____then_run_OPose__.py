'''
  TODO: 
    1.  Automatically   empty gsutil bucket "gs://vrdr_bucket_1/"   and     
      /home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket   .      Currently I have to do it manually, but once we get 1,000+ customers, image file overflow will start to be a serious problem.

'''



import time
import datetime
import subprocess; sp = subprocess
# "sp" vs. "subprocess" is  open-ended;   you don't HAVE to use the "sp" shortcut, but it's there for us.
import difflib
from collections import OrderedDict as OrD
import os











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




# TODO:   def get_bucket_diff(bucket_name):
  # see "get_customer_imgfname()" :    they should be ***VERRY*** similar.



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

  encoding='utf-8'

  local_bucket_fname='/openpose/data/vrdr_customer_imgs________gsutil_bucket/images/'
  # The value of "OP_bin_loc" is locally determined  (if you're on a different machine, OP_bin_loc='/the/path/to/openpose.bin/on/your/machine/openpose.bin' )
  #   For the first iteration writing this, I'm on docker on the GCloud Compute Engine   Virtual Machine called "openpose-ubuntu16-1."  (Today.Date == September  7 2019.)
  cmd=['ls', '-tr', local_bucket_fname]

  # Initial "ls -ltrAh /path/to/imgdir/"      (like a "do-while" loop) :
  bucket_proc = sp.Popen(cmd, stdout=sp.PIPE) 
  # Description of "sp.Popen():"      https://docs.python.org/3/library/subprocess.html   and   https://stackoverflow.com/questions/89228/calling-an-external-command-in-python/92395#92395 .
  prev_bucket_contents, err = bucket_proc.communicate()
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
      print("Been waiting "+str(secs_waited)+" seconds for an image file.") 
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
        img_path=local_bucket_fname+img_fname
        if not   ( img_fname.endswith('png') or img_fname.endswith('jpeg') or img_fname.endswith('jpg') ):
          #prev_bucket_contents=bucket_contents   
          # I was gonna do `prev_bucket_contents=bucket_contents`, but it actually doesn't matter.  
          # We leave "why doesn't it matter?" as an exercise to the reader.  -nxb, Sun Sep  1 23:44:15 EDT 2019
          continue
        #========= image of format .png, .jpg, or .jpeg :=========
        else:
          return img_path # <------ ======================== NOTE: THIS IS THE PATH OUT OF THE LOOP. NOTE: ========================
      #============================================================== end "if new_file_uploaded:" ===============================================================
    prev_bucket_contents=bucket_contents
  #===============================================================end "while True:"===============================================================

  raise Exception("I don't know why you would ever hit this line of code.    Something is wrong, and it's probably Nathan's fault, not yours.")
#=========================================== end function definition of "wait_4_customer_img():"===========================================

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
    MUST BE RUN FROM ROOT /openpose directory.

    Rapid prototyping:   M.V.P.
  '''
  customer_img_path = get_customer_imgfname()
  print(customer_img_path)

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
#===================================================================================================




































































































# old debugging shit:
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

'''
