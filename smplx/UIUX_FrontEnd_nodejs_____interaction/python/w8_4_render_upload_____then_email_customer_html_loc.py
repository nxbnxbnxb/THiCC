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

import smtplib
import ssl


#===================================================================================================
def get_cust_email(filepath='/home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/cust_email.txt'):
  with open(filepath, 'r') as fp:
    return fp.read()
#===================================================================================================

#===================================================================================================
def email_cust_render(
  sender_email    = "vrdr314@gmail.com",
  receiver_email  = "vrdr271@gmail.com",
  full_render_path="/home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/img_server/public/curr_render.png"
    ):
  '''
    Emails the customer an email directing them to their 3-D shopping.
  '''
  # test emails

  port = 465  # For SSL
  smtp_server = "smtp.gmail.com"
  password = '925927789283732691'
  p=full_render_path
  fname=p[p.rfind('/')+1:]
  msg = """\
  Subject: 3-D clothes shopping done!

  To start shopping in 3-D, visit this website:
  http://35.221.45.20:8082/{} """.format(fname)

  context = ssl.create_default_context()
  with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
      server.login(sender_email, password)
      server.sendmail(sender_email, receiver_email, msg)
  return {
    "Success": True,
  }



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
def get_render_fname():
  '''
    Returns the filename of the picture of the customer     that was uploaded via the simple webpage.

  1.
    TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
    We wanna try to avoid making a high-detail large-file-size mesh be sent too many times.
    Can we do a function "upload_img_directly_from_gs_util_storage_bucket()" ?
    Or do we have to copy the image locally BEFORE we run SMPLify-X on it?   I could ***probably*** do everything remotely  (on gcloud storage/bucket)  if I installed the GCloud API, but we're very much in "rapid-prototyping-startup-MVP mode" right now.   ***Besides***, the best way we can make the end product ***fastest*** is by using the ***profiler*** at the end, rather than trying to predict which part of the process is going to bottleneck it for the customer.

  2.
  # TODO NOTE:  we have to sync "gs://vrdr_bucket_1/" to '/home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket' MORE OFTEN than every minute (that's what crontab is currently doing)

  3.
  # TODO: error handling.
  '''
  return bucket_diff(
    local_rsync_dir_path= "/home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/img_server/public/",
    file_types=['png', 'jpg', 'jpeg']
  )
#=========================================== end function definition of "get_render_fname():"===========================================






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
  customer_render_path=get_render_fname()
  cust_show_path="/home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/img_server/public/curr_render.png"
  #===================================================================================================
  # Overwrite "cust_show_path" with the current image:
  #===================================================================================================
  cp_cmd=["cp", customer_render_path, cust_show_path]
  cp_success = not sp.call(cp_cmd)
  if not cp_success:
    raise Exception("There was an error when attempting to run    the command  '" + str(cp_cmd) + "'"+ 4*"\n")
  cust_email    = get_cust_email()
  final_success = email_cust_render(
    receiver_email  = cust_email,
    full_render_path= cust_show_path)
  '''
    1. cp to /home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/img_server/public/curr_render.png
    2. E-mail to the      customer E-mail address found in text file /home/cat_macys_vr/web/nodejs-docs-samples/appengine/storage/standard/cust_email.txt.
  '''

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









