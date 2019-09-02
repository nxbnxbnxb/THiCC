'''
  TODO: 
    1.  Automatically   empty gsutil bucket "gs://vrdr_bucket_1/"   and     
      /home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket   .      Currently I have to do it manually, but once we get 1,000+ customers, image file overflow will start to be a serious problem.



'''



import time
import subprocess; sp = subprocess
# open-ended (don't HAVE to use the shortcut, but it's there for us.)            ("sp" vs. "subprocess")
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


#===================================================================================================
if __name__=="__main__":
  '''
    Rapid prototyping:   M.V.MP.
  '''
  #tmp_dir_name=mkdir()
  encoding='utf-8'

  # TODO NOTE:  we have to sync "gs://vrdr_bucket_1/" to '/home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket' MORE OFTEN than every minute (that's what crontab is currently doing)
  local_bucket_fname='/home/cat_macys_vr/vrdr_customer_imgs________gsutil_bucket/'
  cmd=['ls', '-tr', local_bucket_fname]
  # this was the "diff"  :   -rw-r--r--1cat_macys_vrcat_macys_vr0Sep202:35sigmoid.png_.gstmp
  # I put the sync in editcron. (/etc/crontab)        ("sync" more technically means    `gsutil rsync`.   For some reason this (["gsutil", "ls", bucket_name])  was being weird to call within python3 )



  #bucket_name="gs://vrdr_bucket_1/"
    #["gsutil", "ls", bucket_name]   # this (["gsutil", "ls", bucket_name]) was giving us trouble for some reason.
  # OLD comment.  Only left here briefly so I remember why I changed the code to its current state. -nxb, Sun Sep  1 21:38:26 EDT 2019    
  #   The old comments:   NOTE:  here, I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
  #                       NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )


  # Initial "ls -ltrAh /path/to/imgdir/"      (like do-while loop) :
  bucket_proc = sp.Popen(cmd, stdout=sp.PIPE) 
  # Description of "sp.Popen():"      https://docs.python.org/3/library/subprocess.html   and   https://stackoverflow.com/questions/89228/calling-an-external-command-in-python/92395#92395 .
  prev_bucket_contents, err = bucket_proc.communicate()
  prev_bucket_contents= prev_bucket_contents.decode(encoding)
  assert not err

  secs_waited=0
  go_again=True
  #===================================================================================================
  while go_again:
    time.sleep(1)
    if secs_waited%5==0:
      print("Been waiting "+str(secs_waited)+" seconds for an image file.") 
    secs_waited+=1

    bucket_proc = sp.Popen(cmd, stdout=sp.PIPE)
    bucket_contents, err = bucket_proc.communicate()
    bucket_contents=bucket_contents.decode(encoding)
    assert not err
    # TODO: error handling.
    print(bucket_contents)
    if err:
      pass


    #======================================================================================================================================================================================================
      # NOTE:  here I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  
      #        ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
      # NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )
    #======================================================================================================================================================================================================
    else: #(if not err):

      new_img_uploaded  = not (bucket_contents == prev_bucket_contents)
      print("new_img_uploaded? ",new_img_uploaded)
      if new_img_uploaded:
        print(" bucket_contents:",bucket_contents)
        print("  prev_bucket_contents:",prev_bucket_contents)
        '''
        # TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
        # We wanna try to avoid making a high-detail large-file-size mesh be sent too many times.
        # Can we do a function "upload_img_directly_from_gs_util_storage_bucket()" ?
        # Or do we have to copy the image locally BEFORE we run SMPLify-X on it?   I could ***probably*** do everything remotely  (on gcloud storage/bucket)  if I installed the GCloud API, but we're very much in "rapid-prototyping-startup-MVP mode" right now.   ***Besides***, the best way we can make the end product ***fastest*** is by using the ***profiler*** at the end, rather than trying to predict which part of the process is going to bottleneck it for the customer.
        '''

        # Basically, the next line of code is "img_fname  =  curr - prev"  (the next line of code is "str1_minus_2(curr, prev)")
        img_fname=str1_minus_2(bucket_contents, prev_bucket_contents)
        print('\n'+'='*99)
        print("There's an image from the customer.")
        print('='*99+'\n')
        print('img_fname[-9:]: ',img_fname[-9:])    # There **should** be 9 characters printed.
        img_path=local_bucket_fname+img_fname
        # This was `img_fname` : -rw-r--r--1cat_macys_vrcat_macys_vr0Sep202:35sigmoid.png_.gstmp     (solved this bug with the "if not ..." check right below this line)

        img_fname=img_fname[:-1] # This line `img_fname=img_fname[:-1]` cuts out the newline character.
        if not   ( img_fname.endswith('png') or img_fname.endswith('jpeg') or img_fname.endswith('jpg') ):
          print('Skipping 1 loop iteration')
          continue

        # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string  (to understand "bytes_obj.decode(...)")

        #failed=sp.call(['gsutil', 'cp', bucket_name+img_fname, './'+tmp_dirname])
        # I'm probably making some assumptions to the effect of    "we didn't just change directories somehow, and './' from os.getcwd() is the same dir as the one where I locally "gsutil copied" the image to in the first place.   -nxb, Sun Sep  1 20:23:02 EDT 2019
        go_again=False
        print("Now leaving the 'while' loop.")

        #======================================================================================================================================================================================================
        # TODO.
        #smplifyx(curr_img_path)  # This can't be opening any GUIs.  TODO TODO TODO   NOTE  TODO TODO TODO.
        #======================================================================================================================================================================================================
      #============================================================== end "if new_img_uploaded:" ===============================================================
    prev_bucket_contents=bucket_contents
  #===============================================================end "while True:"===============================================================
   #failed=sp.call(['rm', '-rf', tmp_dir_name])   # NOTE:   can also do "succeeded = not sp.call(...)"




































































































