import subprocess; sp = subprocess
# open-ended (don't HAVE to use the shortcut, but it's there for us.)            ("sp" vs. "subprocess")
import difflib
from collections import OrderedDict as OrD











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
  # end while not "made_dir"
  #================================================================================

  return dir_name+ str(i-1) +  '/'
  # `dir_name+str(i-1)`    is the correct name of the actual directory name  because  "i" got incremented 1 too many times during the "while not made_dir:" loop
#====================================== end function definition of "mkdir():"======================================


#===================================================================================================
if __name__=="__main__":
  bucket_name=''
  tmp_dir_name=mkdir()
  encoding='utf-8'

  prev_bucket_contents=None          # TODO:  Handle this "do while"-esque case.
  bucket_name="gs://vrdr_bucket_1/"
  cmd=["gsutil", "ls", bucket_name]
  # NOTE:  here, I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
  # NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )

  tmp_fname=tmp_dir_name+"/tmp.png"   # NOTE:   Do we have to know the image file type   a priori?  ie. if we get a jpeg but we call it a .png, will imageio bitch at us?

  while True:
    bucket_proc = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
    bucket_contents, err = bucket_proc.communicate()

    # TODO: error handling.
    if err:
      pass
    else: #(if not err):
      new_img_uploaded  = not (bucket_contents == prev_bucket_contents)
      if new_img_uploaded:
        # TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
        # We wanna try to avoid making a high-detail large-file-size mesh be sent too many times.

        # Can we do a function "upload_img_directly_from_gs_util_storage_bucket()" ?
        # Or do we have to copy the image locally BEFORE we run SMPLify-X on it?   I could ***probably*** do everything remotely  (on gcloud storage/bucket)  if I installed the GCloud API, but we're very much in "rapid-prototyping-startup-MVP mode" right now.   ***Besides***, the best way we can make the end product ***fastest*** is by using the ***profiler*** at the end, rather than trying to predict which part of the process is going to bottleneck it for the customer.
        # Basically, the next line of code is "img_fname  =  curr - prev"
        img_fname=str1_minus_2(bucket_contents.decode(encoding), prev_bucket_contents.decode(encoding))
        # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string  (to understand "bytes_obj.decode(...)")
        failed=sp.call(['gsutil', 'cp', dirname+img_fname, './',])
        smplx(img_path)


      #======================================================================================================================================================================================================
        # NOTE:  here, I assume the output of "gsutil ls [bucket-name]" is INVARIANT as long as the bucket hasn't been changed.  ie. `gsutil ls buck` doesn't spit out "buck/1.txt" then "buck/2.txt"    and then 5 seconds later spit out "buck/2.txt" then "buck/1.txt".
        # NOTE:  this assumption might be WRONG.  (If our assumption is WRONG, it would make our code not merely *slow*, but also send back the WRONG mesh [wrong person's body measurements]. )
      #======================================================================================================================================================================================================



































































































