import subprocess as sp
import time
import os

def sync():
  local_dir="/openpose/data/vrdr_customer_imgs________gsutil_bucket/images/"
  bucket="gs://vrdr_bucket_1/"
  gsutil_sync_cmd=[
    "gsutil",
    "rsync",
    bucket,
    local_dir,
    #">>",
    #"./gsutil_rsync_____log.txt"
  ]
  success = not sp.call(gsutil_sync_cmd)
  if not success:
    raise Exception("gsutil rsync command failed!   The full text of rsync command we attempted to run is \n{0} ".format(gsutil_sync_cmd))
  #os.system("gsutil rsync "+ bucket+ " "+ local_dir+" >> ./gsutil_rsync_____log.txt")

  return {
    "Local Directory that's now Synced with the remote gcloud storage bucket": local_dir,
    "gcloud storage bucket name": bucket
  }
#====================================== end function definition of " sync():"======================================






if __name__=="__main__":
  hour=3600 # 3600 **seconds**.
  for i in range(hour):
    sync()
    time.sleep(1)
    if i%60==0:
      print("1 minute passed!")



































































































