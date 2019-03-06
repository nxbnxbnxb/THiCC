from utils import save_vid_as_imgs
import sys
import time

def cut():
  if len(sys.argv) < 6:
    print('To properly control this cut(vid) with command line args, please input more args.  Cutting the default video...')
    if len(sys.argv)>1:
      delay=float(sys.argv[1])
    else:
      delay=60.
    print('also sleeping for {0} seconds to make sure this is what you wanted to do'.format(delay))
    time.sleep(delay)
    local_vid_path = \
      "/home/n/Dropbox/vr_mall_backup/IMPORTANT/nathan_jesus_pose_legs_together_0227191404.mp4"
    root_img_dir   = "/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs"
    secs_btwn_frames=0.1#0.25
    should_put_timestamps=True
    output_img_filetype='jpg' # ie. jpg, png, etc.
  else:
    local_vid_path = sys.argv[1]
    root_img_dir   = sys.argv[2]
    secs_btwn_frames=float(sys.argv[3])
    should_put_timestamps=bool(sys.argv[4])
    output_img_filetype=sys.argv[5] # ie. jpg, png, etc.
  img_dir=save_vid_as_imgs(local_vid_path, root_img_dir, secs_btwn_frames=secs_btwn_frames, should_put_timestamps=should_put_timestamps, output_img_filetype=output_img_filetype)
  return img_dir

if __name__=="__main__":
  cut()

