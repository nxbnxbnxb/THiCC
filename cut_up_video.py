from utils import save_mp4_as_imgs
import sys

def cut():
  if len(sys.argv) < 6:
    print ("error, too few args given to function save_mp4_as_imgs()")
    return
  else:
    local_vid_path = sys.argv[1]
    root_img_dir   = sys.argv[2]
    freq=float(sys.argv[3])
    print("freq is {0}".format(freq))
    should_put_timestamps=bool(sys.argv[4])
    output_img_filetype=sys.argv[5] # ie. jpg, png, etc.
    save_mp4_as_imgs(local_vid_path, root_img_dir, freq=freq, should_put_timestamps=should_put_timestamps, output_img_filetype=output_img_filetype)
    return

if __name__=="__main__":
  cut()

