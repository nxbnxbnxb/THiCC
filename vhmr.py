import skimage.io as skio
from demo import main_nathan_0 as main0

#=======================================================
def make_mesh(vid_fname, secs_between_frames):
  '''
  ------
  Params:
    vid_fname is the fullpath, not a localpath

  '''

  import cv2 # TODO: resolve this s.t. either base or conda or a virtualenv has access to every damn python module I need
  delay = secs_between_frames
  vidcap = cv2.VideoCapture(vid_fname)
  success = True # THOUGHT: there's gotta be a smarter way to do this without the special case at the beginning.
  count = 0
  while success:
    success, img = vidcap.read()
    main0(img,json_path=None) # TODO: deal with the fact that the original img came with a wrapper (io.imread()).  I HOPE it's numpy under the ****ing hood
    if count < 1:
      if debug:
        print('We just read a new frame, true or false? ', success)
    if cv2.waitKey(delay) &  0xFF == ord('q'): # NOTE: double-check exactly what this "ord('q')" means.
      break
    # NOTE:  "I put the imwrite() AFTER the break so if the image fails to read, we don't save an empty .jpg file."   - Nathan (Mon Jan 14 13:33:26 EST 2019)
    cv2.imwrite(img_write_dir+"{0}.{1}".format(prepend_0s(str(count)),output_img_filetype), img) # TODO: there's probably a better way to do the prepend_0s() function
    count += 1
  return 0 # success
#===== end func def of  make_mesh(**lotsa_params): =====
