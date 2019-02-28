#!/bin/bash

# This one WORKS.  Before you throw in a million flags, make sure it WORKS (incremental improvements)

# Nathan's measurements, pic of him, etc.
cd /home/n/openpose && \
./build/examples/openpose/openpose.bin \
  --image_dir /home/n/Dropbox/vr_mall_backup/imgs/n8_front___jesus_legs_closed/ \
  --render_pose 0 \
  --display 0 \
  --num_gpu 0 \
  --write_json /home/n/Dropbox/vr_mall_backup/IMPORTANT/
# Should work for JSON.  If it doens't, you may have to rearrange the cmd line arguments.


  # maybe it can't both write json AND save the right image in the same command?
# time to run :   Total time: 1501.079810 seconds.





'
  --write_images /home/n/Dropbox/vr_mall_backup/imgs/openpose_keypoints___n8_side_nude_jesus_pose_legs_closed___grassy_background_Newark_DE \
  --write_images_format jpg \
  --render_pose 1 \
'































































# Old image dir:
#/home/n/Dropbox/vr_mall_backup/imgs/n8_front_nude____/ \
