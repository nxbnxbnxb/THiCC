#!/bin/bash

# This one WORKS.  Before you throw in a million flags, make sure it WORKS (incremental improvements)
# Maybe it can't both write json AND save the right image in the same command?


# Nathan's measurements, pic of him, etc.
cd /home/n/openpose && \
./build/examples/openpose/openpose.bin \
  --render_pose 0 \
  --display 0 \
  --image_dir /home/n/Dropbox/vr_mall_backup/imgs/n8_front___jesus_legs_closed/ \
  --write_json /home/n/Dropbox/vr_mall_backup/json_imgs_openpose/
# OpenPose demo successfully finished.  Total time:   319.135540 seconds.
# time to run :                         Total time:  1501.079810 seconds.




































































# Old image dir:
#/home/n/Dropbox/vr_mall_backup/imgs/n8_front_nude____/ \
