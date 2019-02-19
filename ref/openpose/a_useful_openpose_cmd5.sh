#!/bin/bash

# This one WORKS.  Before you throw in a million flags, make sure it WORKS (incremental improvements)

# Nathan's measurements, pic of him, etc.
cd /home/n/openpose && \
./build/examples/openpose/openpose.bin \
  --image_dir /home/n/Dropbox/vr_mall_backup/imgs/n8_front_nude____/ \
  --render_pose 0 \
  --display 0 \
  --write_json /home/n/Dropbox/vr_mall_backup/IMPORTANT/ \
    #/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/openpose_json
  --num_gpu 0
# time to run :   Total time: 1501.079810 seconds.


