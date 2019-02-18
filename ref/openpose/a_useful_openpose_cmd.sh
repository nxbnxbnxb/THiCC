#!/bin/bash
  #Openpose will always just be slow.  If you want to use it, you have to deal with this fact & hack around it; either by buying a better AWS CPU briefly or making the customer wait or... idk, some combination of stuffs like that.
img_dir_in=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/tall_women #9.745/
img_dir_in=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/
img_dir_in=$1  # NOTE: ideal is to use cmd line args to take img_dir_in put directory
img_dir_in=/home/n/Pictures/small_openpose #_2/   # NOTE: tiny img of woman from online, but with proper aspect ratio.


json_dir=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/openpose_json
img_dir_out=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/openpose_keypoints
render=1 # 0 is no render; 1 is CPU; 2 is GPU
do_openpose=/home/n/Documents/code/openpose/build/examples/openpose/openpose.bin
disp=2 # 0 for no disp, 2 for 2-D, 3 for 3-D

$do_openpose \
  --image_dir $img_dir_in \
  --num_gpu 0 \
  --write_json $json_dir \
  --write_images_format jpg \
  --write_images $img_dir_out \
  --display $disp \
  --render_pose $render
  #-net_resolution -1x368

# TODO: net_resolution should (almost) always be ~= TV screen (1980x1080, 1280x720, -1x368 etc.).   -net_resolution 'The default dynamic `--net_resolution` is not supported in MKL (MKL CPU Caffe) and OpenCL Caffe versions. Please, use a static `net_resolution` (recommended `--net_resolution 656x368`) or use the Caffe CUDA master branch when processing images and/or when using your custom image reader. OpenPose has automatically set the resolution to 656x368.'

# --face --hand
# try 2:
# /home/n/Documents/code/openpose/build/examples/openpose/openpose.bin --render_pose 1 --image_dir /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/tall_women/ --write_json  /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/json --display 2 --num_gpu 0 --write_images /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/openpose_keypoints/ --write_images_format jpg

