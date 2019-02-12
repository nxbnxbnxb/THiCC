#!/bin/bash

img_dir_input=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_02_03____07\:18_AM__smpl_male_0000000000 #9.745/
json_dir=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/json_imgs_openpose/
output_format=jpg
output_dir_imgs=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/keypoints_imgs/default/
render=1 # 1 is True

do_openpose=/home/ubuntu/Documents/code/openpose/build/examples/openpose/openpose.bin

$do_openpose --image_dir $img_dir_input --num_gpu 0 --write_json $json_dir --write_images_format $output_format --write_images $output_dir_imgs --display 2 --render_pose $render

# --face --hand
