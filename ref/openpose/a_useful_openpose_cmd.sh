#!/bin/bash

img_dir_input=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/tall_women #9.745/
json_dir=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/openpose_json
output_format=jpg
output_dir_imgs=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/openpose_keypoints
render=1 # 1 is True

do_openpose=/home/n/Documents/code/openpose/build/examples/openpose/openpose.bin

$do_openpose --image_dir $img_dir_input --num_gpu 0 --write_json $json_dir --write_images_format $output_format --write_images $output_dir_imgs --display 2 --render_pose $render

# --face --hand
# try 2:
# /home/n/Documents/code/openpose/build/examples/openpose/openpose.bin --render_pose 1 --image_dir /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/tall_women/ --write_json  /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/json --display 2 --num_gpu 0 --write_images /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/openpose_keypoints/ --write_images_format jpg

