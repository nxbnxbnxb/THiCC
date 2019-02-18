#!/bin/bash

img_dir_in=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/
img_dir_in=$1  # NOTE: ideal is to take cmd line args to 
img_dir_in=/home/n/Pictures/small_openpose

./build/examples/openpose/openpose.bin --image_dir $img_dir_in --num_gpu 0 --write_json /home/n/Documents/code/openpose/output_json --write_images_format jpg --write_images /home/n/Documents/code/openpose/output --display 0 --render_pose 1
# TODO: no-gpu, other flags worth reading thru.
# render_pose 1 means "CPU" 


