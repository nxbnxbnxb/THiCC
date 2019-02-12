#!/bin/bash

doc/quick_start.md:68:./build/examples/openpose/openpose.bin --image_dir examples/media/ --face --hand
doc/quick_start.md:95:./build/examples/openpose/openpose.bin --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face
doc/quick_start.md:116:./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1
doc/quick_start.md:118:./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --face --hand
doc/quick_start.md:136:./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --write_json output_folder_path/ --write_video_3d output_folder_path/video_3d.avi
doc/quick_start.md:144:./build/examples/openpose/openpose.bin --flir_camera --num_gpu 0 --write_video output_folder_path/video.avi --write_video_fps 5
doc/quick_start.md:147:./build/examples/openpose/openpose.bin --flir_camera --num_gpu 0 --write_images output_folder_path/ --write_images_format jpg
doc/quick_start.md:157:./build/examples/openpose/openpose.bin --video output_folder_path/video.avi --3d_views 3 --3d --number_people_max 1 --output_resolution {desired_output_resolution}
doc/quick_start.md:159:./build/examples/openpose/openpose.bin --image_dir output_folder_path/ --3d_views 3 --3d --number_people_max 1 --output_resolution {desired_output_resolution}
doc/quick_start.md:166:./build/examples/openpose/openpose.bin --flir_camera --3d --number_people_max 1 --3d_min_views 2 --output_resolution {desired_output_resolution}
doc/quick_start.md:175:./build/examples/openpose/openpose.bin --tracking 5 --number_people_max 1
doc/quick_start.md:181:./build/examples/openpose/openpose.bin --tracking 1 --number_people_max 1
doc/quick_start.md:187:./build/examples/openpose/openpose.bin --tracking 0 --number_people_max 1
doc/installation.md:334:build/examples/openpose/openpose.bin --num_gpu 1 --num_gpu_start 2
doc/installation.md:340:build/examples/openpose/openpose.bin --num_gpu 1 --num_gpu_start 1
doc/demo_overview.md:4:Forget about the OpenPose library code, just compile the library and use the demo `./build/examples/openpose/openpose.bin`.
doc/demo_overview.md:6:In order to learn how to use it, run `./build/examples/openpose/openpose.bin --help` in your bash and read all the available flags (check only the flags for `examples/openpose/openpose.cpp` itself, i.e., the section `Flags from examples/openpose/openpose.cpp:`). We detail some of them in the following sections.
doc/demo_overview.md:43:./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output/ --display 0 --render_pose 0
doc/demo_overview.md:45:./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output/ --display 0 --render_pose 0 --face --hand
doc/demo_overview.md:53:./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_video output/result.avi --write_json output/
doc/demo_overview.md:61:./build/examples/openpose/openpose.bin --hand
doc/demo_overview.md:63:./build/examples/openpose/openpose.bin --hand --hand_scale_number 6 --hand_scale_range 0.4
doc/demo_overview.md:65:./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_detector 3
doc/demo_overview.md:67:./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_scale_number 6 --hand_scale_range 0.4 --hand_detector 3
doc/demo_overview.md:75:./build/examples/openpose/openpose.bin --render_pose 0 --face --face_render 1 --hand --hand_render 1
doc/demo_overview.md:77:./build/examples/openpose/openpose.bin --render_pose 0 --face --face_render 2 --hand --hand_render 2
doc/demo_overview.md:85:./build/examples/openpose/openpose.bin --logging_level 3
doc/demo_overview.md:87:./build/examples/openpose/openpose.bin --logging_level 0
doc/demo_overview.md:95:./build/examples/openpose/openpose.bin --video examples/media/video.avi --num_gpu 2 --num_gpu_start 1
doc/demo_overview.md:103:./build/examples/openpose/openpose.bin --video examples/media/video.avi --heatmaps_add_parts --heatmaps_add_bkg --heatmaps_add_PAFs --write_heatmaps output_heatmaps_folder/
doc/demo_overview.md:109:We enumerate some of the most important flags, check the `Flags Detailed Description` section or run `./build/examples/openpose/openpose.bin --help` for a full description of all of them.
doc/installation_jetson_tx2_jetpack3.1.md:41:./build/examples/openpose/openpose.bin -camera_resolution 640x480 -net_resolution 128x96
doc/standalone_face_or_hand_keypoint_detector.md:9:./build/examples/openpose/openpose.bin --body_disable --face --face_detector 1
doc/modules/calibration_module.md:40:    1. Webcam calibration: `./build/examples/openpose/openpose.bin --num_gpu 0 --write_images {intrinsic_images_folder_path}`.
doc/modules/calibration_module.md:57:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0
doc/modules/calibration_module.md:59:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0 --frame_undistort
doc/modules/calibration_module.md:69:./build/examples/openpose/openpose.bin --num_gpu 0 --video examples/media/video_chessboard.avi --write_images ~/Desktop/Calib_intrinsics
doc/modules/calibration_module.md:71:./build/examples/openpose/openpose.bin --num_gpu 0 --webcam --write_images ~/Desktop/Calib_intrinsics
doc/modules/calibration_module.md:76:./build/examples/openpose/openpose.bin --num_gpu 0 --image_dir ~/Desktop/Calib_intrinsics/ --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
doc/modules/calibration_module.md:78:./build/examples/openpose/openpose.bin --num_gpu 0 --video examples/media/video_chessboard.avi --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
doc/modules/calibration_module.md:80:./build/examples/openpose/openpose.bin --num_gpu 0 --webcam --frame_undistort --camera_parameter_path "models/cameraParameters/frame_intrinsics.xml"
doc/modules/calibration_module.md:87:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 0 --write_images ~/Desktop/intrinsics_0
doc/modules/calibration_module.md:88:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 1 --write_images ~/Desktop/intrinsics_1
doc/modules/calibration_module.md:89:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 2 --write_images ~/Desktop/intrinsics_2
doc/modules/calibration_module.md:90:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --flir_camera_index 3 --write_images ~/Desktop/intrinsics_3
doc/modules/calibration_module.md:99:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --frame_undistort
doc/modules/calibration_module.md:110:./build/examples/openpose/openpose.bin --num_gpu 0 --flir_camera --write_images ~/Desktop/extrinsics
doc/installation_jetson_tx2_jetpack3.3.md:41:./build/examples/openpose/openpose.bin -camera_resolution 640x480 -net_resolution 128x96
README.md:136:./build/examples/openpose/openpose.bin --video examples/media/video.avi
include/openpose/flags.hpp:15:// See all the available parameter options withe the `--help` flag. E.g., `build/examples/openpose/openpose.bin --help`
examples/tutorial_api_thread/2_thread_user_input_processing_output_and_datum.cpp:21:// See all the available parameter options withe the `--help` flag. E.g., `build/examples/openpose/openpose.bin --help`
a_useful_openpose_cmd.sh:3:./build/examples/openpose/openpose.bin --image_dir /home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs/2019_01_30____07:18_AM__nathan_front/ --write_json /home/ubuntu/Documents/code/openpose/output --write_images /home/ubuntu/Documents/code/openpose/output --write_images_format jpg --display 0 --render_pose 1
scripts/travis/run_tests.sh:14:  ./build/examples/openpose/openpose.bin --image_dir examples/media/ --net_resolution -1x32 --write_json output/ --write_images output/ --display 0
scripts/tests/pose_accuracy_coco_val_foot.sh:15:OP_BIN=./build/examples/openpose/openpose.bin
scripts/tests/pose_accuracy_coco_test_dev.sh:16:OP_BIN=./build/examples/openpose/openpose.bin
scripts/tests/pose_time_visual_GUI.sh:14:./build/examples/openpose/openpose.bin --video soccer.mp4 --frame_last 1500
scripts/tests/pose_time_visual_GUI.sh:16:# ./build/examples/openpose/openpose.bin --video soccer.mp4 --frame_last 3750
scripts/tests/pose_accuracy_car_val.sh:26:OP_BIN=./build/examples/openpose/openpose.bin
scripts/tests/pose_accuracy_coco_val.sh:16:OP_BIN=./build/examples/openpose/openpose.bin
scripts/tests/hand_accuracy_test.sh:68:./build/examples/openpose/openpose.bin \
scripts/tests/hand_accuracy_test.sh:81:./build/examples/openpose/openpose.bin \


