#!/bin/bash

# This one WORKS.  Before you throw in a million flags, make sure it WORKS (incremental improvements)
cd ~/home/n/openpose &&\
./build/examples/openpose/openpose.bin --image_dir examples/media/ --render_pose 0 --write_images output/ --display 0 --write_json output/ --write_json output/
# time to run :   Total time: 1501.079810 seconds.

