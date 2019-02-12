#!/bin/bash

# using this hack-y method because I can't get all the modules to ****ing compile with each other.  (cv2 and matplotlib have been the troublesome ones lately, can't get 'em to cooperate with each other (lately as in: Tue Jan 29 07:58:08 EST 2019)

freq=0.5 # NOTE: decrease to chop video up into tinier bits, increase to speed up processing

source /home/ubuntu/Documents/code/hmr/venv_hmr/bin/activate &&\
python2 cut_up_video.py /home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/unclothed_outside_delaware_____uniform_background_with_wobbling.mp4 /home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs $freq True jpg &&\
deactivate
# NOTE: all the directories in this command have to match the ones specified in main.py's main() function
