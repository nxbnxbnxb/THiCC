#!/bin/bash

# using this hack-y method because I can't get all the modules to ****ing compile with each other.  (cv2 and matplotlib have been the troublesome ones lately, can't get 'em to cooperate with each other (lately as in: Tue Jan 29 07:58:08 EST 2019)

freq=10.00 # TODO NOTE: decrease to chop video up into tinier bits, increase to speed up processing

source /home/n/Documents/code/hmr/venv_hmr/bin/activate &&\
python2 cut_up_video.py \
  $1\
  $2\
  $freq True jpg
  # input vid file is 1st cmd line arg.
  # output img dir is 2nd cmd line arg.  

  #/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/imgs
  #/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/unclothed_outside_delaware_____uniform_background_with_wobbling.mp4

deactivate
# NOTE: all the directories in this command have to match the ones specified in main.py's main() function
