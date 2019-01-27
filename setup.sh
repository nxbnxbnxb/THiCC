#!/bin/bash
#
#     Dec. 27, 2018
#
#########################################
#       Installation time (approx)
#
#            real   1m34.194s
#            user   1m10.520s
#            sys    0m7.909s
#########################################
#
#

# download model necessary for segmentation images of people
wget http://columbia.edu/~nxb2101/deeplab_model.tar.gz

# new env
conda create -y --name cat python=3.6.6 && source activate cat
# NOTE: this one is done so far.

# install
conda install -yc conda-forge scikit-image==0.14.1 && pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl && conda install -y matplotlib==2.2.3
# '-y' flag is for auto-yes to all conda prompts

# as of Dec. 27, 2018, :
  # skimage   also installs six, qt, pyqt, imageio, matplotlib, scipy, and numpy 1.15.4
  # tf CPU, py 3.6
  # the newest matplotlib crashes my machine, so I put the 2nd's version info at the end


# test proper install by calling python3.6 tests/install_tests/all_tests.py
cd tests/install_tests/ && python3.6 all_tests.py




#TODO:
  #cv ?
  # we were having trouble with compatibility for this.  Mon Jan 14 14:50:21 EST 2019
