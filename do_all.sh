#!/bin/bash

#=======================================================================
# NOTE:  basically just guidelines; conda doesn't work in here
#=======================================================================

#conda init bash
#conda activate cat
python3 measure.py
#conda deactivate &&\
source /home/n/Documents/code/hmr/venv_hmr/bin/activate &&\
cd /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world &&\
python2 hello_smpl.py 0 0 0 0 0 0 `cat /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/6th_beta.txt` 0 0 0 &&\
blender
