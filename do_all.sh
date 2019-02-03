#!/bin/bash

#===============================================================================================
#   TODO:  basically just guidelines; activating virtualenv or conda doesn't work in here
#===============================================================================================

cd /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018
conda activate cat
python3 measure.py
conda deactivate &&\
source /home/n/Documents/code/hmr/venv_hmr/bin/activate &&\
cd /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world &&\
# Male:     this positioning of the chest-width-to-hip-width ([9] or 10th index) is only good for a male
python2 hello_smpl.py 0 0 0 0 0 0 0 0 0 `cat /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/6th_beta.txt` &&\
# Female:   this positioning of the chest-width-to-hip-width ([6] or 7th index) is only good for a female
#python2 hello_smpl.py 0 0 0 0 0 0 `cat /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/6th_beta.txt` 0 0 0 &&\
blender

deactivate
cd /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018  # go back to main vr_mall dir

