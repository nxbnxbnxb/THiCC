#!/bin/bash


# The following "conda create ..." and "conda activate ..." breaks within this shitty bash script, but it still shows you how this is done

echo "==============================================================================================================================================================================================================="
echo "==============================================================================================================================================================================================================="
echo "==============================================================================================================================================================================================================="
echo "                                    Please NOTE:                                          "
echo "                 CONDA VERSION 4.6.1 assumed in this script.                              "
echo "        Also, you may have to manually execute each command within this bash script;      "
echo "  I have included this message and this 'pause' so you pay attention and save yourself some time. "
echo "==============================================================================================================================================================================================================="
echo "==============================================================================================================================================================================================================="
echo "==============================================================================================================================================================================================================="

sleep 60 # stalls the user to make them read.



#env=cat12
## TODO: pip version number == 19.0.1
## TODO: conda version number == 4.6.1
#conda create -y --name $env python=3.6.8
#conda activate $env # as of now, I don't know how to make "conda activate" work within bash script.  TODO NOTE

pip install pip==19.0.1 &&\
conda install -yc menpo opencv==3.4.2  &&\
conda install -y matplotlib=2.2.3  &&\
conda install -yc anaconda scikit-image && pip install --upgrade scikit-image==0.14.2 &&\
conda install -y sympy==1.3  &&\
pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl &&\
python3.6 /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/tests/install_tests/all_tests.py &&\
python3.6 /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/tests/install_tests/test_imports.py
# cmd line args





































































