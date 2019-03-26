
import glob
import os

eq_50='='*50
print (eq_50+'\n'+eq_50)

py_filenames = glob.glob('*test*.py')
print("running code from files: ",py_filenames)
for test in py_filenames:
  os.system('python '+test)
# TODO:  for every last func call I use, test it in this folder
