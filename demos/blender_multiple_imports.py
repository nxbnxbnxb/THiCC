import os
import bpy

# put the location to the folder where the objs are located here in this fashion
# this line will only work on windows ie C:\objects
path_to_obj_dir = '/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smpl/smpl_webuser/hello_world'

# get list of all files in directory
file_list = sorted(os.listdir(path_to_obj_dir))

# get a list of files ending in 'obj'
obj_list = [item for item in file_list if item.endswith('.obj')]

# loop through the strings in obj_list and add the files to the scene
for item in obj_list:
    path_to_file = os.path.join(path_to_obj_dir, item)
    bpy.ops.import_scene.obj(filepath = path_to_file)