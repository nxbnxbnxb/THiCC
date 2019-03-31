from measure import *

if __name__=="__main__":
  # only 100% good for openpose 1.2.0
  openpose_keypoints_json_fname=sys.argv[1]
  out_dict_fname=sys.argv[2]
  pprint_json(openpose_keypoints_json_fname, out_dict_fname)
