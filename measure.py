import json

json_fname='front__nude__grassy_background_keypoints.json'
with open(json_fname) as json_data:
    d = json.load(json_data)
    print(d)
