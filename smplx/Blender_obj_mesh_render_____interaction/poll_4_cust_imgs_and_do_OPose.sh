#!/bin/bash
python3 ./data/sync_gsutil_buckets_____1_hr.py &
python3 ./data/w8_4_img_upload_____then_run_OPose__.py &

#gsutil cp /openpose/output/keypoints_json/*.json gs://vrdr_OP_json     # <== I put this "gsutil cp" command in "w8_4_img_upload_____then_run_OPose__.py"

# Docker spits out "your platform doesn't support ssh." or something like this when I run "gcloud compute scp."
#gcloud compute scp --project "secret-voice-243500" --zone "us-east1-d" /openpose/data/images/* cat_macys_vr@cuda-version-test-0-vm:/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/nxb_data/images
#gcloud compute scp --project "secret-voice-243500" --zone "us-east1-d" /openpose/output/keypoints_json/* cat_macys_vr@cuda-version-test-0-vm:/home/cat_macys_vr/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/smplx/githubs/smplify-x/nxb_data/openpose_keypoints


/openpose/data/clean_customer_img_bucket.sh # Remove customer img to prepare for the next one.













