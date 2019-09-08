#!/bin/bash

p3 sync_gsutil_json_bucket_____1_hr.py &
p3 sync_gsutil_img_bucket_____1_hr.py &
conda activate SMPLX &&\
p3  w8_4_json_upload_____then_run_SMPLify-X__.py &



























