#!/bin/bash

#ssh -i 1st_VRMall.pem ec2-user@ec2-18-220-116-193.us-east-2.compute.amazonaws.com #1st
#ssh -i 1st_VRMall.pem ubuntu@ec2-3-16-166-14.us-east-2.compute.amazonaws.com      #2nd


#ssh -i 1st_VRMall.pem ubuntu@ec2-18-218-174-60.us-east-2.compute.amazonaws.com # elasticBeanstalk  TODO TODO NOTE: doesn't work!

ssh -i ~/1st_VRMall.pem ubuntu@$AWS_IP
