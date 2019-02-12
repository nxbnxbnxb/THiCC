#!/bin/bash

# TODO: rename the user on the AWS EC instance.  (AND every new AWS instance you make)
#       if u CAN'T do this, THEN rename /home/ubuntu/ to a single variable and use it everywhere, renaming it /home/ubuntu/ when necessary

upload_dir=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/customer_imgs/ # /home/ubuntu/  /home/ubuntu/ # TODO: rename the user on the AWS EC instance.  (AND every new AWS instance you make)
log=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/most_recent_img_upload_loc.txt
hmr_dir=/home/ubuntu/Documents/code/old/hmr # NOTE: watch the trailing forward slash ('/')

pier=ec2-user@ec2-3-17-132-170.us-east-2.compute.amazonaws.com:/home/ec2-user/ # pierlorenzo; the IP WILL change
mesh=/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/mesh.obj
activate=/home/ubuntu/Documents/code/hmr/venv_hmr/bin/activate

inotifywait -m $upload_dir -e create -e moved_to |
    while read $path action file; do
        echo "The file '$file' appeared in directory '$path' via '$action'"  # "action" isn't currently working
        rm $log                             # todo: keep old logs of this?
        echo $upload_dir >> $log
        echo $file >> $log
        source $activate && python2 $hmr_dir/utils.py $log && python2 $hmr_dir/demo.py --img_path `cat $hmr_dir/tmp.txt` # TODO: put /home/ubuntu in a variable and change it back to /home/ubuntu/
        # ./hang_and_then_move_uploaded_image.sh: line 13: 13875 Killed                  python2 /home/ubuntu/Documents/code/old/hmr/demo.py --img_path `cat /home/ubuntu/Documents/code/old/hmr/tmp.txt`  todo: debug?
        #scp $mesh
        #scp -i ~/1st_VRMall.pem  /home/ubuntu/tmp.py  ubuntu@ec2-18-217-52-160.us-east-2.compute.amazonaws.com:~ #<--   NOTE: this command works from Ubuntu laptop to THIS AWS instance.
        sudo scp -i /home/ubuntu/1st_VRMall.pem $mesh $pier
    done
# TODO TODO TODO: send mesh.obj to Pier after it's created.

#source /home/ubuntu/Documents/code/hmr/venv_hmr/bin/activate && python2 /home/ubuntu/Documents/code/old/hmr/demo.py
#source /home/ubuntu/Documents/code/hmr/venv_hmr/bin/activate && python2 /home/ubuntu/Documents/code/old/hmr/demo.py
# todo: this is what we SHOULD do, but then I'd need conda to work properly with this ***t.  
# comment on the above todo (new timestamp (Sun Feb 10 18:14:20 UTC 2019)) would I need conda to work properly with this shit? ...


#/home/ubuntu/Documents/code/old/hmr/src/util/renderer.py:322: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  #if np.issubdtype(image.dtype, np.float):
#Traceback (most recent call last):
  #File "/home/ubuntu/Documents/code/old/hmr/demo.py", line 228, in <module>
    #fix()
  #File "/home/ubuntu/Documents/code/old/hmr/demo.py", line 64, in fix
    #sp.call(['mv', '-f', fresh_outmesh_path, outmesh_path])
  #File "/usr/lib/python2.7/subprocess.py", line 172, in call
    #return Popen(*popenargs, **kwargs).wait()
  #File "/usr/lib/python2.7/subprocess.py", line 394, in __init__
    #errread, errwrite)
  #File "/usr/lib/python2.7/subprocess.py", line 938, in _execute_child
    #self.pid = os.fork()
#OSError: [Errno 12] Cannot allocate memory
#The authenticity of host 'ec2-3-17-132-170.us-east-2.compute.amazonaws.com (172.31.26.36)' can't be established.
#ECDSA key fingerprint is SHA256:GNb77YQ7l0TzCE5hfqGoSgwprMEwiV0KgQt4jQJChSs.
#Are you sure you want to continue connecting (yes/no)? yes
#Warning: Permanently added 'ec2-3-17-132-170.us-east-2.compute.amazonaws.com,172.31.26.36' (ECDSA) to the list of known hosts.
#scp: /home/ubuntu: Permission denied

