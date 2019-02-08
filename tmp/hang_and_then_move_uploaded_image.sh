#!/bin/bash

upload_dir=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/customer_imgs/
log=/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/most_recent_img_upload_loc.txt

inotifywait -m $upload_dir -e create -e moved_to |
    while read $path action file; do
        echo "The file '$file' appeared in directory '$path' via '$action'"  # "action" isn't currently working
        rm $log                             # TODO: keep old logs of this?
        echo $upload_dir >> $log
        echo $file >> $log
        source /home/n/Documents/code/hmr/venv_hmr/bin/activate && python2 /home/n/Documents/code/old/hmr/utils.py /home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/most_recent_img_upload_loc.txt && python2 /home/n/Documents/code/old/hmr/demo.py --img_path `cat /home/n/Documents/code/old/hmr/tmp.txt`

        # TODO: string processing in python :     'action' is either "moved" or "created";  we have to somehow take the 'tail' of the $file string
    done

#source /home/n/Documents/code/hmr/venv_hmr/bin/activate && python2 /home/n/Documents/code/old/hmr/demo.py
#source /home/n/Documents/code/hmr/venv_hmr/bin/activate && python2 /home/n/Documents/code/old/hmr/demo.py # TODO: this is what we SHOULD do, but then I'd need conda to work properly with this ***t

