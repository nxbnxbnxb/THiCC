from __future__ import division
from PIL import Image
from viz import pltshow
fname="/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg"
n8_img=Image.open(fname)

newsize=(int(round(n8_img.size[0] / (n8_img.size[1]/150.))),
         int(round(n8_img.size[1] / (n8_img.size[1]/150.))))
resized=n8_img.resize(newsize,Image.BICUBIC)
pltshow(resized)



