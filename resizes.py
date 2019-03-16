from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from viz import pltshow

from PIL import Image
import sys
import imageio as ii

fname=sys.argv[1]#"/home/n/Dropbox/vr_mall_backup/IMPORTANT/n8_front___jesus_pose___legs_closed___nude___grassy_background_Newark_DE____.jpg"
n8_img=Image.open(fname)

newsize=(int(round(n8_img.size[0] / (n8_img.size[1]/150.))),
         int(round(n8_img.size[1] / (n8_img.size[1]/150.))))
resized=n8_img.resize(newsize,Image.BICUBIC)
ii.imwrite('resized.jpg',resized)
pltshow(resized)



