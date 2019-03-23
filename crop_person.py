from seg import *
import seg

if __name__=="__main__":
  if len(sys.argv) == 2:
    img_path=sys.argv[1]
    print ("currently segmenting image found at location: \n  "+img_path)
    img=np_img(img_path)
    no_background, segmap = segment_black_background(img_path)
    cropped,_=crop_person(img,segmap)
    crop_fname='cropped.png'
    ii.imwrite(crop_fname,cropped.astype('uint8'))
    no_background, segmap = segment_black_background(crop_fname)
    ii.imwrite("mask.png",segmap.astype('float64'))#.astype("uint8"))
