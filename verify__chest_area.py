import matplotlib.pyplot as plt
import numpy as np

def chest_test():
  # see front__nude__grassy_background_keypoints.json  
  #   And https://stackoverflow.com/questions/1329546/whats-a-good-algorithm-for-calculating-the-area-of-a-quadrilateral
  x0,y0 = 435.225,676.318
  x1,y1 = 650.009,660.497
  x2,y2 = 498.099,1011.46
  x3,y3 = 623.893,1001
  chest=np.array([[x0,y0],
                  [x1,y1],
                  [x2,y2],
                  [x3,y3]]).astype('float64')
  x=0
  y=1
  plt.scatter(chest[:,x],chest[:,y])
  plt.show()

  x2_line = x2 - x0
  y2_line = y2 - y0
  area = 0.5 * abs( y2_line * (x1 - x0 + x3 - x0) + x2_line * (y1 - y0 + y3 - y0) )

  #area = 0.5 * abs( x0 * y1 - x1 * y0 + x1 * y2 - x2 * y1 + 
                    #x2 * y3 - x3 * y2 + x3 * y0 - x0 * y3 )
  return area
if __name__=="__main__":
  area=chest_test()
  print(area)
