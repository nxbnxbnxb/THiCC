import numpy as np

def Jonah_Hill():
  Thetas=np.array([
    [ 1.59637403e+00,   3.68165188e-02,   6.11614883e-01,   2.93165731e+00,
     -6.52522221e-02,   1.54586732e-01,  -3.29840362e-01,   2.00239588e-02,
      3.19421813e-02,  -5.80910802e-01,   1.37323420e-03,  -6.45631328e-02,
      3.87461931e-01,  -1.16511047e-01,   6.42893929e-03,   4.95812535e-01,
      4.38999534e-02,  -6.74179271e-02,   9.34259415e-01,  -1.16546549e-01,
      5.52092344e-02,  -4.59641293e-02,  -2.04198603e-02,  -2.24000476e-02,
     -8.83596689e-02,   2.77116060e-01,   1.46933924e-02,  -2.57417560e-02,
     -2.95253992e-01,   2.57571906e-01,   4.59690168e-02,  -1.66937858e-02,
     -2.99302451e-02,  -2.51239240e-01,   1.17296547e-01,   1.62703469e-01,
     -1.77877381e-01,   2.05892712e-01,  -4.34909314e-01,  -3.67220901e-02,
     -1.17212988e-01,   1.96012817e-02,   2.06915960e-02,   3.42829898e-03,
     -4.12347972e-01,  -1.21252500e-02,   1.12188004e-01,   4.02212620e-01,
      5.67307174e-02,  -1.13363445e-01,   5.67273274e-02,   2.02369452e-01,
     -1.72869444e-01,  -8.97249877e-01,   2.91510314e-01,   2.26088673e-01,
      8.97367716e-01,   3.67233634e-01,  -8.74226451e-01,   9.96801257e-02,
      2.01347247e-01,   9.80678737e-01,   7.00953901e-02,   1.42729461e-01,
     -1.16421312e-01,  -1.18872322e-01,   2.79948115e-04,   1.50046259e-01,
      1.59663528e-01,  -2.16814071e-01,  -8.09478983e-02,  -1.79625914e-01,
     -9.61264297e-02,   1.11921124e-01,   1.49394467e-01,   2.67139935e+00,
     -9.69958603e-01,   7.32920885e-01,   2.72537589e+00,   6.57512546e-01,
      1.50535142e+00,  -2.15233669e-01,   1.60330284e+00,  -3.88408363e-01, 1.21354020e+00 ]
    ]).astype("float64")

  #import render_smpl
  thetas  = Thetas[:,  :69]
  betas   = Thetas[:,70:80]#69:79]    # NOTE: these 2 are A) virtually indistinguishable.  So even with a real fat guy, the betas don't work.  If there's ANY way we can get it to work, I think we'll have to code it by hand.  We have to MAKE DAMN SURE we understand every last ***king detail of what's going on
  print("betas.shape is ",betas.shape)
  return betas
#================================ end Jonah_Hill() =================================


