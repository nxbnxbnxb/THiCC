import matplotlib.pyplot as plt
import numpy as np

#######################################################################################################
################################# visualization functions #############################################
#######################################################################################################
#=========================================================================
def cross_sections_biggest(m):
#=========================================================================
    '''
        @precondition m is a cube (a 3-d np.array)
    '''
    # NOTE:  
    # NOTE:  this isn't working right.  not sure why.  Dec. 12, 2018
    # NOTE:  
    i_3=i_2=i_1=0
    max_vol_3=max_vol_2=max_vol_1=0
    for i in range(m.shape[0]):
        vol_1=np.count_nonzero(m[i,:,:])
        if vol_1 > max_vol_1: max_vol_1 = vol_1; i_1=i
        vol_2=np.count_nonzero(m[:,i,:])
        if vol_2 > max_vol_2: max_vol_2 = vol_2; i_2=i
        vol_3=np.count_nonzero(m[:,:,i])
        if vol_3 > max_vol_3: max_vol_3 = vol_3; i_3=i
    pltshow(m[i_1,:,:])
    pltshow(m[:,i_2,:])
    pltshow(m[:,:,i_3])
    return
#=========================================================================


#######################################################################################################
#=========================================================================
def pltshow(x):
    plt.imshow(x); plt.show(); plt.close()
#=========================================================================

#######################################################################################################
#=========================================================================
def show_cross_sections(model_3d, axis='z'):
    if axis.lower()=='z':
        for i in range(model_3d.shape[2]):
            if np.any(model_3d[:,:,i]):
                pltshow(model_3d[:,:,i])
                print 'height is {0}'.format(i)
    elif axis.lower()=='y':
        for i in range(model_3d.shape[1]):
            if np.any(model_3d[:,i,:]):
                pltshow(model_3d[:,i,:])
                print 'loc is {0}'.format(i)
    elif axis.lower()=='x':
        for i in range(model_3d.shape[0]):
            if np.any(model_3d[i,:,:]):
                pltshow(model_3d[i,:,:])
                print 'loc is {0}'.format(i)
    else:
        print "Usage: please input axis x, y, or z in format:\n\n  show_cross_sections([model_name], axis='z')"
    return
#=========================================================================














































































