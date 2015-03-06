import numpy as np
from PIL import Image

def zero_pad(im, bx, by):
    x, y = im.shape
    
    for i in range (0, bx):
        z = 0
        #Insert row of zero to first row
        im = np.insert(im, 0, z, axis = 0)
        #Insert row of zero to last row
        im = np.insert(im, x+1+i, z, axis = 0)

    for i in range (0, by):
        z = 0
        #Insert column of zero to first column
        im = np.insert(im, 0, z, axis=1)
        #Insert column of zero to last column
        im = np.insert(im, y+1+i, z, axis=1)

    return im
