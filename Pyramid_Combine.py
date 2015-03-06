# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:08:16 2015
A Gaussian pyramid is basically a series of increasingly decimated images,
traditionally at downsampling rate r=2. At each level, the image is first blurred by convolving with a Gaussian-like filter to prevent aliasing
in the downsampled image. We then move up a level in the Gaussian pyramid by downsampling the image (halving each dimension).
To build the Laplacian pyramid, we take each level of the Gaussian pyramid and
subtract from it the next level interpolated to the same size.
@author: bxiao from http://pauljxtan.com/blog/011315/
"""
import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
from zero_pad import zero_pad
from scipy import delete

# create a Binomial (5-tap) filter
kernel = (1.0/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])

#image to be cut out
image1 = misc.imread('image1.jpg',flatten=0)
#image to be pasted onto
image2 = misc.imread('image2.jpg',flatten=0)
#mask
mask = misc.imread('mask.jpg',flatten=1)
    
#GR = gaussian_pyramids(mask)
#LA = laplacian_pyramids(image1)
#LB = laplacian_pyramids(image2)

def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    #return sig.convolve2d(image_up, 4*kernel, 'same')
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')

def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    """
    # Blur
    #image_blur = sig.convolve2d(image, kernel, 'same')
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]

####PYRAMIDS#####
def gaussian_pyramids(image):
    """
    Constructs Gaussian pyramids.
    Parameters :
    image : the original image (i.e. base of the pyramid)
    Returns :
    G : the Gaussian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
    return G[:-1]

def laplacian_pyramids(image):
    """
    Constructs Laplacian pyramids.
    Parameters :
    image : the original image (i.e. base of the pyramid)
    Returns :
    L : the Laplacian pyramid
    """
    # Initialize pyramids
    L = []
    G = [image, ]
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))
    return L

###Blend the Laplacian of the two images and the mask
#LS(i, j) = GR(i, j)*LA(i, j)+(1-GR(i,j))*LB(i, j)
def blend(LA, LB, GR):
    k = len(GR)
    blended_pyramid = []
    for i in range (0, k):
        #for each level in the pyramid (i)
        p1 = GR[i]*LA[i]
        p2 = (255- GR[i])*LB[i]
        p3 = p1+p2
        blended_pyramid.append(p3)
    return blended_pyramid

##Display all of the pyramid images together
def display(img1, image):  
    rows, cols = img1.shape
    composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
    composite_image[:rows, :cols] = image[0]

    i_row = 0
    for p in image[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
        
    fig, ax = plt.subplots()
    ax.imshow(composite_image,cmap='gray')
    plt.show()
    return

##Combine the pyramid images into one image
def collapse(blended):
  #Count down to 0
  for i in range(len(blended)-1,0,-1):
    #undo decimate, up size by two  
    end = interpolate(blended[i])
    end2 = blended[i-1]
    #now same size so add
    total = end + end2
    #delete last two items (end, end2)
    blended.pop()
    blended.pop()
    blended.append(total)
    output = total
  return output

#Combine all three colors
def combine(red, green, blue):
    output = np.ndarray((len(red),len(red[0]),3), dtype = float)

    for i in range (0,len(collapsed_red)):
        for j in range(0,len(red[0])):
            output[i][j][0] = red[i][j]
            output[i][j][1] = green[i][j]
            output[i][j][2] = blue[i][j]

    output = misc.bytescale(output)
    return output

def power_2(image):
    x, y = image.shape
    #check if the width and height are a power of two
    x2 = (x != 0 and ((x & (x - 1)) == 0))
    y2 = (y != 0 and ((y & (y - 1)) == 0))
    #if odd, pad by one so now even
    if x%2 != 0:
        image = np.insert(image, 0, 0, axis = 0)
    if y%2 != 0:
        image = np.insert(image, 0, 0, axis = 1)
    #if not, pad by one and repeat
    while (x2 == False):
           image = zero_pad(image, 1, 0)
           x, y = image.shape
           x2 = (x != 0 and ((x & (x - 1)) == 0))
    while (y2 == False):
           image = zero_pad(image, 0, 1)
           x, y = image.shape
           y2 = (y != 0 and ((y & (y - 1)) == 0))
    return image

#Sepereate each color
red = image1[:,:,0]
green = image1[:,:,1]
blue = image1[:,:,2]

red2 = image2[:,:,0]
green2 = image2[:,:,1]
blue2 = image2[:,:,2]

originalx, originaly = red.shape
red = power_2(red)
green = power_2(green)
blue = power_2(blue)
print blue.shape
red2 = power_2(red2)
green2 = power_2(green2)
blue2 = power_2(blue2)
print blue2.shape
print 2
mask = power_2(mask) 
GR = gaussian_pyramids(mask)

#Blend red
LA_red = laplacian_pyramids(red)
LB_red = laplacian_pyramids(red2)
blended_red = blend(LA_red, LB_red, GR)

#Blend green
LA_green = laplacian_pyramids(green)
LB_green = laplacian_pyramids(green2)
blended_green = blend(LA_green, LB_green, GR)

#Blend blue
LA_blue = laplacian_pyramids(blue)
LB_blue = laplacian_pyramids(blue2)
blended_blue = blend(LA_blue, LB_blue, GR)

#Collapse each color into single image
collapsed_red = collapse(blended_red)
collapsed_green = collapse(blended_green)
collapsed_blue = collapse(blended_blue)

final = combine(collapsed_red, collapsed_green, collapsed_blue)

#Remove border
def crop(cropped, orignalx, originaly):
    newx, newy, z = final.shape
    bx =(newx - originalx)/2
    by = (newy - originaly)/2

    removex = list(xrange(bx))
    cropped = delete(cropped, removex, 0)
    removey = list(xrange(by))
    cropped = delete(cropped, removey, 1)
    removex = list(xrange(newx-2*bx, newx-bx))
    removey = list(xrange(newy-2*by, newy-by))
    cropped = delete(cropped, removex, 0)
    cropped = delete(cropped, removey, 1)
    return cropped

final = crop(final, originalx, originaly)
plt.imshow(final)
plt.show()

