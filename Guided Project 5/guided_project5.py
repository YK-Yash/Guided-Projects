# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:15:32 2021

@author: Yash
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt  

#Conversion to grayscale img
image = cv2.imread('Figo.jpg')
image = cv2.imread('fruit_basket.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Canny edge detection
edges = cv2.Canny(gray,100,200)
rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
rgb *= np.array((1,0,0),np.uint8) # set g and b to 0, leaves red
out = np.bitwise_or(image, rgb)
plt.subplot(131),plt.imshow(gray,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(out,cmap = 'gray')
plt.title('Edge Image - color'), plt.xticks([]), plt.yticks([])
plt.show()

#Simple thresholding
ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
plt.subplot(1,1,1),plt.imshow(thresh1,'gray',vmin=0,vmax=255)
plt.title("Simple Thresholding")
plt.xticks([]),plt.yticks([])
plt.show()

# Otsu's thresholding
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.subplot(1,1,1),plt.imshow(th2,'gray',vmin=0,vmax=255)
plt.title("Otsu's Thresholding")
plt.xticks([]),plt.yticks([])
plt.show()