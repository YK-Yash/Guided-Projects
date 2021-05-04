# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:40:19 2021

@author: Yash Kumar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib qt

# Reading multiple images from a folder
folder = "./YALE"
images = []
for file in os.listdir(folder):
    img = mpimg.imread(os.path.join(folder, file))
    if img is not None:
        images.append(img)
        
# Vectorizing the images and storing it in a list
image_vector = []
for image in images:
    row,col = image.shape
    img_vec = image.reshape(row*col)
    img_vec_norm = img_vec / np.linalg.norm(img_vec)  # Making it a unit vector
    image_vector.append(img_vec_norm)
    
def genRandomHashVectors(m, length):  # Generate random unit vectors for Hashing
    hash_vector = []
    for i in range(m):
        v = np.random.uniform(-1,1,length)
        vcap = v / np.linalg.norm(v)
        hash_vector.append(vcap)
    return hash_vector 

def localSensitiveHashing(hash_vector ,data): 
    hash_code = []
    for i in range(len(hash_vector)):
        if np.dot(data,hash_vector[i]) > 0:
            hash_code.append('1')
        else:
            hash_code.append('0')
    return hash_code

hash_vector = genRandomHashVectors(20,len(image_vector[0]))

localSensitiveHashing(hash_vector,image_vector[0])

# Creating a Image Dictionary using the hash as the keys
image_dict = {}
for i in range(len(image_vector)):
    hash_code = localSensitiveHashing(hash_vector,image_vector[i])
    str_hash_code = ''.join(hash_code)
    if str_hash_code not in image_dict.keys():
        image_dict[str_hash_code] = [i]
    else:
        image_dict[str_hash_code].append(i)  
        
col_names = ["Hash_Codes","Image_Index"]
df = pd.DataFrame(list(image_dict.items()),columns=col_names)
df.head(30)

# Getting the keys and values of the Dictionary
keys = list(image_dict.keys())
values = list(image_dict.values())

# Plotting images with same hash code
imgs = [images[i] for i in range(len(images)) if i in values[2]]
fig = plt.figure()
cols = 2
n_images = len(imgs)
for n,image in zip(range(n_images),imgs):
    ax = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
    plt.gray()
    plt.imshow(image)
fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
plt.show()