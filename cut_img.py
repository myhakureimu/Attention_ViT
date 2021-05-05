#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:41:20 2021

@author: lzq
"""
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('monkey.jpg')

img = np.concatenate([img[:,-500:],img,255*np.ones_like(img[:,-500:])],1)

#img = np.concatenate([img[:,-300:],img[:,:-300]],1)

plt.figure()
plt.imshow(img)

plt.imsave('monkey2_img.jpg', img)