#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:38:45 2021

@author: lzq
"""
import torch as tc
import numpy as np
def ig(model, image, label, num_step, BS=1):
    
    image = image.repeat(num_step,1,1,1)
    steps = (tc.arange(num_step)+0.5).reshape([-1,1,1,1]).cuda()
    image = image*steps
    model.cuda()
    grads = []
    for i in range(int(np.ceil(num_step/BS))):
        image_batch = image[i*BS:(i+1)*BS].clone().detach().cuda()
        image_batch.requires_grad = True
        output = model(image_batch)
        tc.sum(output[:,label]).backward()
        grads.append(image_batch.grad)
    grads = tc.cat(grads, dim=0)
    return grads