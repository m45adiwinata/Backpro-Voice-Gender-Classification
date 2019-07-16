# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:51:51 2018

@author: Grenceng
"""
import numpy as np
def create_hidden_layer(neuron1,neuron2,tipe):
    if tipe == "gold":
        hlayer = 2*np.random.rand(5,neuron1)-1
        hlayer2 = 2*np.random.rand(neuron1,neuron2)-1
        w = 2*np.random.rand(neuron2,1)-1
        return hlayer,hlayer2,w
    elif tipe == "silver" or "bronze":
        hlayer = 2*np.random.rand(4,neuron1)-1
        hlayer2 = 2*np.random.rand(neuron1,neuron2)-1
        w = 2*np.random.rand(neuron2,1)-1
        return hlayer,hlayer2,w

def activation(x,deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))