# -*- coding: utf-8 -*-
"""
Created on Mon May 28 02:12:25 2018

@author: Grenceng
"""

import os, sys
import numpy as np
from scipy.io.wavfile import read
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"lib/"))
import backpropagation as bp
import mfcc
import file_manager as fm
from sklearn.cluster import KMeans

gender = ["male","female"]
t = [1,0]
target = np.asarray(())
features = np.asarray(())
for i in range(2):
    source   = "pygender/train_data/youtube/"+gender[i]+"/"
    files    = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]
    
    for f in files:
        sr,audio = read(f)
        vector   = mfcc.get_MFCC(sr,audio)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(vector)
        if features.size == 0:
            features = kmeans.cluster_centers_
        else:
            features = np.vstack((features, kmeans.cluster_centers_))
    target = np.hstack((target, t[i]*np.ones(4*len(files))))

#features,target = randoming(features,target)
file = open("pygender/train_data/dataset.txt","w")
for j in range(features.shape[0]):
    for f in features[j]:
        file.write("%s " % f)
    file.write("\n")
file.close()        
file = open("pygender/train_data/target.txt","w")
for f in target:
    file.write("%s\n" % f)
file.close()
temp = np.zeros((len(target),1))
for i in range(len(target)):
    temp[i,0] = target[i]
target = temp
h1_gold, h2_gold, w_gold = bp.create_hidden_layer(4,3,"gold")
h1_silver, h2_silver, w_silver = bp.create_hidden_layer(4,3,"silver")
h1_bronze, h2_bronze, w_bronze = bp.create_hidden_layer(4,3,"bronze")

epoh = 0
while True:
    #Gold Layer
    x = features[:,:5]
    layer1 = bp.activation(np.dot(x,h1_gold))
    layer2 = bp.activation(np.dot(layer1,h2_gold))
    output = bp.activation(np.dot(layer2,w_gold))
    error = target - output
    error_g = np.mean(np.abs(error))
    dw = error * bp.activation(output, deriv=True)
    h2_error = dw.dot(w_gold.T)
    dh2 = h2_error * bp.activation(layer2, deriv=True)
    h1_error = dh2.dot(h2_gold.T)
    dh1 = h1_error * bp.activation(layer1, deriv=True)
    w_gold += layer2.T.dot(dw)
    h2_gold += layer1.T.dot(dh2)
    h1_gold += x.T.dot(dh1)
    
    #Silver Layer
    x = features[:,5:9]
    layer1 = bp.activation(np.dot(x,h1_silver))
    layer2 = bp.activation(np.dot(layer1,h2_silver))
    output = bp.activation(np.dot(layer2,w_silver))
    error = target - output
    error_s = np.mean(np.abs(error))
    dw = error * bp.activation(output, deriv=True)
    h2_error = dw.dot(w_silver.T)
    dh2 = h2_error * bp.activation(layer2, deriv=True)
    h1_error = dh2.dot(h2_silver.T)
    dh1 = h1_error * bp.activation(layer1, deriv=True)
    w_silver += layer2.T.dot(dw)
    h2_silver += layer1.T.dot(dh2)
    h1_silver += x.T.dot(dh1)
    
    #Bronze Layer
    x = features[:,9:]
    layer1 = bp.activation(np.dot(x,h1_bronze))
    layer2 = bp.activation(np.dot(layer1,h2_bronze))
    output = bp.activation(np.dot(layer2,w_bronze))
    error = target - output
    error_b = np.mean(np.abs(error))
    dw = error * bp.activation(output, deriv=True)
    h2_error = dw.dot(w_bronze.T)
    dh2 = h2_error * bp.activation(layer2, deriv=True)
    h1_error = dh2.dot(h2_bronze.T)
    dh1 = h1_error * bp.activation(layer1, deriv=True)
    w_bronze += layer2.T.dot(dw)
    h2_bronze += layer1.T.dot(dh2)
    h1_bronze += x.T.dot(dh1)
    if epoh % 333 == 0:
        print "Error :",error_g, error_s, error_b
    epoh += 1
    e = np.array((error_g,error_s,error_b))
    if e.mean() < 0.05:
        print "\nEpoh = ",epoh
        print "Error :",error_g, error_s, error_b
        break

fm.write_weight(
        h1_gold, h1_silver, h1_bronze,
        h2_gold, h2_silver, h2_bronze,
        w_gold, w_silver, w_bronze)