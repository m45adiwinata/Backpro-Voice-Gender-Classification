# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:40:38 2018

@author: Grenceng
"""
import numpy as np
import os

dest     = "pygender/training model/"
def write_weight(
        h1_gold,h1_silver,h1_bronze,
        h2_gold,h2_silver,h2_bronze,
        w_gold,w_silver,w_bronze):
    file = open(dest+"hidden_1_gold.txt","w")
    for f in h1_gold:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    file = open(dest+"hidden_1_silver.txt","w")
    for f in h1_silver:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    file = open(dest+"hidden_1_bronze.txt","w")
    for f in h1_bronze:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    
    file = open(dest+"hidden_2_gold.txt","w")
    for f in h2_gold:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    file = open(dest+"hidden_2_silver.txt","w")
    for f in h2_silver:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    file = open(dest+"hidden_2_bronze.txt","w")
    for f in h2_bronze:
        for g in f:
            file.write("%s " % g)
        file.write("\n")
    file.close()
    
    file = open(dest+"output_w_gold.txt","w")
    for f in w_gold:
        file.write("%s " % f)
        file.write("\n")
    file.close()
    file = open(dest+"output_w_silver.txt","w")
    for f in w_silver:
        file.write("%s " % f)
        file.write("\n")
    file.close()
    file = open(dest+"output_w_bronze.txt","w")
    for f in w_bronze:
        file.write("%s " % f)
        file.write("\n")
    file.close()

def read_weight():
    #path to saved models
    modelpath  = "pygender/training model"
    
    #get models
    
    files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.txt')]
    for i in range(len(files)):
        files[i] = files[i].split("\\")[1]
    
    #hidden 1
    file = open(modelpath+"/"+files[0],"r")
    h1_bronze = file.read().split("\n")
    del h1_bronze[-1]
    for i in range(len(h1_bronze)):
        h1_bronze[i] = h1_bronze[i].split()
    h1_bronze = np.array((h1_bronze), dtype='Float64')
    
    file = open(modelpath+"/"+files[1],"r")
    h1_gold = file.read().split("\n")
    del h1_gold[-1]
    for i in range(len(h1_gold)):
        h1_gold[i] = h1_gold[i].split()
    h1_gold = np.array((h1_gold), dtype='Float64')
    
    file = open(modelpath+"/"+files[2],"r")
    h1_silver = file.read().split("\n")
    del h1_silver[-1]
    for i in range(len(h1_silver)):
        h1_silver[i] = h1_silver[i].split()
    h1_silver = np.array((h1_silver), dtype='Float64')
    
    #hidden 2
    file = open(modelpath+"/"+files[3],"r")
    h2_bronze = file.read().split("\n")
    del h2_bronze[-1]
    for i in range(len(h2_bronze)):
        h2_bronze[i] = h2_bronze[i].split()
    h2_bronze = np.array((h2_bronze), dtype='Float64')
    
    file = open(modelpath+"/"+files[4],"r")
    h2_gold = file.read().split("\n")
    del h2_gold[-1]
    for i in range(len(h2_gold)):
        h2_gold[i] = h2_gold[i].split()
    h2_gold = np.array((h2_gold), dtype='Float64')
    
    file = open(modelpath+"/"+files[5],"r")
    h2_silver = file.read().split("\n")
    del h2_silver[-1]
    for i in range(len(h2_silver)):
        h2_silver[i] = h2_silver[i].split()
    h2_silver = np.array((h2_silver), dtype='Float64')
    
    #output weight
    file = open(modelpath+"/"+files[6],"r")
    w_bronze = file.read().split("\n")
    del w_bronze[-1]
    for i in range(len(w_bronze)):
        w_bronze[i] = w_bronze[i].split("[")[1].split("]")[0]
    w_bronze = np.array(w_bronze, dtype='Float64')
    
    
    file = open(modelpath+"/"+files[7],"r")
    w_gold = file.read().split("\n")
    del w_gold[-1]
    for i in range(len(w_gold)):
        w_gold[i] = w_gold[i].split("[")[1].split("]")[0]
    w_gold = np.array((w_gold), dtype='Float64')
    
    file = open(modelpath+"/"+files[8],"r")
    w_silver = file.read().split("\n")
    del w_silver[-1]
    for i in range(len(w_silver)):
        w_silver[i] = w_silver[i].split("[")[1].split("]")[0]
    w_silver = np.array((w_silver), dtype='Float64')
    
    file.close()
    
    return (h1_gold, h1_silver, h1_bronze,
            h2_gold, h2_silver, h2_bronze,
            w_gold, w_silver, w_bronze)