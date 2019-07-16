# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:49:48 2018

@author: Grenceng
"""

import python_speech_features as mfcc
from sklearn import preprocessing
def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features