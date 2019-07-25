# -*- coding: utf-8 -*-
"""
Created on Mon May 28 02:51:29 2018

@author: Grenceng
"""

from Tkinter import *
import tkFileDialog
import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(
        os.path.realpath(__file__)),"lib/"))
import file_manager as fm
import mfcc
import backpropagation as bp
import pyaudio
import wave
import scipy.io.wavfile as wavfile
from sklearn.cluster import KMeans


#get models
(h1_gold, h1_silver, h1_bronze,
h2_gold, h2_silver, h2_bronze,
w_gold, w_silver, w_bronze) = fm.read_weight()

class Main:
    def __init__(self,parent,title):
        self.parent = parent
        self.parent.title(title)
        self.parent.config(background="#316068")
        self.komponen()
    def komponen(self):
        self.input_wav = Frame(self.parent,bg="#316068")
        self.input_wav.grid(row=0,column=0,sticky=NW,padx=10,pady=10)
        label = Label(self.input_wav, width=40,height=2,
                      bg="black",fg="white",text="import file.wav")
        label.grid(row=0,column=0,padx=10,pady=10,sticky=N,columnspan=3)
        self.btnBrowse = Button(self.input_wav, text='Browse File',
                                command=self.ambilWav,
                                width=30,height=2,bg="#a1dbcd")
        self.btnBrowse.grid(row=1,column=0,sticky=N,pady=10,padx=10,
                            columnspan=3)
        
        self.output = Frame(self.parent, bg="#316068")
        self.output.grid(row=0,column=1,sticky=NW,padx=10,pady=10)
        label1 = Label(self.output, width=40,height=2,
                      bg="black",fg="white",text="dikenali sebagai")
        label1.grid(row=0,column=0,padx=10,pady=10,sticky=N,columnspan=4)
        self.r = StringVar()
        result = Entry(self.output, bd=7, width=40, relief=FLAT,
                       textvariable=self.r)
        result.grid(row=1,column=0,padx=10,pady=10,columnspan=4)
        self.r.set("")
        male_icon = Label(self.output, width=5, height=3, bg="black",
                          fg="white",text="L")
        male_icon.grid(row=2,column=0,padx=3,pady=10)
        self.r_male = StringVar()
        result_male = Entry(self.output, bd=7, width=8, relief=FLAT,
                            textvariable=self.r_male)
        result_male.grid(row=2,column=1,padx=5,pady=10)
        self.r_male.set("")
        female_icon = Label(self.output, width=5, height=3, bg="black",
                            fg="white",text="P")
        female_icon.grid(row=2,column=2,padx=3,pady=10)
        self.r_female = StringVar()
        result_female = Entry(self.output, bd=7, width=8, relief=FLAT,
                            textvariable=self.r_female)
        result_female.grid(row=2,column=3,padx=5,pady=10)
        self.r_female.set("")
        
        
    def ambilWav(self):
        path = tkFileDialog.askopenfilename()
        wf = wave.open(path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(
            format = p.get_format_from_width(wf.getsampwidth()),
            channels = wf.getnchannels(),
            rate = wf.getframerate(),
            output = True
        )
        self.sr, self.data = wavfile.read(path)
        x = wf.readframes(1024)
        while x != '':
            stream.write(x)
            x = wf.readframes(1024)
        features = self.extraksi()
        identified = self.identifikasi(features)
        if round(identified.mean()) == 1:
            self.r.set("Laki-laki")
        else:
            self.r.set("Perempuan")
        
        self.r_male.set("%s" % (identified.mean()*100))
        self.r_female.set("%s" % (100-identified.mean()*100))
        
    
    def extraksi(self):
        vector   = mfcc.get_MFCC(self.sr,self.data)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(vector)
        features = kmeans.cluster_centers_
        return features
    
    def identifikasi(self, features):
        outputs = np.asarray(())
        #gold layer
        x = features[:,:5]
        layer1 = bp.activation(np.dot(x,h1_gold))
        layer2 = bp.activation(np.dot(layer1,h2_gold))
        output = bp.activation(np.dot(layer2,w_gold))
        outputs = np.hstack((outputs,output.mean()))
        #silver layer
        x = features[:,5:9]
        layer1 = bp.activation(np.dot(x,h1_silver))
        layer2 = bp.activation(np.dot(layer1,h2_silver))
        output = bp.activation(np.dot(layer2,w_silver))
        outputs = np.hstack((outputs,output.mean()))
        #bronze layer
        x = features[:,9:]
        layer1 = bp.activation(np.dot(x,h1_bronze))
        layer2 = bp.activation(np.dot(layer1,h2_bronze))
        output = bp.activation(np.dot(layer2,w_bronze))
        outputs = np.hstack((outputs,output.mean()))
        return outputs

root = Tk()
Main(root,".:: Voice Gender Clasificator ::.")
root.mainloop()