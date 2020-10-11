import sounddevice as sd
import numpy as np

import torch
from utils.util import load_image, load_audio
from PIL import ImageDraw, ImageFont

import pyaudio
import pylab
import time
import sys

import os
from net import MelspectrogramStretch
from utils import plot_heatmap

import pyaudio
import numpy as np
import pylab
import time
import sys
import matplotlib.pyplot as plt

RATE = 44100
CHUNK = 2048 # RATE / number of updates per second
DURATION = 3
LENGTH = RATE * DURATION

class AudioInference:

    def __init__(self, model, transforms):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms

        self.mel = MelspectrogramStretch(norm='db')
        self.mel.eval()


    def infer(self, path):
        p=pyaudio.PyAudio()
        stream=p.open(format=pyaudio.paFloat32,
                      channels=1,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK,
		      #stream_callback=callback,
                      input_device_index=1)        
        t1=time.time()
        data2 = np.fromstring(stream.read(LENGTH),dtype=np.float32)
        #stream.start_stream()
        #print(data2)
        #print(data2.shape)
        stream.stop_stream()
        stream.close()
        #p.terminate()
        #data = load_audio(path)
        data, rate = data2, 44100
        #print(data.shape) #(array(....), 44100) == (audio, sr)
        #sig_t, sr, _ = self.transforms.apply(data, None)
        sig_t, sr = self.transforms.apply(data, rate)

        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict( data )


        return label, conf

