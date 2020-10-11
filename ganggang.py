#!/usr/bin/env python3
import argparse, json, os
import torch
from utils import Logger
import data as data_module
import net as net_module
import pyaudio
from train import Trainer

from eval import ClassificationEvaluator, AudioInference

import wave
import numpy as np
import scipy.signal as sp
from scipy import interpolate
import math
import statistics as st
import time
import serial 

CHUNK = 2048
#CHANNELS = 1
RATE = 44100 #sample rate
#RECORD_SECONDS = 3
#WAVE_OUTPUT_FILENAME = "sound.wav"
interpFactor=8
b=sp.firwin((2*interpFactor*8-1), 1/interpFactor)
groupDelay = 63.5
sig=np.zeros((CHUNK,2))
buffer=np.zeros((3,1))
T=1/RATE
Niter = CHUNK//CHUNK
#p = pyaudio.PyAudio()
ser=serial.Serial('/dev/ttyUSB0',9600)

def DelayToAngle(delayInSamples, fs, pairSeparation):
    c = 340

    delayInSeconds = delayInSamples / fs

    sinOfAngle = delayInSeconds * c / pairSeparation
    if(abs(sinOfAngle) > 1):
        angleInRadians = (np.sign(sinOfAngle)) * np.pi/2 +np.pi/2
    else:
        angleInRadians = math.acos(sinOfAngle)
    return angleInRadians

def nextpow2(N):
    A=(int(math.log(N*2-1,2)))
    if A==math.log(2*N-1,2):
        A=2**(int(math.log(N*2-1,2)))
    else:
        A=2**(int(math.log(2*N-1,2)))*2
    return A

def callback(in_data, frame_count, time_info, flag):
    global data, recording, ch1, ch2, angle, state
    data = np.fromstring(in_data, dtype=np.float32)
    ch1=data[0::2] 
    ch2=data[1::2]
    if (max(data)>0.350): #0.997 #0.839 #500
        yy1= np.zeros(CHUNK, dtype=np.float32)
        yy2= np.zeros(CHUNK, dtype=np.float32)
        
        yy1=ch1
        yy2=ch2
        
        sig[:,0]=np.array(yy1[range(0,CHUNK)])
        sig[:,1]=np.array(yy2[range(0,CHUNK)])
    
        FFTsig1=np.fft.fft(sig[:,0],A)
        FFTsig2=np.fft.fft(sig[:,1],A)
    
    
        R1=FFTsig2*np.conj(FFTsig1)
            
        corr1=np.fft.fftshift(np.fft.ifft(R1/abs(R1)))
        
        interCo1=interpolate.interp1d(range(0,2*CHUNK),corr1)   
        
        interpolationRange=np.linspace(0,2*CHUNK-1,2*CHUNK*interpFactor)
        
        interpCorr1=sp.lfilter(b,1,interCo1(interpolationRange))
     
        indloc1=np.argmax(interpCorr1)
                
        tau=(indloc1-groupDelay)/interpFactor-CHUNK
        
        MicDis1 = 0.058 #0.058
        
        angle=DelayToAngle(tau, RATE, MicDis1)*180/np.pi
    
        state=1
    return (in_data, pyaudio.paContinue)


def infer_main(file_path, config, checkpoint):
    # Fix bugs
    if checkpoint is None:
        model = getattr(net_module, config['model']['type'])()
    else:
        m_name, sd, classes = _get_model_att(checkpoint)
        model = getattr(net_module, m_name)(classes, config, state_dict=sd)
        model.load_state_dict(checkpoint['state_dict'])

    tsf = _get_transform(config, 'val')
    inference = AudioInference(model, transforms=tsf)
    label, conf = inference.infer(file_path)
    

    #print(label, conf)
    if label == 5 or label == 0 or label == 6: # engine, aircondi, gun
        print('Unknown')
        return 181

    elif label == 2 or label == 9: #children, street
        print('Person')
        return 186


    elif label == 4 or label == 7: # drilling, jackhammer
        print('Construction')
        return 182

    elif label == 1: # car_horn, dog_bark, siren
        print('Vehicle')
        return 183

    elif label == 3: # dog
	    print('Dog')
	    return 184

    elif label == 8: #
	    print('Siren')
	    return 185

def _get_transform(config, name):
    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    return getattr(data_module, tsf_name)(name, tsf_args)

def _get_model_att(checkpoint):
    m_name = checkpoint['config']['model']['type']
    sd = checkpoint['state_dict']
    classes = checkpoint['classes']
    return m_name, sd, classes



while True:

    if __name__ == '__main__':
        argparser = argparse.ArgumentParser(description='PyTorch Template')

        argparser.add_argument('action', type=str,
                           help='what action to take (train, test, eval)')
    
        argparser.add_argument('-c', '--config', default=None, type=str,
                           help='p.terminate()config file path (default: None)')
        argparser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
        argparser.add_argument('--net_mode', default='init', type=str,
                           help='type of transfer learning to use')

        argparser.add_argument('--cfg', default=None, type=str,
                           help='nn layer config file')

        args = argparser.parse_args()


        # Resolve config vs. resume
        checkpoint = None
        if args.config:
            config = json.load(open(args.config))
            config['net_mode'] = args.net_mode
            config['cfg'] = args.cfg
        elif args.resume:
            checkpoint = torch.load(args.resume)
            config = checkpoint['config']

        else:
            raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
        # Pick mode to run


        if args.action == 'infer':
           file_path = args.action #file_path = saved_cv
           label = infer_main(file_path, config, checkpoint)
           ser.write(bytes([int(label)]))
           p2 = pyaudio.PyAudio()
           A=nextpow2(CHUNK)
           state = 1
           angle = -1


           stream2 = p2.open(format=pyaudio.paFloat32,
                            channels=2,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=callback,
                            input_device_index=1)

           stream2.start_stream()
           
           record = angle

           while stream2.is_active() :
                  
                 if angle != record :
                    ser.write(bytes([int(-1*(angle-180))]))
                    print([int(-1*(angle-180))])
                    break

                 if state == 1:
                    #ser.write(bytes([int(-1*(angle-180))]))
                    record = angle
                 state = 0
   
           stream2.stop_stream()
           stream2.close()
           #p.terminate()

