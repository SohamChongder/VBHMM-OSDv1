##################################################################################################################
# A script for VAD benchmarking 
################
# VBx VAD      #
################
# TDNN DIHARD3 #
################
# Silero VAD   #
################
# WEB RTC VAD  #
################
# Pyannote VAD #
################
##################################################################################################################
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import os
import argparse 
import subprocess
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pyannote.audio import Model



def silero_vad(input_wav):
	model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
		                      model='silero_vad',
		                      force_reload=True)
	(get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks) = utils
	#print(utils)
	sampling_rate = 16000 
	wav = read_audio(input_wav, sampling_rate=sampling_rate)
	nfile=input_wav.split("/")[-1].split(".")[0]	
	#print(input_wav.split("/")[-1].split(".")[0])
	speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
	#print("".join(speech_timestamps))
	with open("silero_vad"+nfile+".pkl",'wb') as f:	
		pickle.dump(speech_timestamps,f)

def VBxVAD(input_wav):
	#print(input_wav.split("/")[-1].split(".")[0])
	subprocess.run(["sh","/home1/somil/VBx-VAD/VB_VAD.sh",input_wav])
		
def pyaanote_vad(input_wav):
	model = Model.from_pretrained("pyannote/segmentation",use_auth_token="ADD+YOURs")
	pipeline = VoiceActivityDetection(segmentation=model)
	HYPER_PARAMETERS = {
	  # onset/offset activation thresholds
	  "onset": 0.5, "offset": 0.5,
	  # remove speech regions shorter than that many seconds.
	  "min_duration_on": 0.0,
	  # fill non-speech regions shorter than that many seconds.
	  "min_duration_off": 0.0
	}
	pipeline.instantiate(HYPER_PARAMETERS)
	vad = pipeline(input_wav)
	#print(vad)
	nfile=input_wav.split("/")[-1].split(".")[0]	
	with open("eval/pyannote_vad"+nfile+".pkl",'wb') as f:	
		pickle.dump(vad,f)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-audio', type=str, help="Input audio file")
	parser.add_argument('--in-VAD', type=str, help="Input vad TYPE: 1. VBx_VAD | 2. silero_VAD | 3. Pyannote_VAD ")
	args = parser.parse_args()		
	input_wav=args.in_audio
	vad_type=args.in_VAD
	
	if vad_type=="VBx_VAD":
		VBxVAD(input_wav)
	elif vad_type=="silero_VAD":
		silero_vad(input_wav)
	elif vad_type=="Pyannote_VAD":
		pyaanote_vad(input_wav)
	
	
main()
#python3 /home1/somil/VBx-VAD/VAD.py /home1/somil/language_diarization/Displace_data/eval_1 Pyannote_VAD

