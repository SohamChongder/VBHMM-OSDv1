#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Authors: Lukas Burget, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
# modified by  Somil Jain  [github: coderatwork7, somiljain71100@gmail.com]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import onnxruntime
import soundfile as sf
import torch.backends
import features
from models.resnet import *
import argparse
import os
import itertools
import fastcluster
import h5py
import kaldi_io
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh
from diarization_lib import read_xvector_timing_dict, l2_norm,cos_similarity, twoGMMcalib_lin, merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VBx import VBx
torch.backends.cudnn.enabled = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import os
# from overlap_utils import *
from PyannoteOSDv1 import *
from generate_embeddings import generate_embeddings
from vbhmm import vbhmm_resegmentation
from PyannoteVAD import pyannote_vad






#entire diarization pipeline for a single file
def pipeline_single_file(wav_path):
    pyannote_segementation_token="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"
    if len(wav_path) >= 0 and os.path.exists(wav_path):
        full_name = os.path.basename(wav_path)
        filename = os.path.splitext(full_name)[0]
        print(filename)
    else:
        raise ValueError('Wrong path parameters provided (or not provided at all)')
    vad_path = f"exp/lab/{filename}.lab"
    pyannote_vad(pyannote_segementation_token, vad_path, wav_path)
    config = {
        "ndim": 64,
        "embed_dim": 256,
        "seg_len": 144,
        "seg_jump": 24,
        "in_file_list": "exp/list.txt",
        "out_ark_fn": f"exp/ark/{filename}.ark",
        "out_seg_fn": f"exp/seg/{filename}.seg",
        "weights": "models/ResNet101_16kHz/nnet/final.onnx",
        "backend": "onnx",
        "init": "AHC+VB",
        "out_rttm_dir": "exp/rttm",
        "xvec_ark_file": f"exp/ark/{filename}.ark",
        "segments_file": f"exp/seg/{filename}.seg",
        "xvec_transform": "models/ResNet101_16kHz/transform.h5",
        "plda_file": "models/ResNet101_16kHz/plda",
        "threshold": -0.015,
        "lda_dim": 128,
        "Fa": 0.3,
        "Fb": 17,
        "loopP": 0.99,
        "target_energy": 1.0,
        "init_smoothing": 5.0,
    }
    
    generate_embeddings(args, wav_path, vad_path, config)
    vbhmm_resegmentation(filename, config)
    postprocessing_osd_single_file(wav_path, "exp/rttm", "exp/overlap", "exp/rttm_osd")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--in-wav-path', type=str, default='/Users/sohamchongder/Desktop/Soham/third_dihard_challenge_eval/data/wav', help='input file')
    args = parser.parse_args()
    

    #Create directory structure
    os.makedirs("exp", exist_ok=True)
    os.makedirs("exp/lab", exist_ok=True)
    os.makedirs("exp/ark", exist_ok=True)
    os.makedirs("exp/seg", exist_ok=True)
    os.makedirs("exp/rttm", exist_ok=True)
    os.makedirs("exp/overlap", exist_ok=True)
    os.makedirs("exp/rttm_osd", exist_ok=True)

    wav_path=args.in_wav_path

    #Check if file or directory
    if os.path.isdir(wav_path):
        for root, dirs, files in os.walk(wav_path):
            for file in files:
                if file.endswith(".wav"):
                    pipeline_single_file(os.path.join(root, file))
    elif os.path.isfile(wav_path):
        pipeline_single_file(wav_path)
    else:
        raise ValueError('Provided path is neither a file nor a directory')
    
