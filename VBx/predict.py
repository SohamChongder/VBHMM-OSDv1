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
from overlap_utils import *


def pyannote_vad(token,path,wav_path):
    model = Model.from_pretrained(
    "pyannote/segmentation-3.0", 
    use_auth_token=token)
    assert os.path.exists(wav_path), f"wavfile Path does not exist: {wav_path}"
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(wav_path)
    with open(path,'w') as f:
        f.write(vad.to_lab())
    f.close()
    assert os.path.exists(path), f"Lab File processing didnt complete: {path}"


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info(f'Start: {self.name}: ')

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(f'End:   {self.name}: Elapsed: {time.time() - self.tstart} seconds')
        else:
            logger.info(f'End:   {self.name}: ')


def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def load_utt(ark, utt, position):
    with open(ark, 'rb') as f:
        f.seek(position - len(utt) - 1)
        ark_key = kaldi_io.read_key(f)
        assert ark_key == utt, f'Keys does not match: `{ark_key}` and `{utt}`.'
        mat = kaldi_io.read_mat(f)
        return mat


def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            f.write(f'{name}  [ {" ".join(str(x) for x in data_dict[name])} ]{os.linesep}')


def get_embedding(fea, model, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'pytorch':
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == 'onnx':
        return model.run([label_name],
                         {input_name: fea.astype(np.float32).transpose()
                         [np.newaxis, :, :]})[0].squeeze()

def predict(args,wav_path,vad_path,config):
    if len(wav_path)>=0 and os.path.exists(wav_path):
        full_name = os.path.basename(wav_path)
        filename = os.path.splitext(full_name)[0]
        print(filename)
    else:
        raise ValueError('Wrong path parameters provided (or not provided at all)')
    if not os.path.exists(config['weights']):
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')
    device = ''
    if args.gpus != '':
        logger.info(f'Using GPU: {args.gpus}')

        # gpu configuration
        initialize_gpus(args)
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    model, label_name, input_name = '', None, None

    if config['backend'] == 'onnx':
        model = onnxruntime.InferenceSession(config['weights'])
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

    else:
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')

    with torch.no_grad():
        with open(config['out_seg_fn'], 'w') as seg_file:
            with open(config['out_ark_fn'], 'wb') as ark_file:
                with Timer(f'Processing file {filename}'):
                    signal, samplerate = sf.read(wav_path)
                    labs = np.atleast_2d((np.loadtxt(vad_path,usecols=(0, 1)) * samplerate).astype(int))
                    if samplerate == 8000:
                        noverlap = 120
                        winlen = 200
                        window = features.povey_window(winlen)
                        fbank_mx = features.mel_fbank_mx(winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
                    elif samplerate == 16000:
                        noverlap = 240
                        winlen = 400
                        window = features.povey_window(winlen)
                        fbank_mx = features.mel_fbank_mx(winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
                    else:
                        raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')

                    LC = 150
                    RC = 149
                    np.random.seed(3)  # for reproducibility
                    signal = features.add_dither((signal*2**15).astype(int))
                    for segnum in range(len(labs)):
                        seg = signal[labs[segnum, 0]:labs[segnum, 1]]
                        if seg.shape[0] > 0.01*samplerate:  # process segment only if longer than 0.01s
                                # Mirror noverlap//2 initial and final samples
                            seg = np.r_[seg[noverlap // 2 - 1::-1],
                                        seg, seg[-1:-winlen // 2 - 1:-1]]
                            fea = features.fbank_htk(seg, window, noverlap, fbank_mx,
                                                         USEPOWER=True, ZMEANSOURCE=True)
                            fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)

                            slen = len(fea)
                            start = -config['seg_jump']

                            for start in range(0, slen - config['seg_len'], config['seg_jump']):
                                data = fea[start:start + config['seg_len']]
                                xvector = get_embedding(
                                data, model, label_name=label_name, input_name=input_name, backend=config['backend'])
                                key = f'{filename}_{segnum:04}-{start:08}-{(start + config['seg_len']):08}'
                                if np.isnan(xvector).any():
                                    logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                else:
                                    seg_start = round(labs[segnum, 0] / float(samplerate) + start / 100.0, 3)
                                    seg_end = round(
                                        labs[segnum, 0] / float(samplerate) + start / 100.0 + config['seg_len'] / 100.0, 3
                                    )
                                    seg_file.write(f'{key} {filename} {seg_start} {seg_end}{os.linesep}')
                                    kaldi_io.write_vec_flt(ark_file, xvector, key=key)

                            if slen - start - config['seg_jump'] >= 10:
                                data = fea[start + config['seg_jump']:slen]
                                xvector = get_embedding(
                                        data, model, label_name=label_name, input_name=input_name, backend=config['backend'])

                                key = f'{filename}_{segnum:04}-{(start + config['seg_jump']):08}-{slen:08}'

                                if np.isnan(xvector).any():
                                    logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                else:
                                    seg_start = round(
                                        labs[segnum, 0] / float(samplerate) + (start + config['seg_jump']) / 100.0, 3
                                    )
                                    seg_end = round(labs[segnum, 1] / float(samplerate), 3)
                                    seg_file.write(f'{key} {filename} {seg_start} {seg_end}{os.linesep}')
                                    kaldi_io.write_vec_flt(ark_file, xvector, key=key)
    print("Done")

def write_output(fp,file_name, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


def vbhmm_resegmentation(filename,config):
    assert 0 <= config['loopP'] <= 1, f'Expecting config loopP between 0 and 1, got {config['loopP']} instead.'
    segs_dict = read_xvector_timing_dict(config['segments_file'])
    kaldi_plda = read_plda(config['plda_file'])
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    # Open ark file with x-vectors and in each iteration of the following
    # for-loop read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(config['xvec_ark_file'])
    # group xvectors in ark by recording name
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(config['xvec_transform'], 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        if config['init'] == 'AHC' or config['init'].endswith('VB'):
            if config['init'].startswith('AHC'):
                # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                # similarities between all x-vectors)
                scr_mx = cos_similarity(x)
                # Figure out utterance specific args.config['threshold'] for AHC
                thr, _ = twoGMMcalib_lin(scr_mx.ravel())
                # output "labels" is an integer vector of speaker (cluster) ids
                scr_mx = squareform(-scr_mx, checks=False)
                lin_mat = fastcluster.linkage(
                    scr_mx, method='average', preserve_input='False')
                del scr_mx
                adjust = abs(lin_mat[:, 2].min())
                lin_mat[:, 2] += adjust
                labels1st = fcluster(lin_mat, -(thr + config['threshold']) + adjust,
                    criterion='distance') - 1
            if config['init'].endswith('VB'):
                # Smooth the hard labels obtained from AHC to soft assignments
                # of x-vectors to speakers
                qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                qinit[range(len(labels1st)), labels1st] = 1.0
                qinit = softmax(qinit * config['init_smoothing'], axis=1)
                fea = (x - plda_mu).dot(plda_tr.T)[:, :config['lda_dim']]
                q, sp, L = VBx(
                    fea, plda_psi[:config['lda_dim']],
                    pi=qinit.shape[1], gamma=qinit,
                    maxIters=40, epsilon=1e-6,
                    loopProb=config['loopP'], Fa=config['Fa'], Fb=config['Fb'])
                np.save("../example/gamma_"+file_name+".npy",q) #  timeframe * Speaker posterior probabilities 
                labels1st = np.argsort(-q, axis=1)[:, 0]
                if q.shape[1] > 1:
                    labels2nd = np.argsort(-q, axis=1)[:, 1]
        else:
            raise ValueError('Wrong option for args.initialization.')

        assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
        start, end = segs_dict[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)
        mkdir_p(config['out_rttm_dir'])
        with open(os.path.join(config['out_rttm_dir'], f'{file_name}.rttm'), 'w') as fp:
            write_output(fp,file_name, out_labels, starts, ends)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--in-wav-path', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    #parser.add_argument('--in-vad-path', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    args = parser.parse_args()
    wav_path=args.in_wav_path
    pyannote_segementation_token="ADD_YOURS_HERE"
    if len(wav_path)>=0 and os.path.exists(wav_path):
        full_name = os.path.basename(wav_path)
        filename = os.path.splitext(full_name)[0]
        print(filename)
    else:
        raise ValueError('Wrong path parameters provided (or not provided at all)')    
    vad_path=f"exp/{filename}.lab"
    pyannote_vad(pyannote_segementation_token,vad_path,wav_path)
    config = {
    "ndim": 64,
    "embed_dim": 256,
    "seg_len": 144,
    "seg_jump": 24,
    "in_file_list": "exp/list.txt",
    "out_ark_fn": f"exp/{filename}.ark",
    "out_seg_fn": f"exp/{filename}.seg",
    "weights": "VBx/models/ResNet101_16kHz/nnet/final.onnx",
    "backend": "onnx",
    "init": "AHC+VB",
    "out_rttm_dir": "exp",
    "xvec_ark_file": f"exp/{filename}.ark",
    "segments_file": f"exp/{filename}.seg",
    "xvec_transform": "VBx/models/ResNet101_16kHz/transform.h5",
    "plda_file": "VBx/models/ResNet101_16kHz/plda",
    "threshold": -0.015,
    "lda_dim": 128,
    "Fa": 0.3,
    "Fb": 17,
    "loopP": 0.99,
    "target_energy": 1.0,
    "init_smoothing": 5.0,
    }
    
    predict(args,wav_path,vad_path,config)
    vbhmm_resegmentation(filename,config)
    #meeting_id=filename.split(".")[0]
    #labels_matching(pyannote_segementation_token,"/Career/NTU/VBx/example/audios/16k","/Career/NTU/VBx/example/seg",meeting_id,"/Career/NTU/VBx/exp") 

