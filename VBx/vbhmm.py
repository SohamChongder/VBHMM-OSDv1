import logging
import torch.backends
from models.resnet import *
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
import os

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
        # print(file_name)
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
        print("Initial Diarization completed")    
