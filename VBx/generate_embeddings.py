import logging
import time
import onnxruntime
import soundfile as sf
import torch.backends
import features
from models.resnet import *
import os
import kaldi_io
import numpy as np
torch.backends.cudnn.enabled = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
import os



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


# def load_utt(ark, utt, position):
#     with open(ark, 'rb') as f:
#         f.seek(position - len(utt) - 1)
#         ark_key = kaldi_io.read_key(f)
#         assert ark_key == utt, f'Keys does not match: `{ark_key}` and `{utt}`.'
#         mat = kaldi_io.read_mat(f)
#         return mat


# def write_txt_vectors(path, data_dict):
#     """ Write vectors file in text format.

#     Args:
#         path (str): path to txt file
#         data_dict: (Dict[np.array]): name to array mapping
#     """
#     with open(path, 'w') as f:
#         for name in sorted(data_dict):
#             f.write(f'{name}  [ {" ".join(str(x) for x in data_dict[name])} ]{os.linesep}')


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

def generate_embeddings(args,wav_path,vad_path,config):
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
    print("Embeddings generated successfully")