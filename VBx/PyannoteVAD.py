import logging
import torch.backends
import os
torch.backends.cudnn.enabled = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

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
    print("VAD Completed")
