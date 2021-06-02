import numpy as np
import os
import time
import copy

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import IPython.display as ipd
# import pyaudio as pa
import librosa
import nemo
import nemo.collections.asr as nemo_asr

# sample rate, Hz
SAMPLE_RATE = 16000

vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
    "vad_marblenet"
)
# Preserve a copy of the full config
cfg = copy.deepcopy(vad_model._cfg)
# print(OmegaConf.to_yaml(cfg))
