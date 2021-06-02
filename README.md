# Plume ASR

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> Generates text from audio containing speech
---

# Table of Contents

* [Prerequisites](#prerequisites)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

# Prerequisites
```bash
# apt install libsndfile-dev ffmpeg
```

# Features

* ASR using Jasper (from [NemoToolkit](https://github.com/NVIDIA/NeMo) )
* ASR using Wav2Vec2 (from [fairseq](https://github.com/pytorch/fairseq) )

# Installation
To install the packages and its dependencies run.
```bash
python setup.py install
```
or with pip
```bash
pip install .[all]
```

The installation should work on Python 3.6 or newer. Untested on Python 2.7

# Usage
### Library
> Jasper
```python
from plume.models.jasper_nemo.asr import JasperASR
asr_model = JasperASR("/path/to/model_config_yaml","/path/to/encoder_checkpoint","/path/to/decoder_checkpoint") # Loads the models
TEXT = asr_model.transcribe(wav_data) # Returns the text spoken in the wav
```
> Wav2Vec2
```python
from plume.models.wav2vec2.asr import Wav2Vec2ASR
asr_model = Wav2Vec2ASR("/path/to/ctc_checkpoint","/path/to/w2v_checkpoint","/path/to/target_dictionary") # Loads the models
TEXT = asr_model.transcribe(wav_data) # Returns the text spoken in the wav
```
### Command Line
```
$ plume
```
### Pretrained Models
**Jasper**
https://ngc.nvidia.com/catalog/models/nvidia:multidataset_jasper10x5dr/files?version=3
**Wav2Vec2**
https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md
