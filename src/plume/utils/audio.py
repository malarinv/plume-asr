import sys
from io import BytesIO

from .lazy_import import lazy_module, lazy_callable

np = lazy_module("numpy")
pydub = lazy_module("pydub")
lfilter = lazy_callable("scipy.signal.lfilter")
butter = lazy_callable("scipy.signal.butter")
read = lazy_callable("scipy.io.wavfile.read")
write = lazy_callable("scipy.io.wavfile.write")
# from scipy.signal import lfilter, butter
# from scipy.io.wavfile import read, write
# import numpy as np


def audio_seg_to_wav_bytes(aud_seg):
    b = BytesIO()
    aud_seg.export(b, format="wav")
    return b.getvalue()


def audio_wav_bytes_to_seg(wav_bytes):
    b = BytesIO(wav_bytes)
    return pydub.AudioSegment.from_file(b)


def butter_params(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_params(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    fs, audio = read(sys.argv[1])
    import pdb

    pdb.set_trace()
    low_freq = 300.0
    high_freq = 4000.0
    filtered_signal = butter_bandpass_filter(
        audio, low_freq, high_freq, fs, order=6
    )
    fname = sys.argv[1].split(".wav")[0] + "_moded.wav"
    write(fname, fs, np.array(filtered_signal, dtype=np.int16))
