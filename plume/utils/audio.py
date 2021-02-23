from scipy.signal import lfilter, butter
from scipy.io.wavfile import read, write
from numpy import array, int16
import sys


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
    import pdb; pdb.set_trace()
    low_freq = 300.0
    high_freq = 4000.0
    filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=6)
    fname = sys.argv[1].split(".wav")[0] + "_moded.wav"
    write(fname, fs, array(filtered_signal, dtype=int16))
