import tqdm
from pathlib import Path
import os
import librosa
import soundfile
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy

def calculate_differences_between_arrays(arr1, arr2):
    differences = []
    for val1, val2 in zip(arr1, arr2):
        diff = val1 - val2
        differences.append(diff)
    return differences

def show_plot_for_list(list, header):
    plt.plot(list)
    plt.title(header)
    plt.show()

if __name__ == "__main__":

    target_sample_rate = 16000

    # audio read
    origin_sample_rate, origin_audio = wavfile.read('org.wav')
    wavfile.write('out_org.wav', origin_sample_rate, origin_audio[:,0])

    origin_num_samples, origin_num_channels = origin_audio.shape

    # calculate the resampling ratio
    resampling_ratio = target_sample_rate / origin_sample_rate

    # calculate the new number of samples based on the resampling ratio
    target_num_samples = int(origin_num_samples * resampling_ratio)

    # resampling
    target_audio_scipy = scipy.signal.resample(origin_audio[:,0], target_num_samples).astype(int)
    target_audio_scipy = np.array(target_audio_scipy, np.int16)

    # show differences
    diffArray = calculate_differences_between_arrays(origin_audio[:,0], target_audio_scipy)
    show_plot_for_list(diffArray, 'Differences')

    # write modified audio to WAV file
    wavfile.write('out.wav', target_sample_rate, target_audio_scipy)