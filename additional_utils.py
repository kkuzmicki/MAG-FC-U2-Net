import tqdm
from pathlib import Path
import os
import librosa
import soundfile
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy

# audio - [sample index][channel number] i.e. audio[:,0] audio[0,:]
def show_waveform_2_channels(audio, header):
    # 1. channel
    plt.plot(audio[:,0], label="1. channel")

    # 2. channel
    plt.plot(audio[:,1], label="2. channel")
    
    # samples axis - X
    plt.xlabel('x - samples')

    # amplitude - y
    plt.ylabel('y - amplitude')
  
    # giving a title to my graph
    plt.title(header)
  
    # function to show the plot
    plt.show()

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


def downsample_tracks(origin_directory, target_directory, target_sample_rate):

    for root, subdirs, files in os.walk(origin_directory):

        for file_name in tqdm.tqdm(files):

            # get WAV file's path
            original_file_directory = root + '\\' + file_name

            # read WAV file
            origin_sample_rate, origin_audio = wavfile.read(original_file_directory)

            origin_num_samples, origin_num_channels = origin_audio.shape

            # calculate the resampling ratio
            resampling_ratio = target_sample_rate / origin_sample_rate

            # calculate the new number of samples based on the resampling ratio
            target_num_samples = int(origin_num_samples * resampling_ratio)

            # resampling [1 2][3 4][2 -3]...[5 6] [1 3 4 5 5 33 2][1 2 3 3 3 2 3]
            target_audio_L = scipy.signal.resample(origin_audio[:,0], target_num_samples).astype(int)
            target_audio_L = np.array(target_audio_L, np.int16)

            target_audio_R = scipy.signal.resample(origin_audio[:,1], target_num_samples).astype(int)
            target_audio_R = np.array(target_audio_R, np.int16)

            # transpose from [1 3 4 5 6][3 4 5 4 3] into [1 3][3 4] etc...
            final_target_audio = np.array(list(zip(target_audio_L, target_audio_R)))

            # create target directory
            os.makedirs(root.replace(origin_directory, target_directory), exist_ok=True)

            # create directory to target file
            target_file_directory = root.replace(origin_directory, target_directory) + '\\' + file_name

            # write modified audio to WAV file
            wavfile.write(target_file_directory, target_sample_rate, final_target_audio)



if __name__ == "__main__":
    input_directory = "musdb18_ORG_separated/train"
    output_directory = "DATASET_16kHz_2channels"
    sample_rate = 16000

    downsample_tracks(input_directory, output_directory, sample_rate)

    #8835072 / 44100