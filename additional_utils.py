import tqdm
from pathlib import Path
import os
import librosa
import soundfile
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

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



def downsample_tracks(origin_directory, target_directory, target_sample_rate):

    for root, subdirs, files in os.walk(origin_directory):

        print(root, ' ; ', subdirs, ' - ', files)

        for file_name in tqdm.tqdm(files):

            # get WAV file's path
            original_file_directory = root + '\\' + file_name

            # read WAV file
            origin_sample_rate, origin_audio = wavfile.read(original_file_directory)

            # plot
            #show_waveform_2_channels(origin_audio, 'Original audio waveform')

            # get current number of samples and channels
            origin_num_samples, origin_num_channels = origin_audio.shape

            # calculate the resampling ratio
            resampling_ratio = target_sample_rate / origin_sample_rate

            # calculate the new number of samples based on the resampling ratio
            target_num_samples = int(origin_num_samples * resampling_ratio)

            # resampling
            #target_audio = librosa.resample(origin_audio, orig_sr=origin_sample_rate, target_sr=target_sample_rate)
            target_audio = signal.resample(origin_audio, target_num_samples)
            #target_audio.shape
            #target_audio = origin_audio

            # plot
            #show_waveform_2_channels(target_audio, 'Original audio waveform')

            # create target directory
            os.makedirs(root.replace(origin_directory, target_directory), exist_ok=True)

            # create directory to target file
            target_file_directory = root.replace(origin_directory, target_directory) + '\\' + file_name

            # write modified audio to WAV file
            wavfile.write(target_file_directory, target_sample_rate, target_audio)



if __name__ == "__main__":
    input_directory = "musdb18_ORG_separated"
    output_directory = "DATASET_16kHz_2channels"
    # sample_rate = 16000
    sample_rate = 44100

    downsample_tracks(input_directory, output_directory, sample_rate)

    #8835072 / 44100