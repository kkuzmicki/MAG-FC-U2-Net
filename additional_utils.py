import tqdm
from pathlib import Path
import os
import librosa
import soundfile

def downsample_tracks(directory, destination):
    p = Path(directory)
    for root, subdirs, files in os.walk(directory):

        print(root, ' ; ', subdirs, ' - ', files)

        for file_name in files:
            original_file_directory = root + '\\' + file_name

            os.makedirs(root.replace(directory, destination), exist_ok=True)
            target_file_directory = root.replace(directory, destination) + '\\' + file_name

            #y, s = soundfile.read(original_file_directory)
            y, s = librosa.load(original_file_directory, sr=16000) # librosa used because of possibility to set samplerate, there is downsampling to 16kHz
            print('y: ', y)

            soundfile.write(target_file_directory, y, samplerate=16000) # this samplerate doesn't downsample!
            print('TRACK\'S PATH: ', target_file_directory)



if __name__ == "__main__":
    downsample_tracks("TARGET", "DATASET_16kHz")