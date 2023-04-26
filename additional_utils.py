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
            target_file_directory = root.replace(directory, destination) + '\\' + file_name

            y, s = soundfile.read(original_file_directory)
            print('y: ', y)

            soundfile.write(target_file_directory, y, samplerate=16000)



if __name__ == "__main__":
    downsample_tracks("TARGET", "DATASET_16kHz")