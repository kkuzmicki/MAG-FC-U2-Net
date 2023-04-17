# import librosa

# # define path to the input stem file
# input_stem_path = "C://Users//kkuzm//Desktop//MAG-FC-U2-Net//musdb18//train//Actions - Devil's Words.stem.mp4"

# # define path to the output wav file
# output_wav_path = "C://Users//kkuzm//Desktop//MAG-FC-U2-Net//musdb18//train//AAA//file.wav"

# # use librosa to load the audio data from the stem file using PySoundFile backend
# y, sr = librosa.load(input_stem_path, sr=None, mono=False, res_type='kaiser_best')
# # Note that we use 'res_type' argument to specify the backend for reading audio data.

# # use soundfile to write the audio data to the output wav file
# sr.write(output_wav_path, y, sr, format='WAV', subtype='PCM_16')

from stempeg import Musdb18
import os

# Load all stems in the MUSDB18 dataset
musdb = Musdb18('C:/Users/kkuzm/Desktop/MAG-FC-U2-Net/musdb18/train')

# Loop through each track in the dataset
for track in musdb:

    # Create a directory for the output files
    output_dir = f'C:/Users/kkuzm/Desktop/MAG-FC-U2-Net/out{track.name}'
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each stem in the track
    for stem_idx, stem in enumerate(track.stems):

        # Write the stem to a WAV file
        stem_filename = f'{output_dir}/stem{stem_idx}.wav'
        stem.write_wav(stem_filename)