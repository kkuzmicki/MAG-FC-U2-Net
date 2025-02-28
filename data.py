from utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
import argparse
import random
#import musdb
import torch
import tqdm
import glob
import numpy as np
#import stempeg

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio) # err
        return audio

# -------------------------------------------------------------------

def _augment_gain(audio, low=0.25, high=1.25):

    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio
    
# -------------------------------------------------------------------

def load_datasets(args):

    # load all functions from this file with _augment_+(item from list) in its name
    # iterates through array after 'in', 
    # for each aug it performs function on the left
    # there is also a possibility to add condition on the right side
    # ---
    # GLOBALS: https://www.w3schools.com/python/python_variables_global.asp
    # globals saves to dicitionary global vars and FUNCS, key is func name, value is func itself
    source_augmentations = Compose(
        [globals()['_augment_' + aug] for aug in ['gain', 'channelswap']]
    )

    train_dataset = FixedSourcesTrackFolderDataset(
        root = args.root,
        target_file='vocals.wav',
        interferer_files=['drums.wav','bass.wav','other.wav'] ,
        sample_rate=args.sample_rate, # 16 000
        split='train',
        samples_per_track=args.samples_per_track,
        source_augmentations=source_augmentations,
        seq_duration = (args.dur-1) * args.hop, # 255 * 512 = 130560
        seed = args.seed,
    )
    
    return train_dataset

class FixedSourcesTrackFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            split='train',
            target_file='vocals.wav',
            interferer_files=['accompaniment.wav'],
            seq_duration=None,
            samples_per_track=16,
            random_chunks=True,
            random_track_mix=True,
            source_augmentations=lambda audio: audio,
            sample_rate=16000,
            seed=42,
    ):
        random.seed(seed)
        self.root = Path(root).expanduser() # expand an initial path component ~( tilde symbol) or ~user in the given path to user’s home directory
        self.split = split
        self.sample_rate = sample_rate # 16 000
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = [self.target_file] + self.interferer_files # it's just string list (string var 'target_file' is converted to 1element string, then added to the rest)
        print(self.source_files) # ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
        self.tracks = list(self.get_tracks()) # list of pairs {path --- min_duration}
        self.samples_per_track = samples_per_track # 32

    # if Shuffle=True, index values will be random; it is called problably 'len' times
    def __getitem__(self, index): # single iteration
        # first, get target track
        index = index // self.samples_per_track # 2502 // 32 floor division is division with round down (i.e. 2.4 >>> 2)

        track_path = self.tracks[index]['path']
        min_duration = self.tracks[index]['min_duration'] # 2739954 - so: 2739954 / 16000 = 
        if self.random_chunks: # get random start of song
            start = random.randint(0, min_duration - self.seq_duration) # song's duration - seq_duration
        else:
            start = 0

        audio_sources = []

        target_audio = load_audio( # 8,16second == 130560 samples frame
            track_path / self.target_file, start=start, dur=self.seq_duration # seq_duration = 130560
        )

        # tensors vs arrays (numpy):
        # tensors are faster on GPU
        # Tensors are immutable

        # there are augumented voice and accompaniament tracks (but these can be randomized):
        target_audio = self.source_augmentations(target_audio) # here error was occuring
        audio_sources.append(target_audio)

        for source in self.interferer_files: # iterates through [bass.wav, drums.wav etc]
            if self.random_track_mix: # option: takes from random songs
                random_idx = random.choice(range(len(self.tracks)))
                track_path = self.tracks[random_idx]['path']
                if self.random_chunks: # option: takes random frames
                    min_duration = self.tracks[random_idx]['min_duration']
                    start = random.randint(0, min_duration - self.seq_duration)

            audio = load_audio(
                track_path / source, start=start, dur=self.seq_duration
            )

            audio = self.source_augmentations(audio) # data.py: torch.Size([2, 130560])
            audio_sources.append(audio)

        stems = torch.stack(audio_sources) # torch.Size([4, 2, 130560])
        # # apply linear mix over source index=0
        x = stems.sum(0) # all sources summed up  # orch.Size([2, 130560])

        # y = stems.reshape(-1, stems.size(2))
        y = stems[0] # vocal

        # x: torch.Size([2, 130560]) y: torch.Size([2, 130560])
        return x, y

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    # because of yields, this is generator creating map/dict of song's paths and their durations
    # generator iterates through loop as long as hits any yield
    # if not, then generator is 'empty'
    def get_tracks(self):
        p = Path(self.root, self.split) # musdb18/train
        for track_path in tqdm.tqdm(p.iterdir(), disable=True): # i.e. musdb18\train\A Classic Education - NightOwl.stem.mp4

            if track_path.is_dir(): # p.iterdir() checks every directory, this condition is fullfilled when iterator targets folder
                source_paths = [track_path / s for s in self.source_files] # overrided div op works as concat 2 strings with slash between them
                # https://stackoverflow.com/questions/53083963/python-pathlib-operator-how-does-it-do-it

                print(source_paths) #[WindowsPath('musdb18_16kHz/train/Matthew Entwistle - Dont You Ever/vocals.wav'), WindowsPath('musdb18_16kHz/train/Matthew Entwistle - Dont You Ever/drums.wav')...
                
                if not all(sp.exists() for sp in source_paths): # exclude if there is not every track for proceded song
                    print("exclude track ", track_path)
                    continue

                if self.seq_duration is not None: # seq_duration = 130560
                    infos = list(map(load_info, source_paths)) # create info: samplerate, samples and duration loaded for each instrument path, this is loop
                    
                    min_duration = min(i['duration'] for i in infos) # get minimum duration of song's instrument tracks (they should be the same)

                    if min_duration > self.seq_duration: # PROBLEM: min_duration was (by me) in seconds, seq_duration is frames number (130560 / 16000 = 8,16 as in thesis)
                        yield ({
                            'path': track_path,
                            'min_duration': min_duration
                        })
                else:
                    yield ({'path': track_path, 'min_duration': None}) # return do listy?

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Test')
    parser.add_argument('--root', type=str, default='DATASET_16kHz_2channels') # data root with train and test folders, deeper with song folders

    parser.add_argument('--target', type=str, default='vocals') # what we want to separate

    parser.add_argument('--dur', type=int, default=256)

    parser.add_argument('--fft', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--seq-dur', type=float, default=6.0)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--sample-rate', type=int, default=16_000)

    parser.add_argument('--samples-per-track', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, metavar='S')
    args, _ = parser.parse_known_args()

    #train_dataset's type is: FixedSourcesTrackFolderDataset(torch.utils.data.Dataset)
    train_dataset = load_datasets(args) # I deleted parser as argument and args from vars

    # len is overriden and returns: numbers of songs (100) * samples_per_track (32)
    print("Number of train samples: ", len(train_dataset))

    # iterate over dataloader

    train_sampler = torch.utils.data.DataLoader(
        #train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        train_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(train_sampler)

    # for x, y in tqdm.tqdm(train_sampler): # commented
    for x, y in tqdm.tqdm(train_sampler): # commented
        #print('x: ', x, ' y: ', y)
        pass

    print("Data loaded!")