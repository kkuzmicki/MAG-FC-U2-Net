import torch
import torchaudio
import os
import numpy as np
import math

def torchaudio_info(path):
    import torchaudio

    info = {}
    si = torchaudio.info(str(path)) # deleted , _
    print(si)
    info['samplerate'] = si.sample_rate
    #info['samples'] = si.length // si.channels 
    info['samples'] = si.num_frames
    # info['duration'] = si.num_frames // si.sample_rate # mine
    info['duration'] = si.num_frames
    return info

def torchaudio_loader(path, start=0, dur=None):
    import torchaudio
    # info = torchaudio_info(path)

    if dur is None:
        sig = torchaudio.load(path) # deleted ,rate
        return sig[0]
    else:
        sig = torchaudio.load( # deleted ,rate; tensor has [1 3 4 5 4 ][2 3 4 3 2] format (not pairs [1 2][3 1] like wavfile)
            path, num_frames=dur
        )
        return sig[0] # added index, because torchaudio.load returns tuple [frames; sample_rate]

def soundfile_info(path):
    import soundfile
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = int(sfi.duration * sfi.samplerate)
    return info

def soundfile_loader(path, start=0, dur=None):
    import soundfile

    info = soundfile_info(path)
    start = start
    # check if dur is none
    if dur:
        stop = start + dur
    else:
        stop = dur

    audio, _ = soundfile.read(
        path,
        always_2d=True,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def load_info(path):
    return torchaudio_info(path)


def load_audio(path, start=0, dur=None):
    return torchaudio_loader(path, start=start, dur=dur)


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )
    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(state, is_best, path, target):
    torch.save(state, os.path.join(path, target + '.chkpnt'))
    if is_best:
        torch.save(state['state_dict'], os.path.join(path, target + '.pth'))

    if state['epoch']%5==0 and state['epoch']>100:
        torch.save(state['state_dict'], os.path.join(path, target + str(state['epoch']) +'.pth'))
    


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

def STFT(x, device, n_fft=4096, n_hop=1024): # n_fft = 1024; n_hop = 512
    nb_samples, nb_channels, nb_timesteps = x.size() # train.py: torch.Size([12, 2, 130560]) # 12 is number of songs
    x = x.reshape(nb_samples * nb_channels, -1) # train.py: shape[24, 130560]; test.py: torch.Size([2, 130560]) # channels and songs merge here as each one separate track (so for 2 ch it doubles)

    # torch.stft(input:must be either a 1-D time sequence or a 2-D batch/group of time sequences,
    # n_fft - how many samples are taken for one timestamp,
    # hop_length - distance through how many samples n_fft-sized Hann window moves; so for input's size=130560 and hop_lenght=512 it gives 256 column in spectrogram, this is 256 frames of frequency set,
    # window - window to move
    # )
    x = torch.stft(
        x,
        n_fft=n_fft, hop_length=n_hop,
        window=torch.hann_window(n_fft, device=device),
        center=True, normalized=False, onesided=True, pad_mode='constant', return_complex=False
    ) # train.py: torch.Size([24, 513, 256]) IF return_complex=False, then torch.Size([24, 513, 256, 2])| test.py: torch.Size([2, 513, 256])
    x = x.contiguous().view(nb_samples, nb_channels, n_fft // 2 + 1, -1, 2) # train.py: torch.Size([12, 2, 513, 128, 2]) IF return_complex=False THEN torch.Size([12, 2, 513, 256, 2]) # LEGACY
    #x = x.contiguous().view(nb_samples, nb_channels, n_fft // 2 + 1, -1) # train.py: torch.Size([12, 2, 513, 256])

    return x 
    # train.py: torch.Size([12 <<< audio fragmens/songs, 2 <<< channels, 513 <<< frequencies (rows in spectrogram), 
    # 256 <<< timestamps (columns in spectrogram), 2 <<< real and imaginary number])

def ComplexFFT(x):
    x = x.permute(0, 1, 4, 2, 3)
    x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
    return x

def Spectrogram(x):
    x = torch.norm(x, dim=-1) # train.py: orch.Size([12, 2, 513, 128]) IF return_complex=False THEN torch.Size([12, 2, 513, 256]) # LEGACY
    #x = torch.norm(x, dim=0) # 

    return x
    # return torch.log10(x + 1)

if __name__ == "__main__":
    from scipy import signal
    import torch
    import time

    x = torch.rand((1, 1, 3, 2))
    input_mean = torch.tensor([1, 2, 3])
    print(x)
    x = x.transpose(2, 3)
    x += input_mean
    x = x.transpose(2, 3)
    # x = torch.nn.functional.pad(x, (0,0,0,2), "constant",0)
    # y = torch.zeros((2,3))
    #
    # x = torch.cat((x,y),-1)
    # x[0,0] =12
    print(x.shape)
    print(x)
    torch.autograd.profiler()
    # x = torch.rand((12,2,44100*6))
    # x = STFT(x)
    # print(x.shape)

    # while True:
    #     pass

    # import tqdm
    # import time
    # for i in tqdm.tqdm(range(100),ascii=True):
    #     time.sleep(0.1)

    pass
    # unmix = model.OpenUnmix(
    #     input_mean=None,
    #     input_scale=None,
    #     nb_channels=1,
    #     hidden_size=512,
    #     sample_rate=16000,
    #     n_fft=4096,
    #     n_hop=1024,
    #     max_bin=2049,
    # )
    # unmix = torchvision.models.resnet18()
    # input = torch.rand((1,3,224,224))
    # modelsize(unmix,input)
