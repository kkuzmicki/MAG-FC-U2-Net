import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import argparse
import scipy.signal
import soundfile as sf
import norbert
import json
from pathlib import Path
import utils
import tqdm
import io
import time
import matplotlib.pyplot as plt
import importlib
from u2net import u2net
import experiments as ex

# target: 'vocals'
# model_name: path_to_model
# device: 'cuda'
#
# returns u2net model with weights from .pth file and parameters from .json file
def load_model(target, model_name='umxhq', device=torch.device("cpu")):
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        print("Model does not exist!")
    else:
        with open(Path(model_path, target + '.json'), 'r') as stream: # opens vocals.json
            results = json.load(stream) # saves file stream as python object

        target_model_path = next(Path(model_path).glob("%s.pth" % target)) # pth file stores weights of model and structure
        # print(target_model_path)
        state = torch.load(target_model_path, map_location=device)

        mymodel = u2net(2, 2, results['args']['bins'])
        
        mymodel.load_state_dict(state) # inherited method, loads pytorch model as pytorch model (ML model)
        mymodel.eval() # sets model to evaluation / 'testing' mode
        mymodel.to(device) # moves to CUDA

        params = { # saves results from .json file
            'fft': results['args']['fft'],
            'hop': results['args']['hop'],
            'dur': results['args']['dur'],
            'channels': results['args']['channels'],
            'sample_rate': results['args']['sample_rate'],
        }
        return mymodel, params

def transform(audio, model, fft, hop, device): # audio >>> torch.Size([2, 130560])
    with torch.no_grad():
        audio_stft = utils.STFT(audio[None, ...], None, fft, hop) # audio_stft's size: torch.Size([1, 2, 513, 128, 2]); in index: None == np.newaxis
        audio_torch = utils.Spectrogram(audio_stft) # audio_torch's shape: torch.Size([1, 2, 513, 128])
        audio_torch = audio_torch.to(device)
        mag_target = model(audio_torch)

        mag_target, mag_mask = model(audio_torch)
        mag_target = mag_target * F.sigmoid(mag_mask)
        mag_target = mag_target.cpu().detach()

        mag_target = mag_target.reshape(-1, mag_target.shape[-2], mag_target.shape[-1]) # after: torch.Size([2, 513, 128])
        X = torch.stft(audio, fft, hop, window=torch.hann_window(fft), return_complex=False) # after: torch.Size([2, 513, 256])
        #magnitude, phase = ex.magphase(X) # https://pytorch.org/audio/0.9.0/_modules/torchaudio/functional/functional.html#magphase
        X = torch.view_as_complex(X)
        magnitude = torch.abs(X) # torch.Size([2, 513, 256])
        phase = torch.angle(X) # torch.Size([2, 513, 256])
        complex = torch.stack((mag_target * torch.cos(phase), mag_target * torch.sin(phase)), -1)
        audio_hat = torch.istft(complex, fft, hop, fft, torch.hann_window(fft)).numpy() # https://pytorch.org/docs/stable/generated/torch.istft.html

    return audio_hat

# 'C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\DATASET_16kHz_2channels\\test\\Al James - Schoolboy Facination\\mixture.wav'
# 'vocals'
# 'C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\models\\musdb16_model_first'
# type='cuda'
#
# takes model, input file and separates target
def separate(
        input_file_path, target, model_name='umxhq', device=torch.device("cpu")
):

    Model, params = load_model(target=target, model_name=model_name, device=device) # returns created u2net model with weights from .pth file, and parameters from .json file

    # saved model parameters
    fft = params['fft'] # 1024
    hop = params['hop'] # 512
    dur = params['dur'] # 256

    channels = params['channels'] # 2
    model_sample_rate = params['sample_rate'] # 16000

    audio, file_sample_rate = torchaudio.load(input_file_path) # audio shape: torch.Size([2, 3205468]), i.e.: [1322, 4324], [4324, 324], ...

    # if test song isn't the same sample rate as model, then test song is resampled
    if file_sample_rate != model_sample_rate:
        audio = torchaudio.transforms.Resample(file_sample_rate, model_sample_rate)(audio)

    if channels == 1:
        if audio.shape[0] == 2:
            audio = torch.mean(audio, 0)
    else:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

    total_length = audio.shape[1] # 3205468
    window = hop * (dur * 1 - 1) # 130560
    stride = window // 2 # 65280
    rest = stride - (total_length - window)%stride # 58532
    audio = torch.cat([audio, torch.zeros((channels, rest))], -1) # audio >>> torch.Size([2, 3264000])
    start = 0
    num = np.zeros((channels, audio.shape[1]))
    audio_sum = np.zeros((channels, audio.shape[1]))

    while start < audio.shape[1] - window + 1:
        audio_split = audio[:, start:start + window]
        num[:, start:start + window] += 1

        audio_hat = transform(audio_split, Model, fft, hop, device)

        audio_sum[..., start:start + window] = audio_hat / num[..., start:start + window] 
        + audio_sum[..., start:start + window] * (num[..., start:start + window] - 1) / num[..., start:start + window]

        start += stride

    audio_sum = audio_sum[:,:-rest]

    return audio_sum.T, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Separation')

    parser.add_argument('input', type=str, nargs='+')

    parser.add_argument('--target', type=str, default='vocals')

    parser.add_argument('--model', type=str, default='./models/vocalacc')

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in args.input:
        estimate, params = separate(
            input_file,
            target=args.target,
            model_name=args.model,
            device=device,
        )

        output_path = Path('temp', Path(input_file).stem + '_' + Path(args.model).stem)
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path / Path(args.target).with_suffix('.wav')),
            estimate,
            params['sample_rate']
        )

