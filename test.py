import argparse
from pathlib import Path
import torch
import torchaudio
import tqdm
import separate
import soundfile as sf
import time
import museval
import numpy as np
import pandas as pd
from mir_eval.separation import bss_eval_sources
import scipy.signal
import warnings
import librosa
warnings.filterwarnings("ignore")

def istft(X, rate=16000, n_fft=1024, n_hopsize=512):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio

def median_nan(a):
    return np.median(a[~np.isnan(a)])

# (test, after that evaluate) in loop
def test_eval(args):
    tracks = []
    p = Path(args.root, 'test')
    for track_path in tqdm.tqdm(p.iterdir(), disable=True):
        tracks.append(track_path)
    print("files_len", len(tracks))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    result = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR']) # why 'inf' SIR: https://github.com/craffel/mir_eval/issues/260
    reference_dir = Path(args.root, 'test')
    output_dir = Path(args.output_dir, Path(args.model).name, 'museval')

    for track in tqdm.tqdm(tracks):
        input_file = str(Path(track, args.input)) # songs_name/mixture.wav

        estimate, params = separate.separate(
            input_file, # 'C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\DATASET_16kHz_2channels\\test\\Al James - Schoolboy Facination\\mixture.wav'
            target=args.target, # 'vocals'
            model_name=args.model, # 'C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\models\\musdb16_model_first'
            device=device, # type='cuda'
        )

        output_path = Path(args.output_dir, Path(args.model).stem, 'estimates', Path(input_file).parent.name) # eval/musdb16_model_first/estimates/Al James - Schoolboy Facination
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path) + '/' + args.target + '.wav',
            estimate,
            params['sample_rate']
        )

        estdir = output_path # eval/musdb16_model_first/estimates/Al James - Schoolboy Facination
        refdir = Path(reference_dir, estdir.name) # C:/Users/kkuzm/Desktop/MAG-FC-U2-Net/DATASET_16kHz_2channels/test/Al James
        if refdir.exists():
            ref, sr = sf.read(str(Path(refdir, args.target + '.wav')), always_2d=True) # ref=(3205468, 2) # sr=16000
            est, sr = sf.read(str(Path(estdir, args.target + '.wav')), always_2d=True) # est=(3205468, 2) # sr=16000
            ref = ref[None, ...] # (1, 3205468, 2)
            est = est[None, ...] # (1, 3205468, 2)

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track': estdir.name, # 'Al James - Schoolboy Facination'
                "SDR":   median_nan(SDR[0]), # SDR=(1, 200)
                "ISR":   median_nan(ISR[0]), # ISR=(1, 200)
                "SIR":   median_nan(SIR[0]), # SIR=(1, 200)
                "SAR":   median_nan(SAR[0])  # SAR=(1, 200)
            }
            result.loc[result.shape[0]] = values # (number_of_songs, 5<<<columns) # shape increases after each new record addition
        # print(values)
        # break
    values = {
        'track':'sum',
        "SDR": result['SDR'].median(),
        "ISR": result['ISR'].median(),
        "SIR": result['SIR'].median(),
        "SAR": result['SAR'].median()
    }
    result.loc[result.shape[0]] = values
    print('Summary: ')
    print(list((result.loc[result.shape[0] - 1])[1:])) # displays last row
    result.to_csv(str(output_dir)+'.csv',index=0) # saves to CSV file



def test_main(args):
    tracks = []
    p = Path(args.root, 'test')
    for track_path in tqdm.tqdm(p.iterdir(), disable=True):
        tracks.append(track_path)
    print("files_len", len(tracks))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    start_time = time.time()

    for track in tqdm.tqdm(tracks):
        input_file = str(Path(track, args.input))

        estimate, params = separate.separate(
            input_file,
            target=args.target,
            model_name=args.model,
            device=device,
        )

        output_path = Path(args.output_dir, Path(
            args.model).stem, 'estimates', Path(input_file).parent.name)
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path) + '/' + args.target + '.wav',
            estimate,
            params['sample_rate']
        )
        break

    # print(time.time() - start_time)


def eval_main(args):

    reference_dir = Path(args.root, 'test')
    estimates_dir = Path(args.output_dir, Path(args.model).name, 'estimates')
    output_dir = Path(args.output_dir, Path(args.model).name, args.target)

    result = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])

    estdirs = Path(estimates_dir).iterdir()
    for estdir in tqdm.tqdm(list(estdirs)):
        refdir = Path(reference_dir, estdir.name)
        if refdir.exists():

            ref, sr = sf.read(
                str(Path(refdir, args.target + '.wav')), always_2d=True)
            est, sr = sf.read(
                str(Path(estdir, args.target + '.wav')), always_2d=True)

            ref = ref[None, ...]
            est = est[None, ...]

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track': estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            result.loc[result.shape[0]] = values
            # print(values)
        # break
    values = {
        'track': 'sum',
        "SDR": result['SDR'].median(),
        "ISR": result['ISR'].median(),
        "SIR": result['SIR'].median(),
        "SAR": result['SAR'].median()
    }
    result.loc[result.shape[0]] = values
    print(list((result.loc[result.shape[0] - 1])[1:]))
    result.to_csv(str(output_dir)+'.csv', index=0)

def EVALUATE_WITH_ACCOMPANIMENT(args):

    reference_dir = Path(args.root, 'test')
    estimates_dir = Path(args.output_dir, Path(args.model).name, 'estimates')
    output_dir = Path(args.output_dir, Path(args.model).name, args.target)

    result_voice = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])
    result_accompaniment = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])

    estdirs = Path(estimates_dir).iterdir()
    for estdir in tqdm.tqdm(list(estdirs)):
        refdir = Path(reference_dir, estdir.name)
        if refdir.exists():

            ref, sr = sf.read(
                str(Path(refdir, args.target + '.wav')), always_2d=True)
            ref_accompaniment, sr = sf.read(
                str(Path(refdir, 'accompaniment.wav')), always_2d=True)
            
            est, sr = sf.read(
                str(Path(estdir, args.target + '.wav')), always_2d=True)
            est_accompaniment, sr = sf.read(
                str(Path(estdir, 'accompaniment.wav')), always_2d=True)

            ref = ref[None, ...]
            ref_accompaniment = ref_accompaniment[None, ...]
            ref = np.concatenate((ref, ref_accompaniment), axis=0)

            est = est[None, ...]
            est_accompaniment = est_accompaniment[None, ...]
            est = np.concatenate((est, est_accompaniment), axis=0)

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values_voice = {
                'track': estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            values_accompaniment = {
                'track': estdir.name,
                "SDR": median_nan(SDR[1]),
                "ISR": median_nan(ISR[1]),
                "SIR": median_nan(SIR[1]),
                "SAR": median_nan(SAR[1])
            }

            result_voice.loc[result_voice.shape[0]] = values_voice
            result_accompaniment.loc[result_accompaniment.shape[0]] = values_accompaniment
            # print(values)
        # break
    values_voice = {
        'track': 'sum',
        "SDR": result_voice['SDR'].median(),
        "ISR": result_voice['ISR'].median(),
        "SIR": result_voice['SIR'].median(),
        "SAR": result_voice['SAR'].median()
    }
    values_accompaniment = {
        'track': 'sum',
        "SDR": result_accompaniment['SDR'].median(),
        "ISR": result_accompaniment['ISR'].median(),
        "SIR": result_accompaniment['SIR'].median(),
        "SAR": result_accompaniment['SAR'].median()
    }

    result_voice.loc[result_voice.shape[0]] = values_voice
    result_accompaniment.loc[result_accompaniment.shape[0]] = values_accompaniment
    print(list((result_voice.loc[result_voice.shape[0] - 1])[1:]))
    print(list((result_accompaniment.loc[result_accompaniment.shape[0] - 1])[1:]))
    result_voice.to_csv(str(output_dir)+'___results_vocal.csv', index=0)
    result_accompaniment.to_csv(str(output_dir)+'___results_accompaniment.csv', index=0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MUSIC test')

    # vocals accompaniment
    parser.add_argument('--target', type=str, default='vocals')

    parser.add_argument('--model', type=str, default='C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\models\\musdb16_model_attention_4b_realArticle')

    parser.add_argument('--root', type=str, default='C:\\Users\\kkuzm\\Desktop\\MAG-FC-U2-Net\\DATASET_16kHz_2channels')

    parser.add_argument('--input', type=str, default='mixture.wav')

    parser.add_argument('--output_dir', type=str, default='./eval')

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    EVALUATE_WITH_ACCOMPANIMENT(args)
    #test_eval(args)