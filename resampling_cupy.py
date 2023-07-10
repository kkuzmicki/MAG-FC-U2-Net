import cupy as cp
import cupyx.scipy.signal as cp_signal
from scipy.io import wavfile
import os
import tqdm
from multiprocessing import Process


def downsample_tracks(origin_directory, target_directory, target_sample_rate):
    for root, subdirs, files in os.walk(origin_directory):
        for file_name in tqdm.tqdm(files):
            # Get WAV file's path
            original_file_directory = os.path.join(root, file_name)

            # Read WAV file
            origin_sample_rate, origin_audio = wavfile.read(original_file_directory)

            origin_num_samples, origin_num_channels = origin_audio.shape

            # Calculate the resampling ratio
            resampling_ratio = target_sample_rate / origin_sample_rate

            # Calculate the new number of samples based on the resampling ratio
            target_num_samples = int(origin_num_samples * resampling_ratio)

            # Convert to CuPy arrays
            origin_audio_cp = cp.asarray(origin_audio)

            # Resampling
            target_audio_L = cp_signal.resample_poly(origin_audio_cp[:, 0], target_sample_rate, origin_sample_rate).astype(cp.int16)
            target_audio_R = cp_signal.resample_poly(origin_audio_cp[:, 1], target_sample_rate, origin_sample_rate).astype(cp.int16)

            # Transpose from [1 3 4 5 6][3 4 5 4 3] into [1 3][3 4] etc...
            final_target_audio = cp.array(list(zip(target_audio_L, target_audio_R)))

            # Create target directory
            os.makedirs(root.replace(origin_directory, target_directory), exist_ok=True)

            # Create directory for target file
            target_file_directory = os.path.join(root.replace(origin_directory, target_directory), file_name)

            # Convert final_target_audio back to NumPy array for writing WAV file
            final_target_audio_np = cp.asnumpy(final_target_audio)

            # Write modified audio to WAV file
            wavfile.write(target_file_directory, target_sample_rate, final_target_audio_np)


if __name__ == "__main__":
    input_directory = "DATASET_41kHz_2channels_cores"
    output_directory = "DATASET_16kHz_2channels"
    sample_rate = 16000

    p1 = Process(target=downsample_tracks, args=(input_directory + '/core1', output_directory, sample_rate))
    p2 = Process(target=downsample_tracks, args=(input_directory + '/core2', output_directory, sample_rate))
    p3 = Process(target=downsample_tracks, args=(input_directory + '/core3', output_directory, sample_rate))
    p4 = Process(target=downsample_tracks, args=(input_directory + '/core4', output_directory, sample_rate))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
