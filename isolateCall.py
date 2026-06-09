import os
import librosa
import noisereduce
import soundfile as sf
from pathlib import Path
from glob import glob
from scipy.signal import butter, sosfilt
import numpy as np

def apply_spectral_contrast_gate(y, sr, multiplier):
    # simulating ravens contrast slider
    # create a spectogram with STFT
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    
    # find mean energy of the spectrogram
    magnitude = np.abs(stft)
    mean_energy = np.mean(magnitude)
    threshold = mean_energy * multiplier
    
    # create a mask based on the threshold
    mask = magnitude >= threshold
    
    # apply the mask to the spectrogram
    stft_clean = stft * mask
    
    # convert it back into audio
    y_clean = librosa.istft(stft_clean, hop_length=512)
    
    return y_clean


def bandpass_filter(data,lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data

def isolateCall(
    audio_file, 
    call_start, 
    call_end,
    noise_start,
    noise_end,
    low_cut,
    high_cut,
    output_path
):
    # validate parameters
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if call_start >= call_end:
        raise ValueError("Call start time must be before call end time")
    
    if noise_start >= noise_end:
        raise ValueError("Noise start time must be before noise end time")
    
    if not output_path:
        base, _ = os.path.splitext(audio_file)
        output_path = f"{base}_cleaned_{call_start}s_to_{call_end}s.wav"

    # load audio file
    print(f"Loading '{audio_file}' into NumPy array with librosa")
    try:
        # sr=None preserves original sample rate
        y, sr = librosa.load(audio_file, sr=None, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")
    
    # more parameter checks
    duration = librosa.get_duration(y=y, sr=sr)
    if call_end > duration:
        raise ValueError(f"Call end time ({call_end}s) exceeds audio duration ({duration:.2f}s)")
    
    if noise_end > duration:
        raise ValueError(f"Noise end time ({noise_end}s) exceeds audio duration ({duration:.2f}s)")
    
    # convert time to sample indices
    call_start_idx = int(call_start * sr)
    call_end_idx = int(call_end * sr)
    noise_start_idx = int(noise_start * sr)
    noise_end_idx = int(noise_end * sr)

    # get the call
    print("Isolating call from audio file")
    call_segment = y[call_start_idx:call_end_idx]

    # noise reduction
    print("Performing noise reduction")
    noise_segment = y[noise_start_idx:noise_end_idx]
    filtered_noise = bandpass_filter(noise_segment, low_cut, high_cut, sr)
    clean_call = noisereduce.reduce_noise(y=call_segment, y_noise=filtered_noise, sr=sr)
    
    # applying high and low pass filters
    clean_call = bandpass_filter(clean_call, low_cut, high_cut, sr)
    
    # apply spectral contrast gate
    clean_call = apply_spectral_contrast_gate(clean_call, sr, 7) # 15


    # 3x louder than median thresholding
    # abs_signal = np.abs(clean_call)
    # median_val = np.median(abs_signal)
    # threshold = median_val * 7
    # clean_call[abs_signal < threshold] = 0

    # export
    try:
        sf.write(output_path, clean_call, sr)
        print(f"Success: Cleaned audio saved to '{output_path}'")
    except Exception as e:
        print(f"Failed to write output file: {e}")
    


# FILEPATH = "FFL-Annotations/Audio/TRIAL5_HAWK_14JUN2024_VILLAANA_MADREDEDIOS.wav"
# START = 2.209944167 - 0.1 # leeway
# END = 2.346643614 + 0.1
# NOISE_START = 2.4
# NOISE_END = 3
# LOW_CUT = 1707.097
# HIGH_CUT = 5227.984
# OUTPUT = "./MYAX_C4_Clean.wav"

FILEPATH = "FFL-Annotations/Audio/TRIAL5_HAWK_14JUN2024_VILLAANA_MADREDEDIOS.wav"
START = 25.096730652 - 0.1 # leeway
END = 25.224400754 + 0.1
NOISE_START = START-1.1
NOISE_END = START-0.1
LOW_CUT = 1389.706
HIGH_CUT = 4379.436
OUTPUT = "./MYAX_C3_Clean.wav"
isolateCall(FILEPATH, START, END, NOISE_START, NOISE_END, LOW_CUT, HIGH_CUT, OUTPUT)