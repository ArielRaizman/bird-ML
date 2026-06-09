from asyncio import transports
import os
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import pandas as pd
import numpy as np
# CALL = "MYLO_C"
CALL = "MYAX_C"
# Look at folder raven_analysis_{CALL}, check for all false negatives and false positives, display those spectrograms in a grid on a page. 
def plot_spectrograms(call):
    error_dir = Path(f"raven_analysis_{call}")
    if not error_dir.exists():
        print("Error folder not found")
        return
    
    false_positives = []
    false_negatives = []
    true_positives = []

    # assuming matching audio files exist
    for f in error_dir.glob("*"):
        audio_dir = Path("FFL-Annotations/Audio")
        audio_name = "_".join(str(f.stem).split("_")[3:])
        audio_path = audio_dir / f"{audio_name}.wav"
        if not audio_path.exists():
            print(f"Cannot find audio file {audio_path}, skipping...")
            continue

        # read the raven tables
        df = pd.read_csv(f, sep='\t')

        # df that filters the fps and fns
        errors = df[df['Classification Type'].isin(['false_positive','false_negative','true_positive'])]
        # print(errors)
        for _, row in errors.iterrows():
            start = row['Begin Time (s)']
            end = row['End Time (s)']
            duration = end - start

            # load audio segment into memory
            y, sr = librosa.load(audio_path, offset=start, duration=duration)

            error_data = (y, sr, f"{audio_name}\n{start}s")

            if row['Classification Type'] == 'false_positive':
                false_positives.append(error_data)
            elif row['Classification Type'] == 'false_negative':
                false_negatives.append(error_data)
            else:
                true_positives.append(error_data)
            
    return false_positives, false_negatives, true_positives

def create_spectrogram_grid(data, title):
    if not data:
        print("data not found")
        return
    
    # Calculate the number of rows based off how many spectrograms to display
    cols = 4
    rows = (len(data) // cols) + (1 if len(data) % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4), dpi=300)
    axes = axes.flatten()

    for i, (y, sr, label) in enumerate(data):
        #create spectrogram in memory
        D = np.abs(librosa.stft(y, n_fft=1024, hop_length=16))
        S_db = librosa.amplitude_to_db(D, ref=10) # use np.max to normalize spectorgram signal to loudest sound

        # show the spectrogram
        img = librosa.display.specshow(
            S_db, 
            sr=sr, 
            hop_length=16, 
            ax=axes[i], 
            x_axis='time', 
            y_axis='linear', 
            cmap='gray_r',
            vmin=-80, 
            vmax=0
        )
        
        axes[i].set_title(label, fontsize=9, fontweight='bold')
        axes[i].set_ylim(0, sr/2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{title}.png")
    
fps, fns, tps = plot_spectrograms(CALL)
create_spectrogram_grid(fps, f"{CALL}_fps")
create_spectrogram_grid(fns, f"{CALL}_fns")
create_spectrogram_grid(tps[:20], f"{CALL}_tps")
