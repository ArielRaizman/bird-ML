import joblib
import pandas as pd
import numpy as np
import os
import bioacoustics_model_zoo as bmz
# Currently Temp Gemini Boilerplate TODO: Look over it, adjust everything make it working
def save_to_raven_table(predictions_df, output_filename, class_names):
    raven_data = []
    
    # Iterate through each row (which represents an audio chunk)
    for idx, row in predictions_df.iterrows():
        # OpenSoundscape/bmz index format when return_dfs=True is usually: (file, start_time, end_time)
        # If it's not a multi-index, we can extract from columns if present.
        if isinstance(predictions_df.index, pd.MultiIndex):
            file_path, start_time, end_time = idx
        else:
            # Fallback if start/end time are columns instead of index
            start_time = row.get('start_time', 0.0)
            end_time = row.get('end_time', start_time + 5.0)
            
        # Check which classes were predicted as positive (1)
        # row contains the 1/0 predictions for each class
        predicted_classes = [cls for cls in class_names if row[cls] == 1]
        
        for cls in predicted_classes:
            raven_data.append({
                'Selection': len(raven_data) + 1,
                'View': 'Spectrogram 1',
                'Channel': 1,
                'Begin Time (s)': start_time,
                'End Time (s)': end_time,
                'Low Freq (Hz)': 0,
                'High Freq (Hz)': 12000,
                'Prediction': cls
            })
            
    if raven_data:
        df = pd.DataFrame(raven_data)
        df.to_csv(output_filename, sep='\t', index=False)
        print(f"Saved {len(raven_data)} detections to {output_filename}")
    else:
        print("No detections found. Empty Raven table not created.")


def run_model_on_file(audio_path, pickled_model_path, output_raven_path):
    print("1. Loading PERCH Model from bioacoustics_model_zoo...")
    perch = bmz.Perch()
    
    print(f"2. Extracting embeddings from: {audio_path}")
    # Passing a list with the audio file. return_dfs=True keeps track of start/end times!
    embeddings_df = perch.embed(
        [audio_path],
        return_dfs=True,
        batch_size=32,
        num_workers=0
    )
    
    # If the embeddings are returned as a list of DFs (sometimes happens depending on version)
    if isinstance(embeddings_df, list):
        embeddings_df = pd.concat(embeddings_df)
        
    # The actual features are the values in the dataframe
    features = embeddings_df.values
    
    print("3. Loading your trained Logistic Regression Model...")
    multi_logreg = joblib.load(pickled_model_path)
    
    print("4. Running predictions...")
    # Predict returns a 2D array of 0s and 1s: shape (num_chunks, num_classes)
    predictions = multi_logreg.predict(features)
    
    
    class_names = ["THAR_B", "HARU_C", "MYLO_C", "MYME_C", "MYAX_C", "THAR_P"]
    
    # Create a DataFrame of predictions matching the embeddings dataframe index (which holds start/end times)
    predictions_df = pd.DataFrame(
        predictions, 
        index=embeddings_df.index, 
        columns=class_names
    )
    
    print("5. Exporting to Raven Selection Table...")
    save_to_raven_table(predictions_df, output_raven_path, class_names)


if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    AUDIO_FILE = "FFL-Annotations/Audio/YOUR_TEST_AUDIO.wav"
    MODEL_FILE = "multi_logreg_model.pkl"
    OUTPUT_RAVEN = "output_selections.txt"
    
    # Make sure the file exists before running
    if os.path.exists(AUDIO_FILE) and os.path.exists(MODEL_FILE):
        run_model_on_file(AUDIO_FILE, MODEL_FILE, OUTPUT_RAVEN)
    else:
        print("Please update AUDIO_FILE and MODEL_FILE paths in the script!")
