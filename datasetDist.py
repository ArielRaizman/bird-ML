import os
import pandas as pd
from pathlib import Path


def load_ffl_annotations(data_dir: str = "FFL-Annotations/Data") -> pd.DataFrame:
    # Columns to read (invariant across spectrograms)
    COLUMNS_TO_READ = [
        "Selection", "Channel", "Begin Time (s)", "End Time (s)",
        "Low Freq (Hz)", "High Freq (Hz)", "SPP", "EXP", "TRIAL"
    ]
    
    # Get all txt files in the directory
    data_path = Path(data_dir)
    txt_files = sorted(data_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {data_dir}")
    
    print(f"Found {len(txt_files)} txt files to process")
    
    dataframes = []
    
    for file_path in txt_files:
        print(f"Processing: {file_path.name}")
        
        # First, check the header to see if we have TYPE or CALL
        with open(file_path, 'r') as f:
            header = f.readline().strip().split('\t')
        
        has_type = "TYPE" in header
        has_call = "CALL" in header
        
        if has_type and has_call:
            print(f"ERROR: File {file_path.name} contains both TYPE and CALL columns")
            raise ValueError(f"Both TYPE and CALL columns exist in {file_path.name}")
        
        # Determine which call column to read
        call_column = "TYPE" if has_type else "CALL" if has_call else None
        
        if call_column is None:
            raise ValueError(f"Neither TYPE nor CALL column found in {file_path.name}")
        
        # Read only the columns we need
        columns_for_file = COLUMNS_TO_READ + [call_column]
        df = pd.read_csv(file_path, sep="\t", usecols=columns_for_file)
        
        # Rename CALL to TYPE if necessary
        if call_column == "CALL":
            df = df.rename(columns={"CALL": "TYPE"})
        
        # Remove duplicates (same annotation from different spectrograms)
        rows_before = len(df)
        df = df.drop_duplicates()
        rows_after = len(df)
        duplicates_removed = rows_before - rows_after
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate entries from multiple spectrograms")
        
        dataframes.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\nSuccessfully loaded {len(combined_df)} total annotations from {len(txt_files)} files")
    print(f"Columns: {list(combined_df.columns)}")
    
    return combined_df


if __name__ == "__main__":
    # Load all annotations
    df = load_ffl_annotations()
    
    # Display summary information
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nUnique species in SPP column: {df['SPP'].nunique()}")
    print(f"Species: {sorted(df['SPP'].unique())}")
    
    # Find top 5 most frequent SPP-TYPE combinations
    print(f"\n{'='*60}")
    print("Top 5 Most Frequent SPP-TYPE (CALL) Combinations:")
    print(f"{'='*60}")
    spp_type_counts = df.groupby(['SPP', 'TYPE']).size().sort_values(ascending=False)
    
    for i, ((spp, call_type), count) in enumerate(spp_type_counts.items(), 1):
        print(f"{i}. SPP: {spp:6s}  CALL: {call_type:4s}  Count: {count:5d}")
