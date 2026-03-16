import pandas as pd
import glob
import os

def analyze_call_distribution(folder_path):
    """
    Parses all selection tables in a folder and returns a distribution
    of species (SPP) and vocalization types (TYPE/CALL).
    Handles both 'TYPE' and 'CALL' column names.
    Removes duplicate calls based on unique identifiers.
    """
    # Raven Pro often exports as .txt or .csv
    file_pattern = os.path.join(folder_path, "*.[tc][xs][tv]") 
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"No annotation files found in {folder_path}")
        return None

    all_data = []

    for file in all_files:
        try:
            # engine='python' with sep=None allows pandas to sniff the delimiter (tab vs comma)
            df = pd.read_csv(file, sep=None, engine='python')
            
            # Ensure columns are stripped of whitespace
            df.columns = [c.strip() for c in df.columns]
            
            # Add source file name for tracking and deduplication
            df['source_file'] = os.path.basename(file)
            
            all_data.append(df)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    if not all_data:
        return None

    # Concatenate all selections into one master dataframe
    master_df = pd.concat(all_data, ignore_index=True)

    # Handle both 'TYPE' and 'CALL' column names
    # Normalize to a single 'call_type' column
    if 'TYPE' in master_df.columns and 'CALL' in master_df.columns:
        # If both exist, prefer TYPE but merge non-null values
        master_df['call_type'] = master_df['TYPE'].fillna(master_df['CALL'])
        print("Warning: Both 'TYPE' and 'CALL' columns found. Using merged values.")
    elif 'TYPE' in master_df.columns:
        master_df['call_type'] = master_df['TYPE']
    elif 'CALL' in master_df.columns:
        master_df['call_type'] = master_df['CALL']
    else:
        print(f"Missing required column 'TYPE' or 'CALL'. Found: {list(master_df.columns)}")
        return master_df

    # Check for SPP column
    if 'SPP' not in master_df.columns:
        print(f"Missing required column 'SPP'. Found: {list(master_df.columns)}")
        return master_df

    # Remove duplicates based on unique combination of key fields
    # Create a unique identifier using multiple fields to detect true duplicates
    duplicate_cols = ['SPP', 'call_type']
    
    # Add temporal and frequency information if available for more robust deduplication
    if 'Begin Time (s)' in master_df.columns:
        duplicate_cols.append('Begin Time (s)')
    if 'End Time (s)' in master_df.columns:
        duplicate_cols.append('End Time (s)')
    if 'Low Freq (Hz)' in master_df.columns:
        duplicate_cols.append('Low Freq (Hz)')
    if 'High Freq (Hz)' in master_df.columns:
        duplicate_cols.append('High Freq (Hz)')
    
    # Track duplicates before removal
    initial_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=duplicate_cols, keep='first')
    final_count = len(master_df)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate call(s) based on {duplicate_cols}")

    # 1. Count by Species and Type
    stats = master_df.groupby(['SPP', 'call_type']).size().reset_index(name='Count')

    # 2. Calculate Average Duration (useful for seeing if 'Chirps' are actually short)
    if 'Begin Time (s)' in master_df.columns and 'End Time (s)' in master_df.columns:
        master_df['duration'] = master_df['End Time (s)'] - master_df['Begin Time (s)']
        avg_durations = master_df.groupby(['SPP', 'call_type'])['duration'].mean().reset_index(name='Avg_Dur_s')
        stats = pd.merge(stats, avg_durations, on=['SPP', 'call_type'])

    # Sort by the most frequent calls first
    stats = stats.sort_values(by='Count', ascending=False)
    
    print(f"\nTotal unique calls analyzed: {final_count}")
    print(f"Unique species found: {master_df['SPP'].nunique()}")
    print(f"Unique call types found: {master_df['call_type'].nunique()}")
    
    return stats

# Usage:
results = analyze_call_distribution('/Users/ariel/bioacoustics/FFL-Annotations/Data')
print("\n=== Call Distribution Statistics ===")
print(results)