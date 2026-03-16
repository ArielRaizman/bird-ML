#!/usr/bin/env python3
"""
Algorithm to calculate the total amount of recording time in the bioacoustics dataset.
Scans all audio files (MP3 and WAV) and sums their durations.
"""

import os
from pathlib import Path
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from datetime import timedelta


def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds, or None if there's an error
    """
    try:
        ext = file_path.suffix.lower()
        if ext == '.mp3':
            audio = MP3(file_path)
            return audio.info.length
        elif ext == '.wav':
            audio = WAVE(file_path)
            return audio.info.length
        else:
            print(f"Unsupported format: {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def find_all_audio_files(base_dir):
    """
    Recursively find all audio files in a directory.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of Path objects for audio files
    """
    audio_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory does not exist: {base_dir}")
        return audio_files
    
    # Recursively search for audio files
    for ext in ['*.mp3', '*.wav', '*.MP3', '*.WAV']:
        audio_files.extend(base_path.rglob(ext))
    
    return audio_files


def calculate_total_recording_time(directories):
    """
    Calculate the total recording time across all audio files in given directories.
    
    Args:
        directories: List of directory paths to search
        
    Returns:
        Total duration in seconds
    """
    total_duration = 0
    file_count = 0
    
    print("Scanning for audio files...\n")
    
    for directory in directories:
        print(f"Searching in: {directory}")
        audio_files = find_all_audio_files(directory)
        
        for audio_file in audio_files:
            duration = get_audio_duration(audio_file)
            if duration is not None:
                total_duration += duration
                file_count += 1
                print(f"  [{file_count}] {audio_file.name}: {timedelta(seconds=int(duration))}")
    
    return total_duration, file_count


def format_duration(seconds):
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string with days, hours, minutes, seconds
    """
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    return ", ".join(parts)


def main():
    """Main function to calculate and display total recording time."""
    # Define directories to scan
    workspace_root = Path(__file__).parent
    directories_to_scan = [
        workspace_root / "FFL-Annotations" / "Audio"
    ]
    
    print("=" * 70)
    print("BIOACOUSTICS RECORDING TIME CALCULATOR")
    print("=" * 70)
    print()
    
    # Calculate total recording time
    total_seconds, file_count = calculate_total_recording_time(directories_to_scan)
    
    # Display results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {file_count}")
    print(f"Total recording time: {format_duration(int(total_seconds))}")
    print(f"Total seconds: {total_seconds:.2f}")
    print(f"Total minutes: {total_seconds / 60:.2f}")
    print(f"Total hours: {total_seconds / 3600:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
