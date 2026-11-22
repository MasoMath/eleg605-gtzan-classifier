import os
import shutil
import kagglehub
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed


def download_kaggle_data(output_dir="./GTZAN"):
    path = kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification",
        output_dir=output_dir
    )

    print("Path to dataset files:", path)

def verify_corrupt_file(data_dir="./GTZAN/Data/genres_original"):
    corrupt_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    y, sr = librosa.load(file_path, duration=30)
                except Exception as e:
                    print(f"[ERROR] Corrupt file detected: {file_path}")
                    print(f"\t\tException: {e}")
                    print(f"\t\t{type(e)}")
                    corrupt_files.append(file_path)

    print(f"\nVerification complete. Found {len(corrupt_files)} unreadable/corrupt files.")

def verify_gtzan(
        data_dir="./GTZAN/Data/genres_original", s_lb = 29.9, s_ub=30.1):
    EXPECTED_GENRES = {
        "blues", "classical", "country", "disco", "hiphop", 
        "jazz", "metal", "pop", "reggae", "rock"
    }
    EXPECTED_SR = 22050
    EXPECTED_CHANNELS = 1

    
    
    # Acceptable frame count tolerance bounds (29.5 to 30.5 seconds)
    EXPECTED_MIN_FRAMES = int(EXPECTED_SR * s_lb)  # 651,475 frames
    EXPECTED_MAX_FRAMES = int(EXPECTED_SR * s_ub)  # 671,525 frames

    actual_min_frame = EXPECTED_MAX_FRAMES
    actual_max_frame = EXPECTED_MIN_FRAMES

    issues = {"corrupt": [], "wrong_specs": [], "structural": [], "silent": []}

    # 1. Structural Check: Verify folders exist
    subdirs = {d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))}
    if not EXPECTED_GENRES.issubset(subdirs):
        issues["structural"].append(f"Missing one or more expected genre folders.")

    for genre in EXPECTED_GENRES:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.exists(genre_path):
            continue
        
        files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        if len(files) != 100:
            issues["structural"].append(f"Folder '{genre}' contains {len(files)} files instead of 100.")

        for file in files:
            file_path = os.path.join(genre_path, file)
            
            try:
                # 2. Fast metadata inspection using soundfile
                info = sf.info(file_path)
                
                # 3. Check sample rate and channels
                if info.samplerate != EXPECTED_SR or info.channels != EXPECTED_CHANNELS:
                    issues["wrong_specs"].append(f"{file_path} -> SR: {info.samplerate}, Channels: {info.channels}")
                
                # 4. Check frame count against tolerance window
                if not (EXPECTED_MIN_FRAMES <= info.frames <= EXPECTED_MAX_FRAMES):
                    issues["wrong_specs"].append(f"{file_path} -> Out-of-bounds frame count: {info.frames}")
                    if info.frames < actual_min_frame: actual_min_frame = info.frames
                    if info.frames > actual_max_frame: actual_max_frame = info.frames
                        

                # 5. Load full audio via librosa to check for silent/zero-padded files
                y, sr = librosa.load(file_path, sr=None)
                if np.all(y == 0) or np.max(np.abs(y)) < 1e-5:
                    issues["silent"].append(file_path)

            except Exception as e:
                # Catches truly corrupt/unreadable files like jazz.00054.wav
                issues["corrupt"].append((file_path, str(e)))
                print(f'Exception: {e}')

    # --- Summary Report ---
    print("\n" + "="*40)
    print("       GTZAN VERIFICATION SUMMARY")
    print("="*40)

    total_issues = sum(len(v) for v in issues.values())

    print(
        f'sampling bounds at SR = {EXPECTED_SR}',
        f'\n\tseconds : [{s_lb}, {s_ub}]',
        f'\n\tframes : [{EXPECTED_MIN_FRAMES}, {EXPECTED_MAX_FRAMES}]\n'
    )

    print(f'actual max : {actual_max_frame}')
    print(f'actual min : {actual_min_frame}')

    for category, items in issues.items():
        print(f"\n{category.capitalize()} issues found: {len(items)}")
        # Print up to the first 5 examples for each category if any exist
        for item in items[:5]:
            print(f"  - {item}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more.")

    print("\n" + "="*50)
    if total_issues == 0:
        print("Result: Dataset is pristine and fully valid!")
    else:
        print(f"Result: Found {total_issues} total items requiring attention.")

def extract_gtzan_features(
        mp3_path, chunk_dur=30.0, sr=22050, label=None):
    """
        Extracts GTZAN compatible features from an audio file
        broken down into chunks. Skips corrupted audio tracks.
    """
    mp3_path_str = str(mp3_path)
    
    try:
        y, sr = librosa.load(mp3_path_str, sr=sr, mono=True)
    except Exception as e:
        print(
            "\n[Skipped] Corrupted or unrecognised audio format:",
            f"'{Path(mp3_path_str).name}' ({e})")
        return pd.DataFrame() 
        
    total_samples = len(y)
    chunk_samples = int(chunk_dur * sr) if chunk_dur else total_samples
    features_list = []
    
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = min(start_sample + chunk_samples, total_samples)
        y_chunk = y[start_sample:end_sample]
        
        if len(y_chunk) < 2048: continue
        
        try:
            chroma_stft = librosa.feature.chroma_stft(y=y_chunk, sr=sr)
            rms = librosa.feature.rms(y=y_chunk)
            spec_cent = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y_chunk, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y_chunk)
            
            y_harmonic, y_percussive = librosa.effects.hpss(y_chunk)
            tempo, _ = librosa.beat.beat_track(y=y_chunk, sr=sr)
            tempo_val = 0
            if isinstance(tempo, np.ndarray):
                tempo_val = float(tempo[0])
            else:
                tempo_val = float(tempo)
            
            mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=20)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_vars = np.var(mfccs, axis=1)
        except Exception as e:
            print(
                f"[Warning] Error extracting features for chunk in",
                f"{Path(mp3_path_str).name}: {e}")
            continue
            
        row_data = {
            'filename': Path(mp3_path_str).name, 
            'length': len(y_chunk),
            'chroma_stft_mean': np.mean(chroma_stft),
            'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rms),
            'rms_var': np.var(rms),
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_centroid_var': np.var(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_bandwidth_var': np.var(spec_bw),
            'rolloff_mean': np.mean(rolloff),
            'rolloff_var': np.var(rolloff),
            'zero_crossing_rate_mean': np.mean(zcr),
            'zero_crossing_rate_var': np.var(zcr),
            'harmony_mean': np.mean(y_harmonic),
            'harmony_var': np.var(y_harmonic),
            'perceptr_mean': np.mean(y_percussive),
            'perceptr_var': np.var(y_percussive),
            'tempo': tempo_val,
        }
        
        for i in range(1, 21):
            row_data[f'mfcc{i}_mean'] = mfcc_means[i-1]
            row_data[f'mfcc{i}_var'] = mfcc_vars[i-1]
            
        row_data['label'] = label if label is not None else np.nan
        features_list.append(row_data)
        
    if not features_list:
        return pd.DataFrame()
        
    df_chunked = pd.DataFrame(features_list)
    
    cols = [
        c
        for c in df_chunked.columns
        if c not in ['filename', 'label']
    ]
    ordered_cols = ['filename'] + cols + ['label']
    return df_chunked[ordered_cols]


def extract_features_batch(
        mp3_paths, labels=None, chunk_dur=30.0, sr=22050, n_jobs=-1):
    """
        Extracts features for a list of MP3 files in parallel,
        filtering out failed/corrupted loads.
    """
    if labels is not None:
        if len(labels) != len(mp3_paths):
            raise ValueError(
                f"Length of labels ({len(labels)}) must match length",
                f"of mp3_paths ({len(mp3_paths)}).")
    else:
        labels = [None] * len(mp3_paths)
        
    print(
        f"Processing {len(mp3_paths)} audio files across all CPU cores...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_gtzan_features)(
            path, chunk_dur, sr, label=lbl
        ) for path, lbl in zip(mp3_paths, labels)
    )
    
    valid_results = [
        df
        for df in results
        if df is not None and not df.empty
    ]
    
    if not valid_results:
        print(
            "[Warning] No valid audio files were",
            "successfully processed in this batch.")
        return pd.DataFrame()
        
    master_df = pd.concat(valid_results, ignore_index=True)
    print(
        f"Extraction complete. Successfully processed",
        f"{len(valid_results)}/{len(mp3_paths)} files.",
        f"Master csv shape: {master_df.shape}")
    
    return master_df


def extract_gtzan_split_features(
        path_to_dir, chunk_dur=None, sr=22050, n_jobs=-1):
    """
        Scans the split directory (path_to_dir) for train, valid,
        and test folders, extracts GTZAN features for each audio file using
        extract_features_batch, and saves individual CSV feature datasets
        in the same directory.
        
        Parameters:
        - path_to_dir: Path to the processed directory containing
            train/valid/test folders and the manifest CSV.
        - chunk_dur: Duration of chunks in seconds. Set to None if
            processing pre-split 3s clips or wanting the full duration
            of 30s files as single rows.
        - sr: Sampling rate.
        - n_jobs: Number of parallel jobs for joblib
            (-1 uses all CPU cores).
    """
    base_path = Path(path_to_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")
            
    manifest_files = list(base_path.glob("gtzan_*_manifest.csv"))
    suffix_tag = ""
    
    if manifest_files:
        manifest_name = manifest_files[0].stem
        parts = manifest_name.split('_')
        if len(parts) >= 3:
            mode = parts[1]
            dataset_type = parts[2]
            suffix_tag = f"_{mode}_{dataset_type}"
            print(
                f"Detected configuration from manifest:",
                f"Mode={mode}, Dataset={dataset_type}")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = base_path / split
        if not split_dir.exists():
            print(
                f"[Info] Split folder '{split}'",
                f"not found in {base_path}. Skipping.")
            continue
            
        mp3_paths = []
        labels = []
        
        for genre_dir in split_dir.iterdir():
            if genre_dir.is_dir():
                genre_name = genre_dir.name
                all_songs = list(genre_dir.glob("*.wav"))
                all_songs += list(genre_dir.glob("*.mp3"))
                for audio_file in all_songs:
                    mp3_paths.append(str(audio_file))
                    labels.append(genre_name)
                    
        if not mp3_paths:
            print(f"[Warning] No audio files found inside {split_dir}")
            continue
            
        print(f"\n" + "="*30)
        print(
            f"Processing Split: {split.upper()}",
            f"({len(mp3_paths)} audio files)")
        print(f"="*30)
        
        df_features = extract_features_batch(
            mp3_paths=mp3_paths,
            labels=labels,
            chunk_dur=chunk_dur,
            sr=sr,
            n_jobs=n_jobs
        )

        csv_filename = ""
        if suffix_tag:
            csv_filename = f"{split}{suffix_tag}_features.csv"
        else:
            csv_filename = f"{split}_features.csv"
        csv_output_path = base_path / csv_filename
        
        df_features.to_csv(csv_output_path, index=False)
        print(f"Successfully saved feature dataset to: {csv_output_path}")



def summarize_split_distributions(path_to_dir):
    """
        Scans the directory for feature CSV files, computes the count 
        and percentage of each class label per split,
        and prints a formatted report.
    
        Parameters:
            - path_to_dir: Path to the directory containing the split
                feature CSV files.
    """
    base_path = Path(path_to_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")
        
    feature_files = list(base_path.glob("*_features.csv"))
    if not feature_files:
        print(
            f"[Warning] No feature CSV files matching",
            f"'*_features.csv' found in {base_path}")
        return
        
    summary_results = {}
    
    for file_path in feature_files:
        filename = file_path.name
        
        split_name = None
        if 'train' in filename.lower():
            split_name = 'train'
        elif 'valid' in filename.lower() or 'val' in filename.lower():
            split_name = 'valid'
        elif 'test' in filename.lower():
            split_name = 'test'
        else:
            continue
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[Error] Could not read {filename}: {e}")
            continue
            
        if 'label' not in df.columns:
            print(
                f"[Warning] 'label' column missing in {filename}.",
                f"Skipping summary for this file.")
            continue
            
        counts = df['label'].value_counts().sort_index()
        percentages = (
            df['label'].value_counts(normalize=True).sort_index() * 100
        ).round(2)
        
        split_summary_df = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })
        
        summary_results[split_name] = {
            'dataframe': split_summary_df,
            'total_samples': len(df)
        }
        
    for split, data in summary_results.items():
        print(f"\n" + "="*30)
        print(
            f"SPLIT SUMMARY: {split.upper()}",
            f"(Total Samples: {data['total_samples']})")
        print("="*30)
        print(data['dataframe'])
        print("=" * 30)
        
    return summary_results


def plot_spectrogram(mp3_path, sr=22050, absolute_scale=True, cmap='magma', oneFig=True):
    """
    Loads an audio file and plots both a standard spectrogram
    and a Mel-spectrogram.
    
    Args:
        absolute_scale (bool): If True, uses a fixed reference (ref=1.0) 
                               for natural energy levels (duller, -60dB scale).
                               If False, normalizes to peak (ref=np.max, bright).
    """
    mp3_path_str = str(mp3_path)
    
    try:
        y, sr = librosa.load(mp3_path_str, sr=sr, mono=True)
    except Exception as e:
        print(
            "\n[Skipped] Corrupted or unrecognised audio format:",
            f"'{Path(mp3_path_str).name}' ({e})")
        return
    
    if oneFig: plt.figure(figsize=(12, 8))
    
    # Choose reference based on scaling preference
    ref_val = 1.0 if absolute_scale else np.max
    
    # 1. Log-Frequency Spectrogram (STFT)
    if oneFig:
        plt.subplot(2, 1, 1)
    else:
        plt.figure(figsize=(10,4))
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=ref_val)
    
    img1 = librosa.display.specshow(
        S_db, sr=sr, x_axis='time', y_axis='log', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    title_suffix = "(Absolute Energy Scale)" if absolute_scale else "(Peak Normalized)"
    plt.title(f'Log-Frequency Spectrogram (STFT) {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # 2. Mel-Spectrogram
    if oneFig:
        plt.subplot(2, 1, 2)
    else:
        plt.figure(figsize=(10,4))
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=ref_val)
    
    img2 = librosa.display.specshow(
        mel_spectrogram_db, sr=sr,
        x_axis='time', y_axis='mel',
        fmax=8000, cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    
    plt.tight_layout()
    plt.show()

def get_split_summary(split_dict, metric="count"):
    """Combines train, test, and validation split dataframes into a single summary table.

    Args:
        split_dict (dict): The dictionary containing split dataframes.
        metric (str): The metric to display, either 'count' or 'percentage'.

    Returns:
        pd.DataFrame: DataFrame with genre labels as index and
            ['train', 'test', 'valid'] as columns.
    """
    # Map user input to the correct DataFrame column name
    col_name = "Count" if metric.lower() == "count" else "Percentage"

    # Extract the requested column from each split dataframe
    summary_data = {}
    for split_name in ["train", "test", "valid"]:
        if split_name in split_dict:
            summary_data[split_name] = split_dict[
                split_name]["dataframe"][col_name]

    # Build the consolidated DataFrame and enforce column order
    df_summary = pd.DataFrame(summary_data)[["train", "test", "valid"]]
        
    # Print to the console
    print(f"\n--- Dataset Split Summary: {metric.upper()} ---")
    print(df_summary.to_string())
    print("-" * 45)
    if metric == 'counts':
            counts = df_summary.sum()
            print(counts)
            print(counts / counts.sum())

    return df_summary

def plot_reference_style_spectrogram(mp3_path, sr=22050, ref=1.0, top_db=60.0):
    """
    Plots a Mel-spectrogram styled to match typical academic 
    and dataset references for GTZAN.
    """
    mp3_path_str = str(mp3_path)
    
    try:
        y, sr = librosa.load(mp3_path_str, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {mp3_path_str}: {e}")
        return
    
    plt.figure(figsize=(10, 4))
    
    # Compute Mel spectrogram using power
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    
    # Convert to dB with a strict dynamic range (top_db=60) 
    # and default absolute scaling (ref=1.0) or relative (ref=np.max)
    mel_spectrogram_db = librosa.power_to_db(
        mel_spectrogram, ref=ref, top_db=top_db
    )
    
    # Render using the 'magma' colormap (standard for audio papers)
    img = librosa.display.specshow(
        mel_spectrogram_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel', 
        fmax=8000, 
        cmap='magma'  # <--- Changes the color profile to match literature
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')
    plt.tight_layout()
    plt.show()