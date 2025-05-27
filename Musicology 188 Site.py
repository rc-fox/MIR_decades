import streamlit as st
from pathlib import Path
import IPython.display as ipd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa, librosa.display
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import SpectralEmbedding
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
import soundfile as sf
from collections import Counter
import string
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# conda activate /Users/rachelfox/opt/anaconda3/envs/medsam
# streamlit run Downloads/MusicWebApp/'Musicology 188 Site'.py

### functions
# MIR pipeline
# functions for the music information retrieval analysis where different songs can be inputted

st.set_page_config(page_title="About")

st.title("About")

# tempo
def get_tempo(y, sr):
    # get the tempo of the song
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Tempo: {round(tempo[0], 2)} BPM")
    return round(tempo[0], 2)

# chromagram (harmony)
def get_chroma(y, sr):
    # harmony features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

# key
def get_best_key(chroma_segment):
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    # Get best key (better than just raw max)
    notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 
            'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    scores = []
    for i in range(12):
        major_corr = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_segment)[0, 1]
        minor_corr = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_segment)[0, 1]
        if np.isnan(major_corr): major_corr = -1
        if np.isnan(minor_corr): minor_corr = -1
        scores.append((major_corr, 'major', notes[i]))
        scores.append((minor_corr, 'minor', notes[i]))
    best = max(scores, key=lambda x: x[0])
    return best  # (correlation, mode, root)


def window_key_change(chroma, threshold=0.7):
    keychange = 0
    time_step = librosa.time_to_frames(30)
    print(time_step)
    print(len(chroma[1]))
    keywindows = []
    corrs = []
    start = 0
    times = []
    print(start+time_step)
    while (start+time_step < len(chroma[1])):
        segment = chroma[:, start:start+time_step]
        mean_feat = np.median(segment, axis=1)
        corr, mode, root = get_best_key(mean_feat)
        detected_key = (root, mode)
        keywindows.append(detected_key)
        corrs.append(corr)
        print("Key for time ", librosa.frames_to_time(start), "-", librosa.frames_to_time(start+time_step), "s: ", detected_key, " corr ", corr)
        times.append(start)
        start = start+time_step
    
    overallkey = max(Counter(keywindows), key=Counter(keywindows).get)
    print(overallkey)
    for i, t in enumerate(times):
        print(keywindows[i], corrs[i])
        if (keywindows[i] != overallkey and float(corrs[i])>threshold):
            print("Key change at:", librosa.frames_to_time(t))
            keychange += 1
    
    print("Overall key ", overallkey)
    return overallkey, keychange

def assign_section_labels(chroma, bounds, threshold=0.85):
    segment_feats = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i+1]
        segment = chroma[:, start:end]
        mean_feat = np.median(segment, axis=1)
        segment_feats.append(mean_feat)


    segment_feats = np.array(segment_feats)
    sim_matrix = cosine_similarity(segment_feats)

    # Assign section labels
    labels = [None] * len(segment_feats)
    label_names = list(string.ascii_uppercase)
    label_idx = 0

    for i in range(len(segment_feats)):
        if labels[i] is not None:
            continue
        labels[i] = label_names[label_idx]
        for j in range(i + 1, len(segment_feats)):
            if labels[j] is None and sim_matrix[i, j] > threshold:
                labels[j] = label_names[label_idx]
        label_idx += 1

    return labels


# get sections based on power spectrum
def get_bounds(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    X = chroma.T

    best_k = 3
    best_score = -1

    for k in range(3, 20):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    bounds = librosa.segment.agglomerative(chroma, best_k)
    bounds = np.append(bounds, chroma.shape[1])
    
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    labels = assign_section_labels(chroma, bounds)

    for i, label in enumerate(labels):
        print(f"Section {label}: {bound_times[i]:.2f}s – {bound_times[i+1]:.2f}s")

    print("Sections ", best_k)
    return bound_times, best_k

import tempfile
def getchords(file):
    # Get chord activations using a CNN
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3.write(uploaded_file.getbuffer())
        tmp_mp3_path = tmp_mp3.name

    y, sr = librosa.load(tmp_mp3_path, sr=44100, mono=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name
        sf.write(wav_path, y, sr)

    proc = CNNChordFeatureProcessor()
    activations = proc(wav_path)

    # Decode activations into chord labels using CRF
    crf = CRFChordRecognitionProcessor()
    chords = crf(activations)

    # chords is a list of [start_time, end_time, chord_label]
    start_times = []
    end_times = []
    chord_names = []
    for chord in chords:
        print(f"{chord[0]:.2f}–{chord[1]:.2f} sec: {chord[2]}")
        if chord[2] != "N":
            start_times.append(chord[0])
            end_times.append(chord[1])
            chord_names.append(chord[2])

    length_of_chord = np.subtract(end_times,start_times)
    length_of_chord = [round(x, 2) for x in length_of_chord]

    return length_of_chord, chord_names

def summarize_features(row):
    chroma = row["Chroma"]
    
    # Harmony: how evenly energy is distributed across 12 chroma bins
    chroma_entropy = -np.sum((chroma.mean(axis=1) + 1e-6) * np.log(chroma.mean(axis=1) + 1e-6))
    
    # Rhythm/Tempo: use raw tempo and number of sections
    rhythmic_complexity = row["Tempo"] * row["Segments"]
    
    # Key stability: fewer key changes = more stable
    key_stability = 1 / (1 + row["Number Key Change"])
    
    # Structural complexity: std deviation of section lengths
    structural_variability = np.std(row["Section Lengths"])

    # Harmonic complexity: how many unique chords and how frequently they change
    unique_chords = len(set(row["Chord Names"]))
    harmonic_change_rate = len(row["Chord Names"]) / sum(row["Chord Lengths"])  # chords per second

    # Harmonic variety: entropy over chord distribution
    from collections import Counter
    chord_counts = np.array(list(Counter(row["Chord Names"]).values()))
    chord_probs = chord_counts / chord_counts.sum()
    harmonic_entropy = -np.sum(chord_probs * np.log(chord_probs + 1e-6))

    # Harmonic character: major/minor - all major is 1, all minor is 0
    ismaj=0
    for x in range(0, len(row['Chord Names'])):
        ismaj += (row["Chord Names"][x][-3:] == "maj")
    character = ismaj / len(row['Chord Names'])
    
    return pd.Series({
        "Chroma Entropy": chroma_entropy,
        "Rhythmic Complexity": rhythmic_complexity,
        "Key Stability": key_stability,
        "Structural Variability": structural_variability,
        "Harmonic Variety": unique_chords,
        "Harmonic Change Rate": harmonic_change_rate,
        "Harmonic Entropy": harmonic_entropy,
        "Major Character": character
    })



st.header("Musicology 188 Project")
st.subheader("Analyzing harmonic attributes of rock songs in multiple time periods")

st.markdown('''
    Welcome to my Musicology 188 project site: Analyzing Harmonic Attributes of Rock Songs in Multiple Time Periods.
    \nThis project is created by Rachel Fox in Spring 2025. This is a music information retrieval project related to rock music.
            
    \nThis project uses computational analysis to analyze components from an imported MP3 file of a song, with components including signal analysis, feature extraction, beat tracking, and clustering. This can produce figures and detailed descriptions of what makes up the audio file. I will be using this as a computational exploration of music as well as a comparison of multiple popular songs from different time periods, to see how components of songs have evolved. I also have created a pipeline where the user would be able to input an mp3 file of the song of their choice and that incorporates information from the new song into the model to how this fits into the current dataset.
    In the field of digital musicology, there are different techniques and formats to analyze music, including music information retrieval, optical music recognition, music encoding, digital archives, and metadata processing. There are many digital archives that exist, such as the global jukebox or archives for individual styles and regions of music, and more recently the individual songs are being analyzed for specific features. With the use of machine learning and audio recognition tools, it becomes more feasible to analyze large amounts of songs in a consistent manner. Thus, there are many areas for contributions in music information retrieval. 
    I am examining rock music, which has not been extensively analyzed using computational techniques, and adding a comparison of two time points of popular music will increase understanding of how rock music has evolved over time.  

''')
