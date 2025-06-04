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
import soundfile as sf
from collections import Counter
import string
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# This page is the user input and comparison of their song to the dataset atttributes, this also contains the code for generating the data for the dataset seen on the other page

### functions
# MIR pipeline
# functions for the music information retrieval analysis where different songs can be inputted

st.set_page_config(page_title="Add Songs")
st.title("Add Songs")

st.markdown('''
            
    Now it's your turn! Upload a mp3 file and we will create a dynamic analysis of your song mixed with our dataset. Please have the song input as decade - songname (example: 1980s - Back In Black).
    \nThis page will take the uploaded song of your choice and compute all of the features and summary features dictating rhythmic complexity, harmony, and structure. After displaying the results, the analysis between this song and the 1980s and 2000s dataset is conducted. The metric used is the difference between each summary feature for the new song and the mean for that feature for each of the 1980s and 2010s. Users will be able to see which time period the given song is most similar to for each summary feature, outputted as text on this page. 
            
''')

feat_path = Path(__file__).resolve().parents[1] / "song_feature_data.pkl"
with open(feat_path, "rb") as f:    
    reference_data = pickle.load(f)

summ_path = Path(__file__).resolve().parents[1] / "song_summary_data.pkl"
with open(summ_path, "rb") as f:    
    summary_df = pickle.load(f)
    
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

# get presence of key change
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

# get sections
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


# get bounds for sections based on power spectrum
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

# get chords using librosa
# Define templates for major and minor chords (root position only)
CHORD_TEMPLATES = {}
pitches = ['C', 'C#', 'D', 'D#', 'E', 'F',
           'F#', 'G', 'G#', 'A', 'A#', 'B']

major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

for i, pitch in enumerate(pitches):
    CHORD_TEMPLATES[f"{pitch}:maj"] = np.roll(major_template, i)
    CHORD_TEMPLATES[f"{pitch}:min"] = np.roll(minor_template, i)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def estimate_chord(chroma_vector):
    scores = {
        name: cosine_sim(chroma_vector, template) for name, template in CHORD_TEMPLATES.items()
    }
    return max(scores, key=scores.get)

def getchords(y, sr):
    # Compute chroma
    hop_length = 11025
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Estimate chord at each time frame
    chords = []
    for i in range(chroma.shape[1]):
        chord = estimate_chord(chroma[:, i])
        chords.append((times[i], chord))

    # Group consecutive identical chords
    start_times = []
    end_times = []
    chord_names = []

    prev_chord = chords[0][1]
    start_time = chords[0][0]
    for i in range(1, len(chords)):
        current_chord = chords[i][1]
        current_time = chords[i][0]
        if current_chord != prev_chord:
            end_time = current_time
            start_times.append(start_time)
            end_times.append(end_time)
            chord_names.append(prev_chord)
            start_time = current_time
            prev_chord = current_chord

    start_times.append(start_time)
    end_times.append(times[-1])
    chord_names.append(prev_chord)

    # Calculate durations
    length_of_chord = np.round(np.subtract(end_times, start_times), 2)

    return length_of_chord, chord_names

# create summary features data frame
def summarize_features(row):
    chroma = row["Chroma"]
    
    # Chromagram Entropy: how evenly energy is distributed across 12 chroma bins
    chroma_entropy = -np.sum((chroma.mean(axis=1) + 1e-6) * np.log(chroma.mean(axis=1) + 1e-6))
    
    # Rhythmic complexity: use raw tempo and number of sections
    rhythmic_complexity = row["Tempo"] * row["Segments"]
    
    # Key stability: fewer key changes = more stable
    key_stability = 1 / (1 + row["Number Key Change"])
    
    # Structural variability: std deviation of section lengths
    structural_variability = np.std(row["Section Lengths"])

    # Harmonic variety: unique chords
    # Harmonic complexity: how many unique chords and how frequently they change
    unique_chords = len(set(row["Chord Names"]))
    harmonic_change_rate = len(row["Chord Names"]) / sum(row["Chord Lengths"])  # chords per second

    # Harmonic entropy: entropy over chord distribution
    from collections import Counter
    chord_counts = np.array(list(Counter(row["Chord Names"]).values()))
    chord_probs = chord_counts / chord_counts.sum()
    harmonic_entropy = -np.sum(chord_probs * np.log(chord_probs + 1e-6))

    # Major character: major/minor - all major is 1, all minor is 0
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




# Upload song from user and complete analysis and graph
uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])
import os

decade_labels = []
song_names = []
chroma_vectors = []
tempo_vectors = []
key_vectors = []
num_sections = []
section_lengths = []
keychanges = []
chord_lengths = []
chord_names = []

if uploaded_file is not None:
    with st.spinner("Processing..."):
        print("Uploaded file")
        decade_labels.append(os.path.basename(uploaded_file.name)[:5])
        song_names.append(os.path.basename(uploaded_file.name)[8:-4])
        y, sr = librosa.load(uploaded_file)
        chroma = get_chroma(y, sr)
        chroma_vectors.append(chroma)
        tempo_vectors.append(get_tempo(y, sr))
        boundar, numsect = get_bounds(y, sr)
        overallkey, keychange = window_key_change(chroma)
        key_vectors.append(overallkey)
        keychanges.append(keychange)
        num_sections.append(numsect)
        section_lengths.append([boundar[k]-boundar[k-1] for k in range(1, boundar.size)])
        chord_length, chord_name = getchords(y, sr)
        chord_lengths.append(chord_length)
        chord_names.append(chord_name)
    st.success("Processing complete!")

    featuredf_indiv = pd.DataFrame({
            "Decade": decade_labels,
            "Song Name": song_names, 
            "Chroma": chroma_vectors,
            "Tempo": tempo_vectors,
            "Key": key_vectors,
            "Number Key Change": keychanges,
            "Segments": num_sections,
            "Section Lengths": section_lengths,
            "Chord Lengths": chord_lengths,
            "Chord Names": chord_names
        })
    
    summary_stats = featuredf_indiv.apply(summarize_features, axis=1)
    if isinstance(summary_stats, pd.Series):
        summary_stats = summary_stats.to_frame().T

    summary_df_indiv = pd.concat([featuredf_indiv.reset_index(drop=True), summary_stats.reset_index(drop=True)], axis=1)


    feature_cols = ["Chroma Entropy", "Rhythmic Complexity", "Key Stability", "Structural Variability", "Harmonic Variety", "Harmonic Change Rate", "Harmonic Entropy", "Major Character"]
    data_indiv = summary_df_indiv.loc[:, feature_cols]
    scaler = MinMaxScaler()
    scaler.fit(summary_df[feature_cols])  # summary df is the full dataset

    st.markdown('''This is what the individual features look like
                ''')
    st.dataframe(featuredf_indiv[["Decade", "Song Name", "Tempo", "Key", "Number Key Change", "Segments", "Section Lengths", "Chord Lengths", "Chord Names"]])

    st.markdown('''This is what the summary features look like
                ''')
    st.dataframe(summary_df_indiv[feature_cols])

    min_vals = summary_df[feature_cols].min()
    max_vals = summary_df[feature_cols].max()
    data_indiv = (data_indiv[feature_cols] - min_vals) / (max_vals - min_vals)

    data_indiv["Decade"] = summary_df_indiv["Decade"]
    data_indiv["Song"] = summary_df_indiv["Song Name"]
    df_long_indiv = pd.melt(data_indiv, id_vars=["Song", "Decade"], var_name="Feature", value_name="Value")

    fig, ax = plt.subplots()
    p = sns.catplot(data=df_long_indiv, x="Feature", y="Value", kind="bar", hue="Decade", palette="Set2")
    p.fig.suptitle("Harmonic and Structural Features of Uploaded Song", fontsize=14)
    p.set_axis_labels("Feature", "Feature Value (Normalized)")
    p.set_xticklabels(rotation=90)
    p.tight_layout()
    st.pyplot(p.fig)

    st.markdown('''How does this compare to what we already have?
                ''')

    # compare new measures with existing dataset measures
    for measure in feature_cols:
        g1 = np.mean(summary_df_indiv[measure].values)
        g2 = np.mean(summary_df[summary_df["Decade"] == "1980s"][measure].values)
        g3 = np.mean(summary_df[summary_df["Decade"] == "2010s"][measure].values)
        print("G1 G2 G3")
        print(g1, " \n", g2, "\n", g3)

        print(g1-g2)
        message1 = f"{measure}: {g1-g2:.2f} units ({(g1-g2)/g2*100:.2f}%) different than our 1980s dataset and {g1-g3:.2f} units ({(g1-g3)/g3*100:.2f}%) different than our 2010s dataset"
        st.markdown(message1)

        if abs(g1-g2) > abs(g1-g3):
            message2 = f"{measure} for this inputted song is more similar to the 2010s dataset."
            st.markdown(message2)
        else:
            message2 = f"{measure} for this inputted song is more similar to the 1980s dataset."
            st.markdown(message2)

    # if you want to add to the data frame of songs
    #if data_indiv.loc[0, "Decade"] == "1980s" or data_indiv.loc[0, "Decade"] == "2010s":
    #    summary_df_updated = pd.concat([summary_df, summary_df_indiv], axis=0).reset_index()
    #     summ_path = Path(__file__).resolve().parents[1] / "song_summary_data.pkl"
    #    with open(summ_path, "wb") as f:
    #        pickle.dump(summary_df_updated, f)
    #    st.markdown("This song has now been added to our dataset since it was from one of our decades of interest!")