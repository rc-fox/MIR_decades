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
# streamlit run Downloads/musicwebapp.py

### functions
# MIR pipeline
# functions for the music information retrieval analysis where different songs can be inputted

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
    Welcome to my Musicology 188 project site: analyzing harmonic attributes of rock songs in multiple time periods!
    This project is created by Rachel Fox in Spring 2025. This is a music information retrieval project related to rock music.
            
    This project will be using computational analysis and machine learning to analyze components from an imported MP3 file of a song, with computational components including signal analysis, feature extraction, beat tracking, and clustering. This can produce figures and detailed descriptions of what makes up the audio file. I will be using this as a computational exploration of music as well as a comparison of multiple popular songs from different time periods, to see how components of songs have evolved. After this analysis, I will create a pipeline where the user would be able to input an mp3 file of the song of their choice and this pipeline would be able to incorporate this into the model to analyze time of release and how this fits into the current dataset.

''')

st.markdown('''
    
    This is an example of data that is added into the feature list

''')
with open("/Users/rachelfox/Downloads/song_feature_data.pkl", "rb") as f:
    reference_data = pickle.load(f)
print(type(reference_data))
#st.dataframe(reference_data.head(10))
st.dataframe(reference_data.sample(n=10))

st.markdown('''
            
    This is an example of data that is created in the summary list of features
            
''')

with open("/Users/rachelfox/Downloads/song_summary_data.pkl", "rb") as f:
    summary_df = pickle.load(f)
#st.dataframe(summary_df.head(10))
st.dataframe(summary_df.sample(n=10))
st.markdown(f"There are currently {summary_df.shape[0]} songs in this dataset.")

st.markdown('''
            
    This is what principal components created from the summary features look like
            
''')
X = summary_df[["Chroma Entropy", "Rhythmic Complexity", "Key Stability", "Structural Variability", "Harmonic Variety", "Harmonic Change Rate", "Harmonic Entropy", "Major Character"]]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
summary_df["PCA1"] = X_pca[:, 0]
summary_df["PCA2"] = X_pca[:, 1]
pcadf = X_pca[:, :3]
loadings = pd.DataFrame(pca.components_[:3, :], columns=X.columns, index=[f'PC{i+1}' for i in range(3)])
fig, ax = plt.subplots()
sns.heatmap(loadings.T, vmin=-1, vmax=1, cmap="hot")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.scatterplot(data=summary_df, x="PCA1", y="PCA2", hue="Decade", s=80)
plt.title("PCA")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)


def plot_multiple_fingerprints(df):
    features = ["Chroma Entropy", "Key Stability", "Structural Variability", "Harmonic Variety", "Harmonic Change Rate", "Harmonic Entropy", "Major Character"]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    maxes = [df[f].max() for f in features]
    print(maxes)

    for i in range(0, len(df)):
        row = df.iloc[i]
        values = [row[f] for f in features] 
        values = [values[i]/maxes[i] for i in range(len(values))] # normalize for better visualization
        values += values[:1]
        random_number = np.random.randint(0, 16777215)
        hex_number = str(hex(random_number))
        hex_number = '#' + hex_number[2:].zfill(6)
        plt.polar(angles, values, linewidth=0.5, color=hex_number)
        plt.fill(angles, values, color=hex_number, alpha=0.25)
        plt.xticks(angles[:-1], labels=features)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], labels=[])
        plt.title(row["Decade"], size=10, pad=10)

    plt.tight_layout()
    st.pyplot(fig)

print(len(summary_df.loc[summary_df["Decade"]=="1980s", :]))

st.markdown('''
            
    This is what the summary features look like projected for each decade grouping
            
''')

plot_multiple_fingerprints(summary_df.loc[summary_df["Decade"]=="1980s", :])
plot_multiple_fingerprints(summary_df.loc[summary_df["Decade"]=="2010s", :])

from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

st.markdown('''
            
    This is how the features directly compare!
            
''')

def plot_song_feature_bars(df, feature_cols, normalize=True):
    # Normalize features (0-1 scale)
    data = df.loc[:, feature_cols].copy()
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=feature_cols)

    data["Decade"] = df["Decade"]
    data["Song"] = df["Song Name"]
    df_long = pd.melt(data, id_vars=["Song", "Decade"], var_name="Feature", value_name="Value")
    fig, ax = plt.subplots()
    p = sns.catplot(data=df_long, x="Feature", y="Value", kind="bar", hue="Decade", palette="Set2", hue_order=["1980s", "2010s"])
    sns.move_legend(p, 'upper right')
    p.fig.suptitle("Aggregate of Features Across Decades", fontsize=14)
    p.set_axis_labels("Feature", "Feature Value (Normalized)")
    p.set_xticklabels(rotation=90)
    p.tight_layout()
    st.pyplot(p.fig)

feature_cols = ["Chroma Entropy", "Rhythmic Complexity", "Key Stability", "Structural Variability", "Harmonic Variety", "Harmonic Change Rate", "Harmonic Entropy", "Major Character"]
plot_song_feature_bars(summary_df, feature_cols)

# statistical analysis
st.markdown('''
            
    We can see if any of our measures are statistically significantly different.
            
''')
dft = pd.DataFrame()
for measure in feature_cols:
    group_1 = summary_df[summary_df["Decade"] == "1980s"][measure]
    group_2 = summary_df[summary_df["Decade"] == "2010s"][measure]

    t_stat, p_value = ttest_ind(group_1, group_2, equal_var=False)
    print(measure, "T stat:", t_stat, "P-value:", p_value)
    dft = pd.concat([dft, pd.DataFrame([{"Measure": measure, "T-statistic":t_stat, "P-value":p_value}])], ignore_index=True)
st.dataframe(dft)
print(f"Length of input data {summary_df.shape}")



st.markdown('''
            
    Now it's your turn! Upload a mp3 file and we will create a dynamic analysis of your song mixed with our dataset. Please have the song input as decade - songname (example: 1980s - Back In Black).
            
''')

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
        chord_length, chord_name = getchords(uploaded_file)
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

    min_vals = summary_df[feature_cols].min()
    max_vals = summary_df[feature_cols].max()
    data_indiv = (data_indiv[feature_cols] - min_vals) / (max_vals - min_vals)
    #data_indiv = data_indiv.clip(0, 1)

    data_indiv["Decade"] = summary_df_indiv["Decade"]
    data_indiv["Song"] = summary_df_indiv["Song Name"]
    df_long_indiv = pd.melt(data_indiv, id_vars=["Song", "Decade"], var_name="Feature", value_name="Value")

    #st.write("Summary DF shape:", summary_df_indiv.shape)
    #st.write("Data shape:", data_indiv.shape)
    #st.write("Melted DataFrame (df_long_indiv):", df_long_indiv)

    fig, ax = plt.subplots()
    p = sns.catplot(data=df_long_indiv, x="Feature", y="Value", kind="bar", hue="Decade", palette="Set2")
    p.fig.suptitle("Harmonic and Structural Features of Uploaded Song", fontsize=14)
    p.set_axis_labels("Feature", "Feature Value (Normalized)")
    p.set_xticklabels(rotation=90)
    p.tight_layout()
    st.pyplot(p.fig)

    st.markdown('''How does this compare to what we already have?
                ''')

    for measure in feature_cols:
        g1 = np.mean(summary_df_indiv[measure].values)
        g2 = np.mean(summary_df[summary_df["Decade"] == "1980s"][measure].values)
        g3 = np.mean(summary_df[summary_df["Decade"] == "2010s"][measure].values)
        print("G1 G2 G3")
        print(g1, " \n", g2, "\n", g3)

        print(g1-g2)
        message1 = f"{measure}: {g1-g2:.2f} units ({(g1-g2)/g2*100:.2f}%) different than our 1980s dataset and {g1-g3:.2f} units ({(g1-g3)/g3*100:.2f}%) different than our 2010s dataset"
        st.markdown(message1)

    if data_indiv.loc[0, "Decade"] == "1980s" or data_indiv.loc[0, "Decade"] == "2010s":
        summary_df_updated = pd.concat([summary_df, summary_df_indiv], axis=0)
        with open("/Users/rachelfox/Downloads/song_summary_data.pkl", "wb") as f:
            pickle.dump(summary_df_updated, f)
        st.markdown("This song has now been added to our dataset since it was from one of our decades of interest!")