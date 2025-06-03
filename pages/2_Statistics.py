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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Statistics")
st.title("Statistics")

st.markdown('''
    
    Once all of the features are inputted in (see Dataset page for examples), statistics are calculated for each of the summary features. 
    \nAt the time of the creation of this site, there were visually some differences between harmonic entropy with greater entropy in the 2010s, which can be seen in the polar plots below. Aggregates of features across decades using the normalized feature value for each feature showed that there was a significantly greater harmonic variety and harmonic entropy in 2010s songs, and a significantly greater chroma entropy in 1980s songs. For statistical tests, t-tests with significance at p<0.05 was used. 
    For rock songs, it is common for there to be one time signature throughout the piece and for the key to stay stable or change relatively few times, as compared to genres such as classical music. Listeners are able to hear a lot of these features such as a key change and experienced musicians or listeners could also identify section breaks in a song. However, identifying specific sub-key changes or chord changes can be challenging to identify and statistically differentiate. This analysis shows the results of the computational comparison of these features between the two time periods of interest.
''')

feat_path = Path(__file__).parent / "features.pkl"
#with open("/Users/rachelfox/Downloads/song_feature_data.pkl", "rb") as f:
with open(feat_path, "rb") as f:
    reference_data = pickle.load(f)

summ_path = Path(__file__).parent / "summary_features.pkl"
#with open("/Users/rachelfox/Downloads/song_summary_data.pkl", "rb") as f:
with open(summ_path, "rb") as f:
    summary_df = pickle.load(f)

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
            
    This is what the summary features look like projected for each decade grouping.
            
''')

plot_multiple_fingerprints(summary_df.loc[summary_df["Decade"]=="1980s", :])
plot_multiple_fingerprints(summary_df.loc[summary_df["Decade"]=="2010s", :])

from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

st.markdown('''
            
    This is how the features directly compare.
            
''')

def plot_song_feature_bars(df, feature_cols, normalize=True):
    # Normalize features (0-1 scale)
    data = df.loc[:, feature_cols]
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=feature_cols, index=df.index)

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
