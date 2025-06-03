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

st.set_page_config(page_title="Dataset")
st.title("Dataset")

st.markdown('''
    
    In this dataset, I inputted songs into Python using the library librosa and from there I was able to perform audio processing. The dataset for these songs were chosen from the list of Billboard top mainstream rock songs for each year in the timespans of 1980-1989 and 2010-2019. From the list of these top songs, I chose the songs that were listed as number one for at least 5 weeks. 
    \nFeatures collected were tempo, chromagram, key, presence of key change, sections, and chords. Each of these represents different aspects of music, including rhythm, harmony, and song structure.
    From each of these features, I generated summarized features of chroma entropy, rhythmic complexity, key stability, structural variability, harmonic variety, harmonic change rate, harmonic entropy, and major character.  Chroma entropy describes how evenly energy is distributed among the 12 chroma bins, which is calculated as the sum times the log of the mean chroma. Rhythmic complexity multiplies raw tempo and number of segments, key stability is 1 divided by the number of key changes where fewer key changes is more stable, and structural variability is the standard deviation of the section lengths. Harmonic variety measures entropy over the chord distribution, which takes the negative sum of the chord probabilities times the log of the chord probabilities. Harmonic character is between 0 and 1 for whether the segment keys are 1, all major or 0, all minor. These features are concatenated into a summary data frame for graphing and analysis. 
    \nThis is an example of data that is added into the feature list.

''')
feat_path = Path(__file__).resolve().parents[1] / "song_feature_data.pkl"
#with open("/Users/rachelfox/Downloads/song_feature_data.pkl", "rb") as f:
with open(feat_path, "rb") as f:
    reference_data = pickle.load(f)
print(type(reference_data))
#st.dataframe(reference_data.head(10))
st.dataframe(reference_data.sample(n=10))

st.markdown('''
            
    This is an example of data that is created in the summary list of features.
            
''')

summ_path = Path(__file__).resolve().parents[1] / "song_summary_data.pkl"
#with open("/Users/rachelfox/Downloads/song_summary_data.pkl", "rb") as f:
with open(summ_path, "rb") as f:
    summary_df = pickle.load(f)
#st.dataframe(summary_df.head(10))
st.dataframe(summary_df.sample(n=10))
st.markdown(f"There are currently {summary_df.shape[0]} songs in this dataset.")

st.markdown('''
            
    This is what principal components created from the summary features look like.
            
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