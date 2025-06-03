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

# conda activate /Users/rachelfox/opt/anaconda3/envs/medsam
# streamlit run Downloads/MusicWebApp/'Musicology 188 Site'.py

### functions
# MIR pipeline
# functions for the music information retrieval analysis where different songs can be inputted

st.set_page_config(page_title="About")

st.title("About")

st.header("Musicology 188 Project")
st.subheader("Analyzing harmonic attributes of rock songs in multiple time periods")

st.markdown('''
    Welcome to my Musicology 188 project site: Analyzing Harmonic Attributes of Rock Songs in Multiple Time Periods.
    \nThis project is created by Rachel Fox in Spring 2025. This is a music information retrieval project related to rock music.
            
    \nThis project uses computational analysis to analyze components from an imported MP3 file of a song, with components including signal analysis, feature extraction, beat tracking, and clustering. This can produce figures and detailed descriptions of what makes up the audio file. I will be using this as a computational exploration of music as well as a comparison of multiple popular songs from different time periods, to see how components of songs have evolved. I also have created a pipeline where the user would be able to input an mp3 file of the song of their choice and that incorporates information from the new song into the model to how this fits into the current dataset.
    In the field of digital musicology, there are different techniques and formats to analyze music, including music information retrieval, optical music recognition, music encoding, digital archives, and metadata processing. There are many digital archives that exist, such as the global jukebox or archives for individual styles and regions of music, and more recently the individual songs are being analyzed for specific features. With the use of machine learning and audio recognition tools, it becomes more feasible to analyze large amounts of songs in a consistent manner. Thus, there are many areas for contributions in music information retrieval. 
    I am examining rock music, which has not been extensively analyzed using computational techniques, and adding a comparison of two time points of popular music will increase understanding of how rock music has evolved over time.  

''')
