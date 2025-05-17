import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st
from .preprocessing import preprocess_data

def run_pca_plot(df):
    scaled = preprocess_data(df)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    fig, ax = plt.subplots()
    ax.scatter(pca_df["PC1"], pca_df["PC2"])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)