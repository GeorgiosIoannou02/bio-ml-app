import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
from .preprocessing import preprocess_data

def run_clustering_plot(df):
    n_clusters = st.slider("Αριθμός Clusters", 2, 10, 3)
    scaled = preprocess_data(df)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(scaled)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], c=labels, cmap="Set1")
    ax.set_title("KMeans Clustering (PCA)")
    st.pyplot(fig)