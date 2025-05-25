import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="BioML App", layout="wide")
st.title("🧬 BioML - Ανάλυση Μοριακών Βιολογικών Δεδομένων")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Δεδομένα", "📊 PCA", "🧩 Clustering", "🤖 Classification", "👥 Ομάδα"])

@st.cache_data
def preprocess_data(df):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        st.warning("⚠️ Δεν βρέθηκαν αριθμητικές στήλες στο dataset.")
        return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    return scaled

with tab1:
    st.header("Φόρτωση Δεδομένων")
    uploaded_file = st.file_uploader("Ανέβασε αρχείο CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.apply(pd.to_numeric, errors='ignore')
        st.write("Προεπισκόπηση:")
        st.dataframe(df)
        st.write("Τύποι στηλών:", df.dtypes)
        st.session_state.df = df

with tab2:
    st.header("PCA Ανάλυση")
    if "df" in st.session_state:
        df = st.session_state.df
        scaled = preprocess_data(df)
        if scaled is None:
            st.stop()
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        fig, ax = plt.subplots()
        ax.scatter(pca_df["PC1"], pca_df["PC2"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
    else:
        st.info("Ανέβασε πρώτα δεδομένα στο tab 'Δεδομένα'.")

with tab3:
    st.header("KMeans Clustering")
    if "df" in st.session_state:
        df = st.session_state.df
        scaled = preprocess_data(df)
        if scaled is None:
            st.error("Δεν υπάρχουν αριθμητικά δεδομένα για clustering.")
            st.stop()
        n_clusters = st.slider("Αριθμός Clusters", 2, 10, 3)
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
    else:
        st.info("Ανέβασε πρώτα δεδομένα στο tab 'Δεδομένα'.")

with tab4:
    st.header("KNN Classification")
    if "df" in st.session_state:
        df = st.session_state.df
        if df.select_dtypes(include='number').empty:
            st.warning("⚠️ Δεν υπάρχουν αριθμητικά χαρακτηριστικά για classification.")
            st.stop()
        label_col = st.selectbox("Επέλεξε στήλη-στόχο (label):", df.columns)
        try:
            X = df.drop(columns=[label_col])
            st.write("Στήλες εισόδου μετά την αφαίρεση του label:", X.columns.tolist())
            X = X.select_dtypes(include='number')
            y = df[label_col]
            if X.empty:
                st.error("❌ Δεν υπάρχουν κατάλληλα αριθμητικά δεδομένα για training.")
                st.stop()
            test_size = st.slider("Test size (ποσοστό test)", 0.1, 0.5, 0.2)
            k = st.slider("Αριθμός K για KNN", 1, 15, 3)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.text("Αποτελέσματα Classification:")
            st.text(classification_report(y_test, y_pred))
        except Exception as e:
            st.error(f"Σφάλμα: {e}")
    else:
        st.info("Ανέβασε πρώτα δεδομένα στο tab 'Δεδομένα'.")

with tab5:
    st.header("Στοιχεία Ομάδας")
    st.markdown("""
**Ονοματεπώνυμα:**  
- Γιώργος Ιωάννου (inf2021006)  
- Γιώργος Χρυσοστόμου (inf2021004)  
- Αλέξανδρος Χριστοφόρου (inf2021007)

**GitHub:** [https://github.com/GeorgiosIoannou02/bio-ml-app](https://github.com/GeorgiosIoannou02/bio-ml-app)
""")