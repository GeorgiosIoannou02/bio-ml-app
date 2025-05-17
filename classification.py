import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from .preprocessing import preprocess_data

def run_classification(df):
    label_col = st.selectbox("Επέλεξε στήλη-στόχο (label):", df.columns)
    try:
        X = preprocess_data(df.drop(columns=[label_col]))
        y = df[label_col]

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