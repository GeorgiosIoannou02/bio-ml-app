# BioML App 🧬

Αυτή είναι μια διαδραστική εφαρμογή που υλοποιήθηκε με Streamlit και έχει στόχο την ανάλυση δεδομένων μοριακής βιολογίας με χρήση τεχνικών Μηχανικής Μάθησης.

## 🔧 Λειτουργίες

- 📂 Ανέβασμα αρχείου CSV
- 📊 PCA Ανάλυση (με sklearn)
- 🧩 Clustering (KMeans)
- 🤖 Classification (K-Nearest Neighbors)
- 👥 Πληροφορίες ομάδας

## 🧪 Βιβλιοθήκες

- `streamlit`
- `pandas`
- `matplotlib`
- `scikit-learn`

## ▶️ Εκκίνηση τοπικά

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Εκκίνηση με Docker

```bash
docker build -t bio-ml-app .
docker run -p 8501:8501 bio-ml-app
```

## 📁 Δομή

```
bio-ml-app/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 👨‍💻 Ομάδα

- Γιώργος Ιωάννου inf2021006
- Γιώργος Χρυσοστόμου inf2021004
- Αλέξανδρος Χριστοφόρου inf2021007

