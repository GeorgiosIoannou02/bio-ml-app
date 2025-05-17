from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    numeric_df = df.select_dtypes(include='number')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    return scaled