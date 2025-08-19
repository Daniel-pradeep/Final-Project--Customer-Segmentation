import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="Simple Customer Segmentation", layout="centered")
st.title("🧩 Simple Customer Segmentation (K-Means)")

st.write("Upload your Mall Customers CSV or use the default path.")

uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Expected columns: Gender, Age, Annual Income (k$), Spending Score (1-100)")
default_path = st.text_input("...or enter a local path", "/mnt/data/Final Project 2 Mall Customer Dataset.csv")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif default_path:
    try:
        df = pd.read_csv(default_path)
    except Exception as e:
        st.warning("Could not read the file. Please upload a CSV or fix the path.")

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head())

    # Encode Gender
    df_proc = df.copy()
    if 'Gender' in df_proc.columns:
        df_proc['Gender'] = df_proc['Gender'].map({'Female':0, 'Male':1})

    # Feature selection
    default_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    available = [c for c in default_features if c in df_proc.columns]
    features = st.multiselect("Choose features", available, default=available)

    if len(features) < 2:
        st.info("Please select at least two features.")
    else:
        X = df_proc[features].values

        # Scale
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # K selection
        sil_scores = []
        K = range(2, 11)

        for k in K:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)
            sil_scores.append(silhouette_score(Xs, labels))
        
        k = list(K)[int(np.argmax(sil_scores))]

        if st.button("Run Clustering"):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)

            # Metrics
            sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else np.nan
            st.write("**Metrics**")
            st.json({"silhouette": float(sil), "inertia": float(km.inertia_)})

            # PCA 2D
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(Xs)
            fig = px.scatter(x=X2[:,0], y=X2[:,1], color=labels.astype(str),
                             labels={"x":"PC1", "y":"PC2", "color":"cluster"},
                             title="PCA (2D) — Clusters")
            st.plotly_chart(fig, use_container_width=True)

            # Labeled data + download
            labeled = df.copy()
            labeled['cluster'] = labels
            st.subheader("Clustered Data (first rows)")
            st.dataframe(labeled.head())

            st.download_button("⬇️ Download Labeled CSV",
                               data=labeled.to_csv(index=False).encode('utf-8'),
                               file_name="clustered_customers.csv",
                               mime="text/csv")
            
    # --- User Input for Prediction ---
    st.subheader("🔮 Predict Cluster for a New Customer")

    # Input fields
    age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
    income = st.number_input("Enter Annual Income (k$)", min_value=0, max_value=200, value=50)
    score = st.number_input("Enter Spending Score (1-100)", min_value=1, max_value=100, value=50)

    if st.button("Predict Cluster"):
        # Prepare input as DataFrame
        user_data = pd.DataFrame([[age, income, score]], columns=features)

        # Scale with the same scaler
        user_scaled = scaler.transform(user_data)

        # Predict cluster
        user_cluster = km.predict(user_scaled)[0]
        st.success(f"✅ The customer belongs to **Cluster {user_cluster}**")

        # --- Simple Recommendations ---
        st.subheader("📌 Recommendation")
        if user_cluster == 0:
            st.write("🛒 Cluster 0 → Aged and Average Income, Average spending. Suggest discounted priced products.")
        elif user_cluster == 1:
            st.write("💳 Cluster 1 → Young and Moderate income, Moderate spending. Target with latest trending offers.")
        elif user_cluster == 2:
            st.write("🧑 Cluster 2 → Middle age and high income and very less spending. Approch with highly discounted products.")
        elif user_cluster == 3:
            st.write("👨‍👩‍👧 Cluster 3 → Young and high income, high spending. Recommend luxury + exclusive deals.")
        elif user_cluster == 4:
            st.write("📉 Cluster 4 → Young and high spending despite income. Enable with the premium offers.")
        elif user_cluster == 5:
            st.write("🚀 Cluster 5 → Middle age and less income and less spending. Approch with highly discounted products.")
        else:
            st.write("ℹ️ No recommendation available.")

else:
    st.info("Upload a CSV or provide a path to continue.")


