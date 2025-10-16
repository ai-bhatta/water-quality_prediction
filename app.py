# -*- coding: utf-8 -*-
"""app.py — Water Quality Prediction & Explainable AI"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Streamlit page setup
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")

st.title("💧 Water Quality Clustering and Explainability App")
st.write(
    "Upload your water quality dataset (CSV or Excel) to visualize clusters "
    "and understand feature importance using Explainable AI (SHAP)."
)

# File upload
uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())

    # Load pre-trained model components
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    # Select numeric data only
    df_num = df.select_dtypes(include=[np.number])
    df_num.fillna(df_num.mean(), inplace=True)

    # --- Safe scaling fix ---
    try:
        expected_features = scaler.feature_names_in_
        # Align columns with scaler expectations
        df_num = df_num[[col for col in expected_features if col in df_num.columns]]

        # Warn if columns missing
        missing_cols = set(expected_features) - set(df_num.columns)
        if missing_cols:
            st.warning(
                f"⚠️ Missing columns from uploaded dataset: {', '.join(missing_cols)}. "
                "These will be ignored for clustering."
            )

        X_scaled = scaler.transform(df_num)
    except Exception as e:
        st.error(f"Error during data scaling: {e}")
        st.stop()

    # PCA transform
    X_pca = pca.transform(X_scaled)

    # Predict clusters
    clusters = kmeans.predict(X_scaled)
    df["Cluster"] = clusters

    st.subheader("🧠 Clustered Results")
    st.write(df.head())

    # Cluster visualization
    st.subheader("🔍 Cluster Visualization (PCA)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means Clustering Visualization")
    st.pyplot(fig)

    # Explainable AI (SHAP)
    st.subheader("💡 Explainable AI (SHAP)")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, clusters)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    st.write("Top parameters influencing cluster formation:")
    shap.summary_plot(
        shap_values, X_scaled, feature_names=df_num.columns, plot_type="bar", show=False
    )
    st.pyplot(bbox_inches="tight")

else:
    st.info("👆 Please upload a dataset to start.")
