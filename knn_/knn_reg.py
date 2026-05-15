import streamlit as st
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Title
st.title("🧠 Diabetes Prediction using KNN Regression")

# Load dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

feature_names = diabetes.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN Regressor
k = st.sidebar.slider("Choose K (Neighbors)", 1, 20, 5)

model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Patient Details")

inputs = []

for feature in feature_names:
    val = st.sidebar.slider(
        feature,
        float(X[:, feature_names.index(feature)].min()),
        float(X[:, feature_names.index(feature)].max()),
        float(X[:, feature_names.index(feature)].mean())
    )
    inputs.append(val)

# Prediction
if st.button("Predict Disease Progression"):

    input_array = np.array([inputs])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    st.success(f"🧠 Predicted Disease Progression: {prediction[0]:.2f}")