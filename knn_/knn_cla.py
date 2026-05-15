import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🌸 Iris Classification using KNN")

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names
feature_names = iris.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
k = st.sidebar.slider("Choose K (Neighbors)", 1, 15, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {acc:.2f}")


# Sidebar Inputs
st.sidebar.header("Input Flower Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict Flower Type"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"🌼 Predicted Class: {class_names[prediction[0]]}")