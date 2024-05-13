import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
df=sns.load_dataset("iris")

# Split data into features and target
X = df.drop('species', axis=1) # Features (sepal length, sepal width, petal length, petal width)
y = df['species'] # Target (species)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a k-nearest neighbors classifier
k = 3 # You can adjust this value
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Function to predict species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(new_data)
    return prediction[0]

# Streamlit app
st.title("Iris Species Prediction")

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fbeautiful-flowers&psig=AOvVaw3q9lMhNYy3VblUKQK_iZZl&ust=1715362617162000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCOjkoe2NgYYDFQAAAAAdAAAAABAE") center;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input
sepal_length = st.number_input("Enter sepal length:")
sepal_width = st.number_input("Enter sepal width:")
petal_length = st.number_input("Enter petal length:")
petal_width = st.number_input("Enter petal width:")

# Prediction
if st.button("Predict"):
    predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.subheader(f"Predicted species: {predicted_species}")