# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("listings.csv")
    return data

# EDA Functions
def perform_eda(data):
    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_data.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for correlation analysis.")

    st.subheader("Feature Distributions")
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        st.pyplot(plt)

    st.subheader("Outlier Detection")
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=data[column])
        plt.title(f"Outliers in {column}")
        st.pyplot(plt)

    st.subheader("Grouped Analysis")
    if 'group_column' in data.columns:  # Replace with your group column
        grouped = data.groupby('group_column').mean()  # Replace 'mean()' with relevant aggregation
        st.write(grouped)

# Preprocessing
def preprocess_data(data):
    data = data.dropna()  # Remove missing values (example, adjust as needed)
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # One-hot encoding

    X = data.drop("target_column", axis=1)  # Replace "target_column" with your actual target column
    y = data["target_column"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit App
st.title("Data Science Project")
data = load_data()

st.header("Exploratory Data Analysis (EDA)")
perform_eda(data)

st.header("Preprocessing and Model Training")
X_train, X_test, y_train, y_test = preprocess_data(data)
model = train_model(X_train, y_train)

st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
st.pyplot(plt)

st.header("Feature Importance")
feature_importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
st.write(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance")
st.pyplot(plt)
