# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("listings.csv")
    return data

# EDA Functions
def perform_eda(data):
    st.expander("Summary Statistics")
    st.write(data.describe())

    st.expander("Missing Values")
    st.write(data.isnull().sum())

    st.expander("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_data.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for correlation analysis.")

    st.expander("Feature Distributions")
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        st.pyplot(plt)

    st.expander("Outlier Detection")
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=data[column])
        plt.title(f"Outliers in {column}")
        st.pyplot(plt)

    st.expander("Pairwise Relationships")
    numeric_data = data.select_dtypes(include=[np.number])  # Select numeric columns
    if numeric_data.shape[1] > 1:
        sns.pairplot(numeric_data)
        st.pyplot(plt.gcf())
    else:
        st.write("Not enough numeric features for pairplot.")

    st.expander("Price Analysis")
    if "price" in data.columns:
        # Price Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(data["price"], kde=True, bins=30, color="blue")
        plt.title("Distribution of Price")
        st.pyplot(plt)

        # Price Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data["price"], color="cyan")
        plt.title("Boxplot of Price")
        st.pyplot(plt)

        # Price by Category (if categorical columns exist)
        categorical_cols = data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[col], y=data["price"])
            plt.title(f"Price Distribution by {col}")
            plt.xticks(rotation=45)
            st.pyplot(plt)
    else:
        st.write("Price column is not available for analysis.")

# Preprocessing
def preprocess_data(data):
    st.write("Performing data preprocessing...")
    data = data.dropna()  # Remove missing values
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # One-hot encoding

    X = data.drop("price", axis=1)  # Replace "price" with your actual numeric target column
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train Regression Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=10, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Streamlit App
st.title("Regression Analysis Project")
data = load_data()

st.subheader("Exploratory Data Analysis (EDA)")
perform_eda(data)

X_train, X_test, y_train, y_test = preprocess_data(data)
model = train_model(X_train, y_train)

# Model Evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Feature Importance
st.subheader("Feature Importance")
feature_importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
st.write(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance")
st.pyplot(plt)
